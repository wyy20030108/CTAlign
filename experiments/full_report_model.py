"""Lossless two-section report codec and sentence-conditioned CT decoder."""
import re
from collections import Counter
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from volume_benchmark import CHANNELS, conv_block


PIECES = re.compile(r'\s+|\w+|[^\w\s]', re.UNICODE)
# Split after punctuation at a word boundary, and after lines. Keep delimiters verbatim.
BREAKS = re.compile(r'[.!?;]+(?=\s|$)|\r\n|\r|\n')
PAD, NULL, FIRST_WORD = 0, 257, 258  # 1..256 are the exact UTF-8 byte values + 1


def sentences(section):
    start = 0
    for boundary in BREAKS.finditer(section):
        yield section[start:boundary.end()]
        start = boundary.end()
    if start < len(section):
        yield section[start:]


@dataclass(frozen=True)
class Chunk:
    section: int
    sentence: int
    part: int
    ids: tuple


class FullReportCodec:
    def __init__(self, words, chunk_tokens=48):
        if chunk_tokens < 1 or len(words) != len(set(words)):
            raise ValueError('Invalid chunk length or duplicate vocabulary entry')
        self.words = tuple(words)
        self.word_ids = {s: i+FIRST_WORD for i,s in enumerate(words)}
        self.chunk_tokens = chunk_tokens

    @classmethod
    def fit(cls, pairs, max_words=4096, min_count=2, chunk_tokens=48):
        counts = Counter(piece for pair in pairs for text in pair for sentence in sentences(text)
                         for piece in PIECES.findall(sentence))
        words = sorted((p for p in counts if counts[p] >= min_count),key=lambda p:(-counts[p],p))[:max_words]
        return cls(words,chunk_tokens)

    @property
    def vocab_size(self):
        return FIRST_WORD+len(self.words)

    def encode(self, findings, impression):
        chunks = []
        for section,text in enumerate((findings,impression)):
            if not isinstance(text,str):
                raise TypeError('Pass both report fields as strings, using an empty string for a missing field')
            for idx,sent in enumerate(sentences(text)):
                ids = []
                for piece in PIECES.findall(sent):
                    word = self.word_ids.get(piece)
                    if word is None:
                        ids.extend(b+1 for b in piece.encode('utf8'))
                    else:
                        ids.append(word)
                for start in range(0,len(ids),self.chunk_tokens):
                    chunks.append(Chunk(section,idx,start//self.chunk_tokens,tuple(ids[start:start+self.chunk_tokens])))
        return tuple(chunks)

    def decode(self,chunks):
        sections = [bytearray(),bytearray()]
        for chunk in chunks:
            for ix in chunk.ids:
                if 1<=ix<=256:
                    sections[chunk.section].append(ix-1)
                elif FIRST_WORD<=ix<self.vocab_size:
                    sections[chunk.section].extend(self.words[ix-FIRST_WORD].encode('utf8'))
                else:
                    raise ValueError('Invalid report token')
        return tuple(b.decode('utf8') for b in sections)


@dataclass
class ReportBatch:
    ids: torch.Tensor                 # [sum(valid chunks), longest token sequence]
    token_lengths: torch.Tensor       # CPU lengths for sequence packing
    section: torch.Tensor
    continuation: torch.Tensor
    report_lengths: torch.Tensor      # CPU lengths, with one sentinel chunk for absent text
    present: torch.Tensor


def collate(encoded,device):
    if not encoded:
        raise ValueError('Expected a nonempty minibatch of reports')
    placeholder = Chunk(2,0,0,(NULL,))
    present = [bool(doc) for doc in encoded]
    docs = [doc if doc else (placeholder,) for doc in encoded]
    flat = [c for doc in docs for c in doc]
    lengths = torch.tensor([len(c.ids) for c in flat],dtype=torch.long)
    ids = torch.full((len(flat),int(lengths.max())),PAD,dtype=torch.long)
    for i,c in enumerate(flat):
        ids[i,:len(c.ids)] = torch.tensor(c.ids)
    return ReportBatch(ids.to(device), lengths, torch.tensor([c.section for c in flat],device=device),
        torch.tensor([int(c.part>0) for c in flat],device=device), torch.tensor([len(doc) for doc in docs]),
        torch.tensor(present,dtype=torch.bool,device=device))


class SentenceEncoder(nn.Module):
    def __init__(self,vocab_size,width):
        super().__init__()
        self.embedding=nn.Embedding(vocab_size,width,padding_idx=PAD)
        self.token_gru=nn.GRU(width,width//2,batch_first=True,bidirectional=True)
        self.section=nn.Embedding(3,width)
        self.continuation=nn.Embedding(2,width)
        self.chunk_gru=nn.GRU(width,width//2,batch_first=True,bidirectional=True)

    def forward(self,batch):
        packed=pack_padded_sequence(self.embedding(batch.ids),batch.token_lengths,batch_first=True,enforce_sorted=False)
        _,ends=self.token_gru(packed)
        chunk=torch.cat((ends[0],ends[1]),dim=-1)+self.section(batch.section)+self.continuation(batch.continuation)
        grouped=nn.utils.rnn.pad_sequence(chunk.split(batch.report_lengths.tolist()),batch_first=True)
        packed=pack_padded_sequence(grouped,batch.report_lengths,batch_first=True,enforce_sorted=False)
        contextual,_=self.chunk_gru(packed)
        seq,_=pad_packed_sequence(contextual,batch_first=True)
        key_padding=torch.arange(seq.shape[1],device=seq.device)[None,:]>=batch.report_lengths.to(seq.device)[:,None]
        return seq,key_padding


class FullReportUNet(nn.Module):
    def __init__(self,vocab_size):
        super().__init__()
        self.encoder=nn.ModuleList(conv_block(i,o) for i,o in zip((1,)+CHANNELS,CHANNELS))
        self.up=nn.ModuleList(nn.ConvTranspose3d(i,o,2,stride=2) for i,o in zip(CHANNELS[:0:-1],CHANNELS[-2::-1]))
        self.decoder=nn.ModuleList(conv_block(c*2,c) for c in CHANNELS[-2::-1])
        width=CHANNELS[-2]
        self.text=SentenceEncoder(vocab_size,width)
        self.query_norm=nn.LayerNorm(width)
        self.attention=nn.MultiheadAttention(width,4,batch_first=True,dropout=0.)
        self.output=nn.Conv3d(CHANNELS[0],1,1)

    def forward(self,ct,report=None):
        if report is None:
            report=collate([()] * ct.shape[0],ct.device)
        if len(report.present)!=ct.shape[0]:
            raise ValueError('One report per CT is required')
        skips=[]
        x=ct
        for level,block in enumerate(self.encoder):
            x=block(x)
            if level<len(self.encoder)-1:
                skips.append(x)
                x=F.max_pool3d(x,2)
        for level,(up,block,skip) in enumerate(zip(self.up,self.decoder,reversed(skips))):
            x=block(torch.cat((up(x),skip),dim=1))
            if level==0:
                # Fusion after upsampling and the first skip, i.e. the decoder's /4 feature grid.
                memory,key_padding=self.text(report)
                grid=x.shape
                q=x.flatten(2).transpose(1,2)
                correction,_=self.attention(self.query_norm(q),memory,memory,key_padding_mask=key_padding,need_weights=False)
                x=(q+correction).transpose(1,2).reshape(grid)
        return self.output(x)
