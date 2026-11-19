"""Fit the training-only vocabulary, audit all reports, and exercise the complete model on real CT."""
import copy
import json
import os
import time
from collections import Counter
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG",":4096:8")
import numpy as np
import torch
from torch.nn import functional as F
from fetch_four import sha256, write_json
from full_report_model import FullReportCodec, FullReportUNet, collate
from replay_controls import read_reports, fingerprint
from report_strata import describe_report
from train_partial_baselines import allowed, load_cases, tensors, SPLIT


def pieces(row):
    return row["Findings_EN"],row["Impressions_EN"]


def run():
    os.umask(0o077)
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot-vocab", action="store_true", help="Fit only on the 6 actual training T and eligible U.")
    args=parser.parse_args()
    out=Path(__file__).resolve().parent
    if args.pilot_vocab:
        frozen=json.loads(Path("phase1/baselines/partial_v1/input_lock.private.json").read_text())
        t={r["volume"] for r in frozen["roles"]["T"]}
        if len(t)!=6 or not t<=allowed(SPLIT/"T.csv"):
            raise ValueError("Unexpected pilot T roster")
    else:
        t=allowed(SPLIT/"T.csv")
    allowed_ids=t|allowed(SPLIT/"U_train.csv")
    source=read_reports(allowed_ids)
    docs={n:pieces(row) for n,row in source.items()}
    codec=FullReportCodec.fit(docs.values())
    write_json(out/"vocabulary.private.json",{"words":codec.words,"chunk_size":codec.chunk_tokens})
    counts,states,correct=Counter(),Counter(),0
    encoded={}
    for key,fields in docs.items():
        chunks=codec.encode(*fields)
        decoded=codec.decode(chunks)
        if decoded!=fields or not chunks:
            raise ValueError("Nonreversible nonempty clinical report")
        first=describe_report(*fields)
        if describe_report(*decoded)!=first:
            raise ValueError("Clinical lexical cues lost in full-report representation")
        encoded[key]=chunks
        states[first["state"]]+=1
        counts[len(chunks)]+=1
        correct+=1
    # Check an unseen Unicode term and a byte sequence spanning chunk boundaries.
    audit_example=("右侧 "+("x"*163)+";","未见胸腔积液。")
    assert codec.decode(codec.encode(*audit_example))==audit_example
    torch.manual_seed(891)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cuda.matmul.allow_tf32=False
    model=FullReportUNet(codec.vocab_size).cuda().eval()
    qc=Path("phase1/data/T_partial/provisional_qc.private.json")
    cases, cohort=load_cases(qc,64)
    inputs={r:[v["volume"] for v in cohort["roles"][r]] for r in ["T","U"]}
    ct,fg,_,_=tensors(cases["T"][0],891,64,False)
    u_strong,_,_,u_weak=tensors(cases["U"][0],892,64,True)
    key=inputs["T"][0]
    pair=docs[key]
    with torch.inference_mode():
        null=model(ct,None)
        null_explicit=model(ct,collate([()],"cuda"))
        exact=model(ct,collate([encoded[key]],"cuda"))
        duplicate=model(ct,collate([codec.encode(*pair)],"cuda"))
        # Identical long prefix with one changed word deliberately placed after truncation.
        prefix=pair[1]+" "+("Stable findings. "*24)
        tail1=codec.encode(pair[0],prefix+"Left pleural effusion is present.")
        tail2=codec.encode(pair[0],prefix+"Right pleural effusion is present.")
        a=model(ct,collate([tail1],"cuda"))
        b=model(ct,collate([tail2],"cuda"))
        together=model(torch.cat((ct,ct)),collate([encoded[key],tail1],"cuda"))
        sep=torch.cat((exact,a))
        batch_error=(together-sep).abs().max().item()
        tail_gap=(a.sigmoid()-b.sigmoid()).abs().mean().item()
        if not torch.equal(null,null_explicit) or not torch.equal(exact,duplicate) or batch_error>1e-4 or not tail_gap>0:
            raise ValueError(f"Decoder consistency failed: batching={batch_error}, tail={tail_gap}")
    model.train()
    teacher=copy.deepcopy(model).eval().requires_grad_(False)
    opt=torch.optim.Adam(model.parameters(),lr=1e-4)
    initial=fingerprint(model.state_dict())
    rows=[]
    for i in range(4):
        torch.cuda.synchronize()
        tick=time.perf_counter()
        x,positive,_,_=tensors(cases["T"][i],891+i,64,True)
        u_strong,_,_,u_weak=tensors(cases["U"][i],910+i,64,True)
        t_doc=collate([encoded[inputs["T"][i]]],"cuda") if i%2==0 else None
        u_doc=collate([encoded[inputs["U"][i]]],"cuda")
        with torch.no_grad():
            target=teacher(u_weak,u_doc).sigmoid()
        pos_loss=F.softplus(-model(x,t_doc)[positive]).mean()
        consistency=F.mse_loss(model(u_strong).sigmoid(),target)
        loss=pos_loss+consistency
        opt.zero_grad(set_to_none=True)
        loss.backward()
        g=torch.nn.utils.clip_grad_norm_(model.parameters(),float("inf"),error_if_nonfinite=True)
        word_g=model.text.embedding.weight.grad.abs().max().item()
        if g<=0 or word_g<=0:
            raise ValueError("Missing gradient")
        opt.step()
        with torch.no_grad():
            for avg,online in zip(teacher.parameters(),model.parameters(),strict=True):
                avg.lerp_(online,.01)
        torch.cuda.synchronize()
        rows.append({"loss":loss.item(),"max_word_embedding_grad":word_g,"step_s":time.perf_counter()-tick})
    final=fingerprint(model.state_dict())
    assert final!=initial
    checkpoint=out/"fulltext_engineering.pt"
    model.eval()
    with torch.no_grad(): pred=model(ct).detach().clone()
    torch.save({"student":model.state_dict(),"teacher":teacher.state_dict(),"optimizer":opt.state_dict()},checkpoint)
    reloaded=FullReportUNet(codec.vocab_size).cuda().eval()
    reloaded.load_state_dict(torch.load(checkpoint,weights_only=True)["student"])
    with torch.no_grad(): assert torch.equal(pred,reloaded(ct))
    write_json(out/"fulltext_validation.json",{
        "training_reports":len(docs),"T_reports_for_vocabulary":len(t),
        "training_vocabulary_roster_sha256":__import__("hashlib").sha256("\n".join(sorted(allowed_ids)).encode()).hexdigest(),
        "exact_roundtrip_after_normalizing_whitespace_and_empty_sections":correct,
        "unchanged_all_rule_features":correct,"preserved_polarity_counts":dict(states),
        "vocab_size_including_bytes_and_codes":codec.vocab_size,"vocabulary_sha256":sha256(out/"vocabulary.private.json"),
        "chunk_count_range":[min(counts),max(counts)],"max_tokens_per_chunk":codec.chunk_tokens,
        "tokens_dropped":0,"padded_batch_max_abs_logit_difference":batch_error,
        "duplicate_text_and_learned_empty_token_bitwise_checks":True,
        "controlled_last_sentence_laterality_change_mean_abs_probability_delta":tail_gap,
        "real_seed":891,"real_optimizer_steps":len(rows),"observed_steps":rows,
        "peak_allocated_mib":torch.cuda.max_memory_allocated()/2**20,"checkpoint_reload_equal":True,
        "weights_sha256":sha256(checkpoint),"source_sha256":{p.name:sha256(p) for p in (out/"full_report_model.py",out/"check_full_report.py")},
        "scope":"Only T/U training. Tail perturbation is a sensitivity check. Four updates verify data flow; they cannot estimate clinical performance."})
    print("full text verified",len(docs),len(rows),batch_error,tail_gap,flush=True)


if __name__=="__main__":
    run()
