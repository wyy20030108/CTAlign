"""Exchange source-verbatim clinical sentences or spans, with an explicit audit of their origin."""
import os,sys
from pathlib import Path
import numpy as np
import torch
from swap_volume import maps,auc,read,out,read_reports,sha256,write_json,TEXT,FullReportCodec,FullReportUNet

# Anchors refer to the licensed CT-RATE report. Only actual source substrings are fed to the teacher.
SENTENCES={
 "S008":("It was learned that the patient was being followed up for lung cancer","In addition, there is an irregularly circumscribed nodule adjacent"),
 "S021":("In the evaluation of the lung parenchyma, pneumonic infiltration","It is accompanied by mild septal thickening"),
 "S028":("The lower lobe of the right lung is total atelectatic","Left lung lower lobe is totally atelectatic except"),
 "S033":("When examined in the lung parenchyma window; In the middle lobe of the right lung, patchy","Mild paraseptal centrilobular emphysematous changes"),
}
SPANS={
 "S004":("left lung lower lobe posterobasal segment","upper lobe inferior lingular segment"),
 "S009":("right lung middle lobe medial segment","left lung upper lobe inferior lingular segment"),
 "S012":("Peripheral and central consolidations, more prominent in the lower lobe of the right lung","ground glass areas accompanying the consolidations"),
 "S020":("right lung lower lobe superior","lower lobe posterobasal segment"),
 "S029":("right lung middle lobe","left lung inferior lingular segment"),
 "S036":("middle lobe of the right lung","lingula of the left lung"),
 "S037":("ground glass areas and millimetric centriacinar nodules in the left lung lower lobe anteromediobasal segment","nodular ground glass areas in the pleural area in the right lung middle lobe medial segment anterior section"),
 "S041":("6 mm in the right lung lower lobe laterobasal segment","5 mm in the left lower lobe"),
 "S044":("right lung lower lobe laterobasal segment","left lung upper lobe apicoposterior segment"),
 "S045":("right lung lower lobe posterobasal","left lung lower lobe laterobasal"),
 "S048":("left lung upper lobe lingular segment inferior subsegment","right lung upper lobe medial segment"),
}

def main(role):
    import re
    from swap_data import digest
    torch.set_num_threads(2);torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
    dest=out/"clinical";dest.mkdir(exist_ok=True)
    rows=[r for r in read(out/"index.private.json")["pairs"] if r["role"]==role and r["code"] in (SPANS|SENTENCES)]
    docs=read_reports({r["name"] for r in rows});prompts={}
    for r in rows:
        c=r["code"]; report=docs[r["name"]]
        if c in SENTENCES:
            ss=list(dict.fromkeys(s.strip() for text in report.values() for s in re.split(r'(?<=[.!?])\s+|[\r\n]+',text) if s.strip()))
            pair=[next(s for s in ss if s.startswith(anchor)) for anchor in SENTENCES[c]]
        else:pair=list(SPANS[c])
        assert all(any(q in v for v in report.values()) for q in pair) and pair[0]!=pair[1]
        prompts[c]=pair
    write_json(dest/(role+".private.json"),prompts)
    v=read(TEXT/"vocabulary.private.json");codec=FullReportCodec(v["words"],v["chunk_size"])
    for seed in (891,1901,2707):
        base=Path("phase2/bounded")/str(seed);ck=base/"paired.pt";assert sha256(ck)==read(base/"paired.json")["checkpoint_sha256"]
        net=FullReportUNet(codec.vocab_size).cuda().eval();net.load_state_dict(torch.load(ck,map_location="cpu",weights_only=True)["teacher"])
        for r in rows:
            c=r["code"]; file=dest/f"{c}_{seed}.json"; pair=prompts[c]
            if file.exists(): assert read(file)["script"]==sha256(Path(__file__));continue
            pack=out/(c+".npz");prior=out/"prior"/(c+".npz")
            assert sha256(pack)==read(out/(c+".json"))["sha256"] and sha256(prior)==read(out/"prior"/(role+".json"))[c]["file_sha256"]
            with np.load(pack) as v,np.load(prior) as m:
                ct=v["ct"];x,y=v["masks"];a=x&~y;b=y&~x;assert np.array_equal(v["affine"],m["affine"])
                field,k=maps(net,codec,ct,m["margin"],pair,a,b)
            (ta,ma),(tb,mb)=[field[q] for q in pair];sel=a[a|b]
            correct=(ta[sel].mean()+tb[~sel].mean())/2;swap=(tb[sel].mean()+ta[~sel].mean())/2
            write_json(file,dict(code=c,seed=seed,role=role,kind="sentence" if c in SENTENCES else "span",k_voxels=k,
                   correct=float(correct),swapped=float(swap),delta=float(correct-swap),auc=auc((ma-mb)[sel],(ma-mb)[~sel]),
                   source_substrings_verified=True,source_hashes=[digest(q) for q in pair],script=sha256(Path(__file__))))
            print(c,seed,flush=True)
if __name__=="__main__":
    os.umask(0o077); main(sys.argv[1])
