"""Exchange two actual findings on the same independent scan."""
from argparse import ArgumentParser
from pathlib import Path
import os
import numpy as np
import torch
from thesis_dual import OUT,LOCK
from thesis_evaluate import EVAL,RESULTS,train_file,compare_visuals,donors,SEEDS
from thesis_cohort import MANIFEST,PRIMARY
from thesis_lobe_model import LobeStore
from thesis_model import FindingNet,DEVICE,volume_logits,segment
from finding_train import configure
from frozen_priors import read,sha,write
from diag_localize import seal

PAIR_LOCK=OUT/"paired_evaluation_lock.json"
def freeze():
    assert read(EVAL)["manifest"]==sha(MANIFEST)
    entries=[]
    for source in read(LOCK):
        path=OUT/(source["code"]+".json")
        row=read(path)
        if row["status"]=="eligible":
            entries.append(dict(code=source["code"],mask_sha=sha(path.with_suffix(".npz")),qc_sha=sha(path)))
    seal(PAIR_LOCK,dict(schema="Fixed before scoring either dual condition; true pair, not synthetic negative",
                        manifest=sha(MANIFEST),main=sha(EVAL),pairs=entries,sources=sha(LOCK),code=sha(Path(__file__))))
    print("Dual-reference patients",len(entries),flush=True)
def run(seed,arm):
    p=read(PAIR_LOCK)
    assert p["code"]==sha(Path(__file__)) and p["sources"]==sha(LOCK) and p["main"]==sha(EVAL)
    cohorts=read(MANIFEST)
    source={r["code"]:r for r in read(LOCK)}
    main={r["code"]:r for r in cohorts["test"]}
    model_dir,details,_=train_file(PRIMARY,seed,arm)
    sealed=read(EVAL)["arms"][PRIMARY][arm]
    assert next(q["model"] for q in sealed["checkpoints"] if q["seed"]==seed)==sha(model_dir/"model.pt")
    configure(seed);net=FindingNet().to(DEVICE).eval()
    net.load_state_dict(torch.load(model_dir/"model.pt",map_location=DEVICE,weights_only=True))
    store=LobeStore(limit=2)
    out=OUT/"scores"/str(seed)/arm;out.mkdir(exist_ok=True,parents=True)
    summary=[]
    for row in p["pairs"]:
        c=row["code"];f=out/(c+".json")
        if f.exists():
            old=read(f);assert old["lock"]==sha(PAIR_LOCK) and sha(f.with_suffix(".npz"))==old["sha"]
            summary.append(old);continue
        x=store.get(main[c],True)
        y2file=OUT/(c+".npz")
        assert sha(y2file)==row["mask_sha"]
        with np.load(y2file) as a:y2=a["mask"]
        summary_file=RESULTS/PRIMARY/str(seed)/arm/(c+".json")
        assert source[c]["primary"]==main[c]["query"]
        base=summary_file.with_suffix(".npz")
        assert sha(base)==read(summary_file)["prediction_sha"]
        with np.load(base) as z:one=np.unpackbits(z["correct"],count=x.truth.size).reshape(x.truth.shape).astype(bool)
        logits,windows=volume_logits(net,x,source[c]["alternative"])
        two=segment(logits,float(sealed["chosen"]),details["cap_ml"])
        matrix=[[compare_visuals(a,b) for b in (x.truth,y2)] for a in (one,two)]
        np.savez_compressed(f.with_suffix(".npz"),shape=x.ct.shape,
                            A=np.packbits(one.ravel()),B=np.packbits(two.ravel()))
        record=dict(code=c,lock=sha(PAIR_LOCK),sha=sha(f.with_suffix(".npz")),matrix=matrix,second_tiles=windows)
        write(f,record);summary.append(record)
        print("dual",seed,arm,c,[[round(k["dice"],5) for k in r] for r in matrix],flush=True)
    write(out/"finished.json",dict(arm=arm,seed=seed,lock=sha(PAIR_LOCK),rows=summary))
if __name__=="__main__":
    os.umask(0o077)
    parser=ArgumentParser();parser.add_argument("stage",choices=("lock","run"))
    parser.add_argument("--seed",type=int,default=891);parser.add_argument("--arms",default="BDF")
    a=parser.parse_args()
    if a.stage=="lock":freeze()
    else:
        assert a.seed in SEEDS and set(a.arms)<=set("BDF")
        for arm in a.arms:run(a.seed,arm)

