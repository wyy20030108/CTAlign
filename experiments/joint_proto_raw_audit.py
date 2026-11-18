"""Freeze the prototype-only control under the selected raw local readout."""
from argparse import ArgumentParser
from pathlib import Path
import os

import numpy as np
import torch

from diag_localize import seal
from finding_train import configure
from frozen_priors import read,sha,write
from joint_external_nodule import OUT as DETECTOR_RECEIPTS
from joint_external_proto_only import OUT as PROTO
import joint_external_refine as local
from joint_metrics import summary
from joint_nodule_run import COHORT,DEST
import thesis_model as image
from thesis_component_fast import segment
from thesis_lobe_model import LobeStore
from thesis_ssl import SEEDS


OUT=DEST/"external_proto_raw"
MAIN=local.ROOT/"external_lock.json"


def network(seed):
    model=PROTO/"B"/str(seed)/"R"/"model.pt"
    proof=read(model.with_name("development.json"))
    assert sha(model)==proof["model"]
    local.install()
    image.segment=segment
    configure(seed)
    net=image.FindingNet().to(image.DEVICE).eval()
    net.load_state_dict(torch.load(model,map_location=image.DEVICE,weights_only=True))
    return net,proof


def dev(seed):
    OUT.mkdir(exist_ok=True,parents=True)
    net,proof=network(seed)
    source=read(COHORT)["dev"]
    store=LobeStore(limit=2)
    scores={}
    for row in source:
        result,_=image.case_eval(net,row,store.get(row,True),row["query"],proof["cap_ml"])
        scores[row["code"]]=result
    write(OUT/(f"dev_{seed}.json"),dict(scores=scores,model=proof["model"],cap=proof["cap_ml"],code=sha(Path(__file__))))


def lock():
    principal=read(MAIN)
    assert read(DEST/"validation_selection.json")["chosen"]==local.ROOT.name
    docs=[read(OUT/(f"dev_{seed}.json")) for seed in SEEDS]
    chosen={str(t):float(np.mean([p[str(t)]["dice"] for d in docs for p in d["scores"].values()])) for t in image.THRESHOLDS}
    best=max(chosen,key=lambda t:(chosen[t],-abs(float(t)-.5)))
    assert all(r["code"]==sha(Path(__file__)) for r in docs)
    seal(OUT/"evaluation_lock.json",dict(main=sha(MAIN),reference=principal["reference"],threshold=float(best),
              development=chosen,weights={str(s):{k:d[k] for k in ("model","cap")} for s,d in zip(SEEDS,docs,strict=True)},
              code=sha(Path(__file__))))


def score(seed):
    fixed=read(OUT/"evaluation_lock.json")
    assert fixed["main"]==sha(MAIN) and fixed["code"]==sha(Path(__file__))
    net,proof=network(seed)
    assert fixed["weights"][str(seed)]["model"]==proof["model"]
    refs=read(COHORT)["test"]
    assert {r["code"]:r["sha"] for r in refs}==fixed["reference"]
    store=LobeStore(limit=2)
    where=OUT/str(seed)
    where.mkdir(exist_ok=True)
    for row in refs:
        code=row["code"]
        assert read(DETECTOR_RECEIPTS/(code+".json"))["source"]==row["sha"]
        case=store.get(row,True)
        field,_=image.volume_logits(net,case,row["query"])
        y=segment(field,fixed["threshold"],proof["cap_ml"])
        p=where/(code+".npz")
        np.savez_compressed(p,shape=y.shape,prediction=np.packbits(y.ravel()))
        write(where/(code+".json"),dict(code=code,metrics=image.score(y,case.truth),prediction=sha(p)))


def release():
    principal=read(local.ROOT/"confirmed.json")
    data={}
    for row in read(COHORT)["test"]:
        code=row["code"]
        with np.load(row["path"]) as f:truth=f["mask"].astype(bool)
        records=[]
        for s in SEEDS:
            d=OUT/str(s)
            got=read(d/(code+".json"))
            f=d/(code+".npz")
            assert sha(f)==got["prediction"]
            with np.load(f) as p:y=np.unpackbits(p["prediction"],count=truth.size).reshape(truth.shape).astype(bool)
            assert image.score(y,truth)==got["metrics"]
            records.append(got["metrics"])
        data[code]={m:float(np.mean([p[m] for p in records])) for m in ("dice","recall","fp_ml","miss")}
    compared={m:summary([principal["patients"]["BP"][c][m]-data[c][m] for c in sorted(data)])
              for m in ("dice","recall","fp_ml","miss")}
    write(OUT/"confirmed.json",dict(prototype_only={m:float(np.mean([p[m] for p in data.values()])) for m in compared},
                                   full_minus_prototype_only=compared,patients=data,lock=sha(OUT/"evaluation_lock.json")))
    print(compared,flush=True)


if __name__=="__main__":
    os.umask(0o077)
    p=ArgumentParser()
    p.add_argument("stage",choices=("dev","lock","eval","release"))
    p.add_argument("--seed",choices=SEEDS,type=int,default=SEEDS[0])
    a=p.parse_args()
    if a.stage=="dev":dev(a.seed)
    elif a.stage=="lock":lock()
    elif a.stage=="eval":score(a.seed)
    else:release()
