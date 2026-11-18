"""Apply the already-frozen models and thresholds to new independent patients."""
from argparse import ArgumentParser
from collections import defaultdict
from pathlib import Path
import os
import time

import numpy as np
import torch

from diag_localize import seal
from finding_train import configure
from frozen_priors import read,sha,write
import joint_external_nodule as detector
import joint_external_refine as model
from joint_external_proto_only import OUT as PROTO
import joint_nodule_gate as shape
from joint_metrics import ARM,compare,summary
from joint_nodule_pipeline import alive
from joint_replication_cohort import CANDIDATE,DEST,LOCK,STORAGE
from joint_proto_raw_audit import OUT as PROTO_AUDIT
import thesis_model as image
from thesis_component_fast import segment
from thesis_lobe_model import LobeStore
from thesis_lobes import prepare
from thesis_ssl import SEEDS


LOCK_MAIN=model.ROOT/"external_lock.json"
OUT=DEST/"scored"
PREP=DEST/"cache_receipts"
EVAL=DEST/"model_lock.json"


def anatomical_cache(shard,shards):
    PREP.mkdir(exist_ok=True)
    model_proof=read(DEST/"design.json")["frozen_model_and_thresholds"]
    assert model_proof==sha(LOCK_MAIN)
    candidate=read(CANDIDATE)[shard::shards]
    network=detector.detector()
    got=set()
    while True:
        active=any(alive(pid,"joint_replication_cohort.py") for pid in read(DEST/"fetch_pids.json").values())
        handled=0
        for row in candidate:
            code=row["code"]
            if code in got:
                continue
            p=DEST/"receipts"/(code+".json")
            if not p.is_file():
                continue
            d=read(p)
            got.add(code)
            if d["status"]!="eligible" or d["components"]!=1 or max(d["bounding_mm"])>36:
                continue
            absolute=STORAGE/"arrays"/(code+".npz")
            assert sha(absolute)==d["sha"]
            with_data=dict(row,path=str(absolute),sha=d["sha"])
            out=PREP/(code+".json")
            if out.is_file():
                assert read(out)["scan"]==d["sha"]
                continue
            with np.load(absolute) as a:
                ct,affine=a["ct"],a["affine"]
            try:
                shape.prepare([with_data])
                prepare([with_data])
            except (ValueError,AssertionError) as error:
                write(out,dict(scan=d["sha"],status="anatomy_preparation",error=type(error).__name__))
                print(code,"anatomy_preparation",flush=True)
                continue
            guessed=detector.infer(network,ct,affine)
            card=detector.OUT/(code+".json")
            write(card,dict(code=code,source=d["sha"],model=detector.DIGEST,predictions=guessed))
            write(out,dict(scan=d["sha"],status="ready",detector=sha(card),design=sha(DEST/"design.json")))
            handled+=1
            print(code,"candidates",len(guessed),flush=True)
        if not active:
            return
        if not handled:
            time.sleep(8)


def freeze():
    original=read(LOCK_MAIN)
    assert read(DEST/"design.json")["frozen_model_and_thresholds"]==sha(LOCK_MAIN)
    auxiliary=read(PROTO_AUDIT/"evaluation_lock.json")
    assert auxiliary["main"]==sha(LOCK_MAIN)
    refs=read(LOCK)
    verified={}
    exclusions={}
    for r in refs:
        code=r["code"]
        row=read(PREP/(code+".json"))
        assert row["scan"]==r["sha"]
        if row["status"]!="ready":
            exclusions[code]=row["status"]
            continue
        assert row["detector"]==sha(detector.OUT/(code+".json"))
        verified[code]=row["scan"]
    assert verified
    seal(EVAL,dict(original=sha(LOCK_MAIN),auxiliary=sha(PROTO_AUDIT/"evaluation_lock.json"),source=sha(LOCK),
                   scans=verified,excluded=exclusions,code=sha(Path(__file__)),
                   thresholds=original["thresholds"],full_methods=[*ARM,"BR"],
                   analysis="Fixed-model patient bootstrap replication; same seed aggregation and outcome definitions as the 22-case external confirmation."))
    print("Frozen replication",len(verified),"excluded at anatomy stage",len(exclusions),flush=True)


def models(fixed):
    original=read(LOCK_MAIN)
    extra=read(PROTO_AUDIT/"evaluation_lock.json")
    assert fixed["original"]==sha(LOCK_MAIN) and fixed["auxiliary"]==sha(PROTO_AUDIT/"evaluation_lock.json")
    for arm in (*ARM,"BR"):
        modality,letter=arm
        for seed in SEEDS:
            if arm=="BR":
                info=extra["weights"][str(seed)]
                cap,t=info["cap"],extra["threshold"]
                weights=PROTO/modality/str(seed)/letter/"model.pt"
                digest=info["model"]
            else:
                info=next(x for x in original["weights"][arm] if x["seed"]==seed)
                cap,t=info["cap_ml"],original["thresholds"][arm]["threshold"]
                weights=model.ROOT/modality/str(seed)/letter/"model.pt"
                digest=info["weight"]
            assert sha(weights)==digest
            configure(seed)
            net=image.FindingNet().to(image.DEVICE).eval()
            net.load_state_dict(torch.load(weights,weights_only=True,map_location=image.DEVICE))
            yield arm,seed,net,cap,t


def score(shard,shards):
    fixed=read(EVAL)
    assert fixed["code"]==sha(Path(__file__))
    refs=[r for r in read(LOCK) if r["code"] in fixed["scans"]]
    model.install()
    nets=list(models(fixed))
    store=LobeStore(limit=2)
    OUT.mkdir(exist_ok=True)
    for row in refs[shard::shards]:
        code=row["code"]
        out=OUT/(code+".json")
        masks=out.with_suffix(".npz")
        if out.is_file():
            card=read(out)
            assert card["lock"]==sha(EVAL) and card["prediction"]==sha(masks)
            continue
        assert fixed["scans"][code]==row["sha"]
        case=store.get(row,True)
        q=row["query"]
        packs={}
        answer={}
        for arm,seed,net,cap,t in nets:
            key=arm+"_"+str(seed)
            volume,_=image.volume_logits(net,case,q if arm[0]=="B" else "")
            y=segment(volume,t,cap)
            packs[key]=np.packbits(y.ravel())
            answer[key]=image.score(y,case.truth)
        np.savez_compressed(masks,shape=case.ct.shape,**packs)
        write(out,dict(code=code,answer=answer,lock=sha(EVAL),prediction=sha(masks)))
        print(code,len(answer),flush=True)


def release():
    fixed=read(EVAL)
    patient=defaultdict(dict)
    for row in read(LOCK):
        code=row["code"]
        if code not in fixed["scans"]:
            continue
        disk=Path(row["path"])
        assert sha(disk)==fixed["scans"][code]
        with np.load(disk) as a:truth=a["mask"].astype(bool)
        card=read(OUT/(code+".json"))
        p=OUT/(code+".npz")
        assert card["lock"]==sha(EVAL) and card["prediction"]==sha(p)
        with np.load(p) as z:
            for a in (*ARM,"BR"):
                variants=[]
                for s in SEEDS:
                    key=a+"_"+str(s)
                    pred=np.unpackbits(z[key],count=truth.size).reshape(truth.shape).astype(bool)
                    metric=image.score(pred,truth)
                    assert metric==card["answer"][key]
                    variants.append(metric)
                patient[a][code]={k:float(np.mean([v[k] for v in variants])) for k in ("dice","recall","fp_ml","miss")}
    contrasts=compare(patient)
    contrasts["transport_given_prototypes"]={m:summary([patient["BP"][c][m]-patient["BR"][c][m] for c in sorted(fixed["scans"])])
                                              for m in ("dice","recall","fp_ml","miss")}
    report={a:{m:float(np.mean([v[m] for v in points.values()])) for m in ("dice","recall","fp_ml","miss")} for a,points in patient.items()}
    write(DEST/"confirmed.json",dict(summary=report,comparisons=contrasts,patients=patient,lock=sha(EVAL)))
    print("Independent replication",len(fixed["scans"]),{a:round(v["dice"]*100,3) for a,v in report.items()},flush=True)


if __name__=="__main__":
    os.umask(0o077)
    p=ArgumentParser()
    p.add_argument("stage",choices=("cache","lock","score","release"))
    p.add_argument("--shard",type=int,default=0)
    p.add_argument("--shards",type=int,default=2)
    a=p.parse_args()
    if a.stage=="cache":anatomical_cache(a.shard,a.shards)
    elif a.stage=="lock":freeze()
    elif a.stage=="score":score(a.shard,a.shards)
    else:release()
