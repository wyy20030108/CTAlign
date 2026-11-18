"""Local 3D segmentation around a frozen LUNA16 detector, shared across factorial arms."""
from argparse import ArgumentParser
from pathlib import Path
from functools import lru_cache
import os

import numpy as np
import torch

from diag_localize import seal
from frozen_priors import read,sha,write
import joint_factorial as experiment
from joint_external_nodule import DIGEST,OUT as BOXES
import joint_nodule_growth_refine as growth
from joint_nodule_growth import region_grow
import joint_nodule_gate as gate
from joint_nodule_run import COHORT,DEST
from joint_nodule_size import SIZE_CALIBRATION,get_calibrator
import thesis_model as image
import thesis_ssl as warmup
from thesis_lobe_model import CODES,LobeStore
from thesis_cohort import site


ROOT=DEST/"external_matched"
WARM=ROOT/"warm"
ROI=ROOT/"regions"
WIDTH=32


def origin(case,point):
    return np.clip(np.asarray(point,int)-WIDTH//2,0,np.array(case.ct.shape)-WIDTH)


def slices(start):
    return tuple(slice(int(x),int(x)+WIDTH) for x in start)


@lru_cache(maxsize=120)
def detections(code):
    path=BOXES/(code+".json")
    if not path.is_file():
        return []
    card=read(path)
    assert card["model"]==DIGEST
    return card["predictions"]


def locations(case,query):
    domain=case.domain(query)
    lobar=CODES.get(site({"query":query})) if query else None
    found=[]
    for raw, confidence in detections(case.code):
        point=np.rint(raw).astype(int)
        if np.any(point<0) or np.any(point>=np.array(case.ct.shape)) or not domain[tuple(point)]:
            continue
        # Same CT detector confidence. Text contributes only compatible side and lobe.
        extra=.70 if lobar is not None and case.lobes[tuple(point)]==lobar else 0.
        found.append((point,float(np.log(max(confidence,1e-5))+extra)))
    if not found:
        for i,point in enumerate(gate.candidate_points(case,query)[:8]):
            found.append((point,-3.0-.15*i))
    assert found
    unique=[]
    for point,p in sorted(found,key=lambda x:x[1],reverse=True):
        if all(np.linalg.norm(point-q)>4 for q,_ in unique):
            unique.append((point,p))
        if len(unique)==8:
            break
    return unique


def pos_region(case,rng):
    centre=np.mean(case.positive,axis=0).round().astype(int)
    return slices(origin(case,centre+rng.integers(-6,7,size=3)))


def neg_region(case,query,rng):
    options=locations(case,query)
    for index in rng.permutation(len(options)):
        region=slices(origin(case,options[index][0]))
        if not case.truth[region].any():
            return region
    domain=case.domain(query)
    while True:
        point=rng.integers(0,np.asarray(case.ct.shape))
        if not domain[tuple(point)]:
            continue
        region=slices(origin(case,point))
        if not case.truth[region].any():
            return region


@torch.inference_mode()
def volume(model,case,query):
    domain=case.domain(query)
    field=np.full(case.ct.shape,-40.,np.float32)
    points=locations(case,query)
    best=points[0][1]
    for point,p in points:
        start=origin(case,point)
        region=slices(start)
        coords=np.ogrid[tuple(slice(a-b,a+WIDTH-b) for a,b in zip(start,point,strict=True))]
        dist=sum(q*q for q in coords)
        pctl=model(case.features(region,query),[query])[0].cpu().numpy()
        value=pctl+.8*np.exp(-dist/38.)+max(-2.,.8*(p-best))
        accept=domain[region]&(dist<=100)
        np.maximum(field[region],np.where(accept,value,-40.),out=field[region])
    return field,len(points)


def install():
    image.SIDE,image.slices=WIDTH,slices
    image.positive_region,image.negative_region=pos_region,neg_region
    image.volume_logits=volume


def candidates(kind,role):
    cohort=read(COHORT)
    values=[r["coords"] for r in cohort["train"] if r["code"] in cohort["labels"]["20"]]
    volume_prior=float(np.median(values))
    store=LobeStore(limit=4)
    destination=ROI if role=="unlabelled" else ROOT/"audit"
    destination.mkdir(parents=True,exist_ok=True)
    for r in cohort[role]:
        code=r["code"]
        path=destination/(kind+"_"+code+".json")
        if path.exists():continue
        case=store.get(r,role=="dev")
        query=r["query"] if kind=="B" else ""
        default=locations(case,query)[0][0]
        winner=None
        for loc,confidence in locations(case,query)[:4]:
            proposal=region_grow(case.ct,loc,volume_prior)
            if proposal:
                sl,mask,response=proposal
                value=confidence+.22*np.log(1.+response/200)-.15*abs(np.log(mask.sum()/volume_prior))
                if winner is None or value>winner[0]:winner=(value,loc,sl,mask)
        centre=default if winner is None else winner[1]
        start=origin(case,centre)
        window=slices(start)
        whole=np.zeros(case.ct.shape,bool)
        if winner is not None:whole[winner[2]]=winner[3]
        core=whole[window]
        record=dict(code=code,source=r["sha"],role=role,choices={"agreement":{"origin":start.tolist()}},
                    geometry=DIGEST,foreground=int(core.sum()))
        if role=="unlabelled":
            archive=path.with_suffix(".npz")
            np.savez_compressed(archive,core=np.packbits(core.ravel()),shape=core.shape)
            record["coarse_mask"]=sha(archive)
        else:
            record["metrics"]=image.score(whole,case.truth)
        write(path,record)
        print(role,kind,code,record["foreground"],record.get("metrics",{}).get("dice"),flush=True)


def run(stage,kind,seed,arm):
    data=read(COHORT)
    ROOT.mkdir(parents=True,exist_ok=True)
    if stage=="lock":
        WARM.mkdir(exist_ok=True)
        seal(WARM/"protocol.json",dict(visible=data["labels"]["20"],cohort=sha(COHORT),code=sha(Path(__file__)),model=DIGEST))
        experiment.MANIFEST,experiment.OUT=COHORT,ROOT
        experiment.lock("agreement","agreement","conditioned")
        return
    install()
    if stage in ("dev","unlabelled"):
        candidates(kind,stage)
    elif stage=="warm":
        warmup.OUT=WARM
        rows=[r for r in data["train"] if r["code"] in data["labels"]["20"]]
        warmup.warm("20",seed,kind,data,rows,LobeStore(limit=10))
    else:
        assert read(WARM/"protocol.json")["code"]==sha(Path(__file__))
        experiment.MANIFEST,experiment.PREVIOUS,experiment.OUT,experiment.ROI=COHORT,WARM,ROOT,ROI
        growth.ROI=ROI
        gate.report_volume=get_calibrator(COHORT)
        experiment.teacher_target=growth.target
        experiment.fit(kind,arm,seed)


if __name__=="__main__":
    os.umask(0o077)
    p=ArgumentParser()
    p.add_argument("stage",choices=("lock","dev","unlabelled","warm","train"))
    p.add_argument("--kind",choices=("A","B"),default="B")
    p.add_argument("--seed",type=int,choices=warmup.SEEDS,default=warmup.SEEDS[0])
    p.add_argument("--arm",choices=experiment.OPTIONS,default="N")
    a=p.parse_args()
    run(a.stage,a.kind,a.seed,a.arm)
