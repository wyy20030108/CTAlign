"""Audit CT intensity masks at morphology proposals using only the development group."""
from argparse import ArgumentParser
from pathlib import Path
import os

import numpy as np
from scipy import ndimage as ndi

import joint_nodule_gate as gate
import joint_roi as proposal
import thesis_model as image
from frozen_priors import read,sha,write
from joint_nodule_run import COHORT,DEST,WARM
from joint_nodule_size import get_calibrator
from thesis_component_fast import segment
from thesis_lobe_model import LobeStore


OUT=DEST/"growth_audit"
CONN=ndi.generate_binary_structure(3,2)


def region_grow(ct,pt,expected):
    coords=np.asarray(pt)
    radius=float(np.clip((3*expected/(4*np.pi))**(1/3),1.7,5.5))
    width=9
    lo=np.maximum(0,coords-width)
    hi=np.minimum(np.asarray(ct.shape),coords+width+1)
    region=tuple(slice(a,b) for a,b in zip(lo,hi,strict=True))
    local=np.asarray(pt)-lo
    x=np.ogrid[tuple(slice(a-b,c-b) for a,b,c in zip(lo,coords,hi,strict=True))]
    distance=np.sqrt(sum(a*a for a in x))
    values=ndi.gaussian_filter(ct[region].astype(np.float32),.55)
    shell=values[(distance>6)&(distance<9)]
    baseline=float(np.median(shell))
    peak=float(np.quantile(values[distance<1.6],.75))
    if peak-baseline<70:
        return None
    active=(values>=baseline+.47*(peak-baseline))&(distance<=radius+1.7)
    label,_=ndi.label(active,CONN)
    center_label=label[tuple(local)]
    if not center_label:
        touch=label[distance<=1.45]
        nonzero=touch[touch>0]
        if not len(nonzero):return None
        center_label=np.argmax(np.bincount(nonzero)[1:])+1
    mask=label==center_label
    if not 5<=int(mask.sum())<=max(25,3*expected):
        return None
    return region, mask, float(peak-baseline)


def estimate(case,query,logits,expected):
    outputs=[]
    for i,p in enumerate(gate.candidate_points(case,query)):
        data=region_grow(case.ct,p,expected)
        if data is None:
            continue
        sl,local,response=data
        model=float(np.mean(logits[sl][local]))
        count=local.sum()
        occupancy=abs(np.log(count/expected))
        size_penalty=.20*occupancy
        prior=.30*np.log(1.+response/200)-.035*i-size_penalty
        outputs.append(dict(order=i,region=sl,mask=local,physics=prior,learner=model+prior))
    return outputs


def run(modality):
    OUT.mkdir(exist_ok=True,parents=True)
    data=read(COHORT)
    proposal.PREVIOUS=WARM
    nets=proposal.warm_models(modality)
    store=LobeStore(limit=4)
    supervised=[r for r in data["train"] if r["code"] in data["labels"]["20"]]
    habit=float(np.median([r["coords"] for r in supervised]))
    from_report=get_calibrator(COHORT)
    for r in data["dev"]:
        path=OUT/(modality+"_"+r["code"]+".json")
        if path.exists():continue
        case=store.get(r,True)
        text=r["query"] if modality=="B" else ""
        logits=np.mean([image.volume_logits(model,case,text)[0] for model in nets],axis=0)
        written=from_report(text)
        mass=habit if written is None else float(np.sqrt(habit*np.clip(written,habit/4,habit*4)))
        all_candidates=estimate(case,text,logits,mass)
        measures={}
        for group in ("physics","learner"):
            prediction=np.zeros(case.ct.shape,bool)
            if all_candidates:
                selected=max(all_candidates,key=lambda v:v[group])
                prediction[selected["region"]]=selected["mask"]
            measures[group]=image.score(prediction,case.truth)
        measures["teacher"]=image.score(segment(logits,.5,3.),case.truth)
        measures["teacher_shape"]=image.score(segment(logits+gate.offset(gate.guidance(case,text)),.5,3.),case.truth)
        write(path,dict(code=r["code"],kind=modality,answers=measures,draws=len(all_candidates),source=r["sha"],program=sha(Path(__file__))))
        print(modality,r["code"],len(all_candidates),{k:round(v["dice"],4) for k,v in measures.items()},flush=True)


if __name__=="__main__":
    os.umask(0o077)
    p=ArgumentParser()
    p.add_argument("modality",choices=("A","B"))
    run(p.parse_args().modality)
