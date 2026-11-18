"""Second candidate teacher for the nodule factorial, with local intensity evidence."""
from argparse import ArgumentParser
from functools import lru_cache
from pathlib import Path
import os

import numpy as np
from scipy import ndimage as ndi
import torch

import joint_factorial as experiment
import joint_nodule_gate as gate
import joint_roi as proposals
import thesis_model as image
from diag_localize import seal
from frozen_priors import read,sha,write
from joint_nodule_growth import estimate
from joint_nodule_run import COHORT,DEST,TRAIN,WARM
from joint_nodule_size import SIZE_CALIBRATION,get_calibrator
from thesis_lobe_model import LobeStore
from thesis_ssl import SEEDS


ROOT=DEST/"growth_matched"
ROI=ROOT/"regions"
MIX=.35


def local_prior_mask(selected,case):
    selected_mask=np.zeros(case.ct.shape,bool)
    selected_mask[selected["region"]]=selected["mask"]
    coords=np.argwhere(selected_mask).mean(0).round().astype(int)
    origin=np.clip(coords-image.SIDE//2,0,np.array(case.ct.shape)-image.SIDE)
    window=image.slices(origin)
    return origin,selected_mask[window]


def prepare(kind,which):
    data=read(COHORT)
    proposals.PREVIOUS=WARM
    model=proposals.warm_models(kind)
    labeled=[r for r in data["train"] if r["code"] in data["labels"]["20"]]
    nominal=float(np.median([r["coords"] for r in labeled]))
    store=LobeStore(limit=2)
    folder=ROI if which=="unlabelled" else ROOT/"audit"
    folder.mkdir(exist_ok=True,parents=True)
    for row in data[which]:
        path=folder/(kind+"_"+row["code"]+".json")
        if path.is_file():
            assert read(path)["source"]==row["sha"]
            continue
        text=row["query"] if kind=="B" else ""
        case=store.get(row,which=="dev")
        scores=np.mean([image.volume_logits(net,case,text)[0] for net in model],axis=0)
        candidates=estimate(case,text,scores,nominal)
        if candidates:
            choice=max(candidates,key=lambda r:(1-MIX)*r["physics"]+MIX*r["learner"])
            origin,core=local_prior_mask(choice,case)
        else:
            origin,_=proposals.evidence_region(scores,case.domain(text),1024)
            core=np.zeros((image.SIDE,)*3,bool)
        card=dict(code=row["code"],source=row["sha"],role=which,
                  choices={"agreement":{"origin":list(map(int,origin))}},
                  points=len(candidates),covered=int(core.sum()),program=sha(Path(__file__)))
        if which=="unlabelled":
            arc=path.with_suffix(".npz")
            np.savez_compressed(arc,core=np.packbits(core.ravel()),shape=core.shape)
            card["coarse_mask"]=sha(arc)
        else:
            valid=image.slices(origin)
            y=np.zeros(case.ct.shape,bool)
            y[valid]=core
            card["metrics"]=image.score(y,case.truth)
        write(path,card)
        print(which,kind,row["code"],card["covered"],card.get("metrics",{}).get("dice"),flush=True)


@lru_cache(maxsize=40)
def prior_shape(kind,code):
    path=ROI/(kind+"_"+code+".json")
    card=read(path)
    arc=path.with_suffix(".npz")
    assert sha(arc)==card["coarse_mask"]
    with np.load(arc) as f:
        shape=tuple(f["shape"])
        core=np.unpackbits(f["core"],count=int(np.prod(shape))).reshape(shape).astype(bool)
    if not core.any():
        return None
    field=ndi.distance_transform_edt(~core)
    return field<=2, (2.*np.exp(-field**2/2.25)).astype(np.float32)


@torch.no_grad()
def target(teacher,case,query,origin,rng,prior,use_uot,mode,cap):
    proposal=prior_shape("B" if query else "A",case.code)
    if proposal is None:
        return None,(0,0,None,None,None)
    support,soft_bias=proposal
    window=image.slices(origin)
    x=case.features(window,query)
    valid=torch.from_numpy(case.domain(query)[window].copy()).to(image.DEVICE)
    cue=torch.from_numpy(soft_bias).to(image.DEVICE)
    a=(teacher(image.augment(x,rng),[query])[0]+cue).sigmoid()
    b=(teacher(image.augment(x,rng),[query])[0]+cue).sigmoid()
    average=(a+b)/2
    stable=valid&((a-b).abs()<=.15)
    focus=stable&torch.from_numpy(support.copy()).to(image.DEVICE)
    if int(focus.sum())<16:
        return None,(0,0,None,None,None)
    aligned=average.clone()
    strength=before=after=None
    if use_uot:
        expected=float(np.clip(prior*int(stable.sum())/int(focus.sum()),.0001,.35))
        report=gate.report_volume(query)
        if report is not None:
            named=report/int(focus.sum())
            expected=float(np.clip(np.sqrt(expected*np.clip(named,expected/4,expected*4)),.0001,.35))
        aligned[focus],strength,before,after=experiment.transport(average[focus],expected,mode)
    probs=experiment.sharpen(aligned)
    high=focus&(probs>=.72)&(average>=.40)
    n=int(high.sum())
    if n<12:
        return None,(0,n,strength,before,after)
    if n>cap:
        idx=high.flatten().nonzero(as_tuple=True)[0]
        few=idx[probs.flatten()[idx].topk(cap).indices]
        high.fill_(False)
        high.flatten()[few]=True
    bg=stable&(probs<=.10)&(average<=.20)
    return (x,high,bg,probs),(1,int(high.sum()),strength,before,after)


def train(stage,kind,arm,seed):
    experiment.MANIFEST,experiment.PREVIOUS,experiment.ROI,experiment.OUT=COHORT,WARM,ROI,ROOT
    if stage=="lock":
        experiment.lock("agreement","agreement","conditioned")
        hashes={p.stem:sha(p) for p in ROI.glob("*.json")}
        assert len(hashes)==72
        seal(ROOT/"growth_protocol.json",dict(selection_mix=MIX,regions=hashes,source=sha(Path(__file__)),
             imaging=sha(TRAIN/"shape_protocol.json"),size=sha(SIZE_CALIBRATION),cohort=sha(COHORT)))
        return
    assert read(ROOT/"growth_protocol.json")["source"]==sha(Path(__file__))
    gate.report_volume=get_calibrator(COHORT)
    gate.activate(experiment)
    experiment.teacher_target=target
    experiment.fit(kind,arm,seed)


if __name__=="__main__":
    os.umask(0o077)
    p=ArgumentParser()
    p.add_argument("stage",choices=("dev","unlabelled","lock","train"))
    p.add_argument("--kind",choices=("A","B"),default="B")
    p.add_argument("--arm",choices=experiment.OPTIONS,default="N")
    p.add_argument("--seed",type=int,choices=SEEDS,default=SEEDS[0])
    a=p.parse_args()
    if a.stage in ("dev","unlabelled"):prepare(a.kind,a.stage)
    else:train(a.stage,a.kind,a.arm,a.seed)
