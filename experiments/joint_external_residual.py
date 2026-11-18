"""Keep the verified candidate boundary and evaluate all trained models as residual refiners."""
from argparse import ArgumentParser
from functools import lru_cache
from pathlib import Path
import os

import numpy as np
from scipy import ndimage as ndi
import torch

from diag_localize import seal
from finding_train import configure
from frozen_priors import read,sha,write
import joint_external_refine as primary
from joint_nodule_growth import region_grow
from joint_nodule_run import COHORT,DEST
from thesis_component_fast import segment
from thesis_lobe_model import LobeStore
import thesis_model as image
from thesis_ssl import SEEDS


ROOT=DEST/"external_residual"
RESIDUAL_WEIGHT=.75


@lru_cache(maxsize=1)
def annotation_mass():
    co=read(COHORT)
    return float(np.median([r["coords"] for r in co["train"] if r["code"] in co["labels"]["20"]]))


def base_contour(case,query):
    volume=annotation_mass()
    selected=None
    for loc,confidence in primary.locations(case,query)[:4]:
        x=region_grow(case.ct,loc,volume)
        if x is None:
            continue
        sl,mask,response=x
        value=confidence+.22*np.log(1.+response/200)-.15*abs(np.log(mask.sum()/volume))
        if selected is None or value>selected[0]:
            selected=(value,loc,sl,mask)
    return selected


@torch.inference_mode()
def refine(model,case,query):
    selected=base_contour(case,query)
    if selected is None:
        return primary.volume(model,case,query)
    _,centre,sl,mask=selected
    start=primary.origin(case,centre)
    window=primary.slices(start)
    initial=np.zeros(case.ct.shape,bool)
    initial[sl]=mask
    crop=initial[window]
    if not crop.any():
        return primary.volume(model,case,query)
    z=model(case.features(window,query),[query])[0].cpu().numpy()
    distance=ndi.distance_transform_edt(crop)-ndi.distance_transform_edt(~crop)
    rim=(np.abs(distance)<=1.5)&case.domain(query)[window]
    center=float(np.median(z[rim])) if rim.any() else float(np.median(z))
    correction=RESIDUAL_WEIGHT*np.clip(z-center,-2.,2.)
    estimate=np.where(case.domain(query)[window],distance+correction,-40.)
    whole=np.full(case.ct.shape,-40.,np.float32)
    whole[window]=estimate
    return whole,1


def install():
    primary.install()
    image.volume_logits=refine
    image.segment=segment


def lock():
    ROOT.mkdir(exist_ok=True,parents=True)
    seal(ROOT/"protocol.json",dict(source=sha(Path(__file__)),cohort=sha(COHORT),training=sha(primary.ROOT/"protocol.json"),
          external_model=sha(primary.WARM/"protocol.json"),readout=RESIDUAL_WEIGHT,role="development only before selection"))


def assess(arm,seed):
    source=ROOT/"protocol.json"
    fixed=read(source)
    assert fixed["source"]==sha(Path(__file__)) and fixed["cohort"]==sha(COHORT)
    k,a=arm
    src=primary.ROOT/k/str(seed)/a
    destination=ROOT/k/str(seed)/a
    dest=destination/"development.json"
    if dest.is_file():
        assert read(dest)["protocol"]==sha(source) and read(dest)["model"]==sha(destination/"model.pt")
        return
    original=read(src/"development.json")
    weight=src/"model.pt"
    assert original["model"]==sha(weight) and original["protocol"]==fixed["training"]
    destination.mkdir(exist_ok=True,parents=True)
    archive=destination/"model.pt"
    if not archive.exists():archive.symlink_to(weight.resolve())
    assert sha(archive)==original["model"]
    install()
    configure(seed)
    net=image.FindingNet().to(image.DEVICE).eval()
    net.load_state_dict(torch.load(weight,map_location=image.DEVICE,weights_only=True))
    store=LobeStore(limit=2)
    scores=[]
    for row in read(COHORT)["dev"]:
        q=row["query"] if k=="B" else ""
        values,_=image.case_eval(net,row,store.get(row,True),q,original["cap_ml"])
        scores.append(dict(code=row["code"],value=values))
    write(dest,dict(model=sha(archive),score=scores,cap_ml=original["cap_ml"],protocol=sha(source),
                    parent=sha(src/"development.json")))
    print(arm,seed,flush=True)


if __name__=="__main__":
    os.umask(0o077)
    p=ArgumentParser()
    p.add_argument("stage",choices=("lock","dev"))
    p.add_argument("--arm",default="BS")
    p.add_argument("--seed",type=int,choices=SEEDS,default=SEEDS[0])
    a=p.parse_args()
    if a.stage=="lock":lock()
    else:assess(a.arm,a.seed)
