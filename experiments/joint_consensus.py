"""Measure whether independent teacher agreement yields usable pseudo foreground."""
from argparse import ArgumentParser
from pathlib import Path
import os

import numpy as np
from scipy import ndimage as ndi

import thesis_model as image
from frozen_priors import read, sha, write
from joint_roi import ROOT, warm_models
from thesis_cohort import MANIFEST
from thesis_lobe_model import LobeStore


OUT = ROOT / "consensus_audit"
COUNTS = (256, 1024, 4096)


def support(field, domain, n):
    available = np.flatnonzero(domain)
    values = field.ravel()[available]
    count = min(n,len(available))
    indices = available[np.argpartition(values,-count)[-count:]]
    result = np.zeros(domain.shape,bool)
    result.ravel()[indices] = True
    return result


def form_votes(maps, case, query):
    domain = case.domain(query)
    choices = {}
    consensus = np.mean(maps,axis=0)
    bound = domain & (case.ct > -750)
    first = maps[0]
    for n in COUNTS:
        vote = sum(support(f,domain,n).astype(np.uint8) for f in maps)
        choices[f"fixed_{n}"] = support(first,domain,n)
        choices[f"mean_{n}"] = support(consensus,domain,n)
        choices[f"two_{n}"] = vote>=2
        choices[f"three_{n}"] = vote==3
        choices[f"two_density_{n}"] = (vote>=2)&bound
    return choices, consensus


def agree(role,kind,mode):
    data = read(MANIFEST)
    models = warm_models(kind)
    store = LobeStore(limit=2)
    dest = OUT/role
    dest.mkdir(parents=True,exist_ok=True)
    for row in data[role]:
        code=row["code"]
        current=dest/(kind+"_"+code+".json")
        if current.is_file():
            assert read(current)["source"]==row["sha"]
            continue
        query = row["query"] if kind=="B" else ""
        case = store.get(row,role=="dev")
        maps = [image.volume_logits(m,case,query)[0] for m in models]
        masks,mean = form_votes(maps,case,query)
        if role=="dev":
            report={key:dict(n=int(mask.sum()),tp=int((mask&case.truth).sum()),truth=int(case.truth.sum())) for key,mask in masks.items()}
        else:
            allowed=masks[mode]
            reference=(mean + allowed*10).copy()
            reference[~case.domain(query)]=-100
            center=np.array(np.unravel_index(np.argmax(reference),reference.shape))
            origin=np.clip(center-image.SIDE//2,0,np.array(mean.shape)-image.SIDE)
            r=image.slices(origin)
            retained=allowed[r]
            expanded=ndi.binary_dilation(retained,iterations=2)
            archive=current.with_suffix(".npz")
            np.savez_compressed(archive,focus=np.packbits(expanded.ravel()),shape=expanded.shape)
            report=dict(n=int(retained.sum()),allowed=int(expanded.sum()),origin=list(map(int,origin)),sha=sha(archive),mode=mode)
        write(current,dict(source=row["sha"],code=code,mode=kind,metrics=report,program=sha(Path(__file__))))
        print(role,kind,code,report.get("n"),flush=True)


if __name__=="__main__":
    os.umask(0o077)
    p=ArgumentParser()
    p.add_argument("role",choices=("dev","unlabelled"))
    p.add_argument("kind",choices=("A","B"))
    p.add_argument("--scheme")
    a=p.parse_args()
    assert a.role=="dev" or a.scheme
    agree(a.role,a.kind,a.scheme)
