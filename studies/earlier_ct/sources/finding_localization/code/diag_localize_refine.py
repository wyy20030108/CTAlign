"""Actual shared-proposal patch refinement; distinct from scoring clipped full-field predictions."""
import argparse
import collections
import json
import os
from pathlib import Path

import numpy as np
import torch
from diag_localize import (BASE, GROUPS, KS, SRC, WIDTH, cases, configure, image, read, score_mask,
                           sha, target_metrics, threshold_and_weights, write)


@torch.inference_mode()
def refine(net,ct,docs,starts,threshold):
    shape=np.asarray(ct.shape)
    r=dict(ct=ct,ax=[np.linspace(-1,1,n,dtype=np.float32) for n in shape])
    # A candidate gets 64^3 CT context; only voxels in its central/edge-adjusted 32^3 core are emitted.
    sums=np.zeros((2,*shape),np.float32)
    hits=np.zeros(shape,np.uint8)
    cut=float(np.log(np.float32(threshold)/(1-np.float32(threshold))))
    for loc in starts:
        pos=np.asarray(loc)
        begin=np.clip(pos-(64-WIDTH)//2,0,shape-64)
        patch=tuple(slice(int(a),int(a)+64) for a in begin)
        core=tuple(slice(int(a),int(a)+WIDTH) for a in pos)
        inner=tuple(slice(int(a),int(a)+WIDTH) for a in pos-begin)
        vals=net(image(r,patch),docs).cpu().numpy()
        sums[(slice(None),*core)] += vals[(slice(None),*inner)]
        hits[core]+=1
    region=hits>0
    assert np.array_equal(region,score_mask(shape,starts))
    binary=np.zeros_like(sums,dtype=bool)
    binary[:,region]=sums[:,region]/hits[region]>=cut
    return binary,region


def run(arm):
    base=BASE/arm
    previous=read(base/"results.json")
    starts={r["code"]:r["starts"] for r in read(BASE/"candidate_lock.json")["cases"]}
    oracle={r["code"]:r["oracle_boxes"] for r in previous["cases"]}
    queries,clinical={},{}
    for g in GROUPS:
        queries.update({r["code"]:r["queries"] for r in read(SRC/(g+".private.json"))})
        clinical.update(read(Path("phase3_swap/clinical")/(g+".private.json")))
    net,threshold,ck=threshold_and_weights(arm)
    assert threshold==previous["threshold"] and ck==read(base/"prediction_lock.json")["checkpoint_sha256"]
    modes=("finding","clinical")
    # Seal actual patch predictions for automatic regions separately, prior to using oracle regions.
    all_data=[]
    for variant in ("automatic","oracle"):
        d=base/("refinement_"+variant);d.mkdir(exist_ok=True)
        for r in cases():
            code=r["code"];path=Path(r["path"])
            pack=d/(code+".npz")
            if not pack.exists():
                with np.load(path) as v:ct=v["ct"]
                windows={f"K{k}":starts[code][:k] for k in KS} if variant=="automatic" else {"oracle_2":oracle[code]}
                buffers={"shape":np.asarray(ct.shape)}
                for g,boxes in windows.items():
                    for mode,docs in (("finding",queries[code]),("clinical",clinical.get(code))):
                        if docs is None:continue
                        binary,region=refine(net,ct,docs,boxes,threshold)
                        buffers[g+"__"+mode]=np.packbits(binary[:,region],bitorder="little")
                tmp=pack.with_suffix(".tmp.npz");np.savez_compressed(tmp,**buffers);tmp.replace(pack)
        checks={r["code"]:sha(d/(r["code"]+".npz")) for r in cases()}
        write(d/"prediction_lock.json",dict(variant=variant,checkpoint_sha256=ck,threshold=threshold,checks=checks,
             code_sha256=sha(Path(__file__)),all_candidates_shared_between_queries=True,
             cases_using_target_derived_coordinates=0 if variant=="automatic" else len(checks)))
        for r in cases():
            code=r["code"];pack=d/(code+".npz")
            assert sha(pack)==checks[code]
            with np.load(r["path"]) as v:masks=v["masks"].astype(bool)
            shape=masks.shape[1:]
            with np.load(pack) as v:
                assert tuple(v["shape"])==shape
                for key in sorted(set(v.files)-{"shape"}):
                    g,mode=key.split("__")
                    boxes=starts[code][:int(g[1:])] if variant=="automatic" else oracle[code]
                    region=score_mask(shape,boxes)
                    p=np.zeros(masks.shape,bool)
                    p[:,region]=np.unpackbits(v[key],count=int(region.sum())*2,bitorder="little").reshape(2,-1)
                    scores={}
                    for condition,pair in (("correct",p),("swapped",p[::-1])):
                        scores[condition]=[target_metrics(x,m,region) for x,m in zip(pair,masks)]
                    all_data.append(dict(code=code,region=g,mode=mode,scores=scores))
    summary={}
    for mode in modes:
        summary[mode]={}
        for g in ("K1","K2","K5","oracle_2"):
            rows=[r for r in all_data if r["mode"]==mode and r["region"]==g]
            hits_by_case=[[m for m in r["scores"]["correct"] if m["hit50"]] for r in rows]
            conditional=[np.mean([m["dice_inside"] for m in hits]) for hits in hits_by_case if hits]
            def dice(cond):
                return float(np.mean([np.mean([m["dice_whole"] for m in r["scores"][cond]]) for r in rows]))
            reference=previous["summary"][mode][g]
            assert sum(map(len,hits_by_case))==reference["hit50_targets"] and len(conditional)==reference["hit50_patients"]
            summary[mode][g]=dict(n=len(rows),targets=len(rows)*2,hit50_targets=reference["hit50_targets"],
                hit50_patients=len(conditional),conditional_inside_dice=float(np.mean(conditional)) if conditional else None,
                whole_target_dice=dice("correct"),swapped_whole_target_dice=dice("swapped"))
    write(base/"refine_results.json",dict(arm=arm,threshold=threshold,source_code_sha256=sha(Path(__file__)),rows=all_data,summary=summary))
    print("actual shared-crop refinement",arm,summary["finding"]["K2"],flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("arm");a=p.parse_args()
    os.umask(0o077);configure(891);run(a.arm)
