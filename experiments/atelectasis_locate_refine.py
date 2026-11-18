"""Supervised whole-CT localization followed by a bounded high-resolution refinement."""
from argparse import ArgumentParser
from collections import Counter
from pathlib import Path
from itertools import product
import os,time

import numpy as np
from scipy import ndimage as nd
import torch
from torch.nn import functional as F

from atelectasis_b0 import LOCK, ROOT, GRID, SHAPE, STEPS, coarse_image, load_row
from atelectasis_sliding import OUT as SLIDING, RUNS, DualScale, sliding, score
from diag_localize import seal
from finding_train import QueryNet, configure, photo
from frozen_priors import read, sha, write
from swap_data import side

OUT=ROOT/"locate_refine"
K=4
CORE=32
PROTOCOL={
    "cohort":"Reuse two labeled masks (3 seeds) and the paired 20-mask engineering control (891), plus the same six development patients.",
    "scanner":"Find broad lung fields by HU and 3D connected components, with a soft margin. Use explicit laterality and above/below or anterior/posterior text only. No anatomy labels or external weights.",
    "locator":"Same 96-cube image-text network, trained against a physical 12 mm Gaussian centered on the annotated training target, using weighted normalized spatial cross entropy for 384 matched updates.",
    "proposal":"On the requested side and broad lung support, rank coordinate candidates with 0.55 smoothed fine evidence percentile plus 0.25 learned center percentile plus 0.20 broad text spatial prior. Four 48 mm cores, centers at least 36 mm apart. Same rules for all runs.",
    "refinement":"Keep the matched high resolution sliding-window checkpoint. In each auto core, retain the component around the strongest plausible fine score using floor p=0.08 and peak minus 0.85 logit. Apply a per-study cap of max(3 mL, 3 times the median training target volume).",
    "diagnostics":"The older B1 at 0.5 stays the reference. Also compare B1 at 0.20 without localization and a pure restriction of B1 at 0.5. Score auto proposal coverage separately. All development metrics are paired; keep held-out test patients sealed."
}

def lung_field(ct):
    low=ct<-425
    # Fill the body silhouette in each axial plane, retaining pleural lung up to a margin.
    inside=np.empty(ct.shape,dtype=bool)
    midline=[]
    for z in range(ct.shape[2]):
        soft=ct[:,:,z]>-350
        # Remove table and isolated objects by keeping the main soft tissue shell.
        lab,n=nd.label(soft)
        cnt=np.bincount(lab.ravel());cnt[0]=0
        body=nd.binary_fill_holes(lab==np.argmax(cnt)) if n else soft
        inside[:,:,z]=body
        if body.sum()>ct.shape[0]*ct.shape[1]*.12:
            x=np.flatnonzero(body.any(axis=1))
            if len(x):midline.append((x[0]+x[-1])/2)
    midpoint=float(np.median(midline)) if midline else (ct.shape[0]-1)/2
    masks={}
    xx=np.arange(ct.shape[0])[:,None,None]
    for key,half in (("right",xx>midpoint),("left",xx<=midpoint)):
        air=low & inside & half
        lab,n=nd.label(air)
        cnt=np.bincount(lab.ravel());cnt[0]=0
        if n==0 or cnt.max()<5000:
            # The center is still constrained by image side if the shell is broken.
            seeds=low & half
            labels,n=nd.label(seeds)
            counts=np.bincount(labels.ravel());counts[0]=0
            at_boundary=np.unique(np.concatenate((labels[0].ravel(),labels[-1].ravel(),labels[:,0].ravel(),labels[:,-1].ravel())))
            counts[at_boundary]=0
            lung=(labels==np.argmax(counts)) if counts.max()>0 else air
        else:
            lung=lab==np.argmax(cnt)
        if lung.sum()<5000:raise ValueError("could not recover a valid lung field")
        masks[key]=lung
    return masks,inside,midpoint

def prior(query,bounds,shape):
    # These broad anatomical weights remain soft, including at lobe edges.
    q=query.lower()
    y0,y1,z0,z1=bounds
    y=(np.arange(shape[1])-y0)/max(1,y1-y0)
    z=(np.arange(shape[2])-z0)/max(1,z1-z0)
    if "middle lobe" in q or "lingul" in q:
        yc,zc,sy,sz=.77,.43,.28,.26
    elif "lower lobe" in q:
        yc,zc,sy,sz=.33,.22,.31,.31
        if "superior segment" in q:zc=.55
    elif "upper lobe" in q:
        yc,zc,sy,sz=.55,.74,.37,.29
    else:
        yc,zc,sy,sz=.5,.5,.5,.5
    if "posterior" in q or "posterobasal" in q:yc=min(yc,.32)
    if "anterior" in q:yc=max(yc,.64)
    return np.exp(-.5*(((y-yc)/sy)**2)[:,None]-.5*(((z-zc)/sz)**2)[None,:]).astype(np.float32)

def center_target(mask):
    actual=np.asarray(mask.shape)
    center=np.argwhere(mask).mean(0)
    axes=[(np.arange(n)+.5)*size/n-.5 for n,size in zip(SHAPE,actual)]
    factors=[np.exp(-.5*(((a-c)*GRID)/12)**2) for a,c in zip(axes,center)]
    g=factors[0][:,None,None]*factors[1][None,:,None]*factors[2][None,None,:]
    g=(g/g.sum()).astype(np.float32)
    return torch.from_numpy(g).cuda()

def fit(training,seed,out):
    configure(seed);model=QueryNet().cuda()
    path=out/"locator.pt"
    if path.exists():
        assert sha(path)==read(out/"training.json")["sha"]
        model.load_state_dict(torch.load(path,weights_only=True,map_location="cuda"))
        return model.eval()
    rng=np.random.default_rng(seed)
    order=np.concatenate([rng.permutation(len(training)) for _ in range((STEPS+len(training)-1)//len(training))])[:STEPS]
    items=[]
    for row in training:
        ct,m=load_row(row)
        items.append((coarse_image(ct).cuda(),row["query"],center_target(m)))
    opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
    trace=[]
    for j,i in enumerate(order,1):
        ct,q,g=items[i]
        log=model(photo(ct,rng),[q])[0]
        loss=-(g.flatten()*F.log_softmax(log.flatten(),dim=0)).sum()
        opt.zero_grad(set_to_none=True);loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),5,error_if_nonfinite=True)
        opt.step();trace.append(float(loss.detach()))
        if j%64==0:print("fit locator",seed,j,round(float(np.mean(trace[-64:])),5),flush=True)
    tmp=path.with_suffix(".part");torch.save(model.state_dict(),tmp);tmp.replace(path)
    write(out/"training.json",dict(sha=sha(path),train_codes=[r["code"] for r in training],loss=trace))
    return model.eval()

def box(loc,shape):
    origin=np.clip(np.asarray(loc)-CORE//2,0,np.asarray(shape)-CORE)
    return tuple(slice(int(a),int(a)+CORE) for a in origin)

def percentile(array,roi):
    pooled=array[roi][::5].copy();pooled.sort()
    assert len(pooled)>1
    ranks=np.searchsorted(pooled,array[roi],side="right")/len(pooled)
    return ranks.astype(np.float32)

def candidates(ct,query,fine,heat):
    region_by_side,body,midpoint=lung_field(ct)
    selected=region_by_side[side(query)]
    coords=np.where(selected)
    bounds=(int(coords[1].min()),int(coords[1].max()),int(coords[2].min()),int(coords[2].max()))
    support=nd.binary_dilation(selected,iterations=5)&body
    ranking=nd.gaussian_filter(fine,sigma=1.8)
    prior_map=np.broadcast_to(prior(query,bounds,ct.shape)[None],ct.shape)
    score_map=np.full(ct.shape,-np.inf,dtype=np.float32)
    score_map[support]=.55*percentile(ranking,support)+.25*percentile(heat,support)+.20*prior_map[support]
    coords=[]
    for _ in range(K):
        peak=np.unravel_index(np.argmax(score_map),ct.shape)
        assert np.isfinite(score_map[peak])
        coords.append(tuple(int(a) for a in peak))
        distance=24  # at least 36 mm separation
        work=[np.arange(max(0,j-distance),min(n,j+distance+1)) for j,n in zip(peak,ct.shape)]
        grids=np.ix_(*work)
        dd=sum((a-j)**2 for a,j in zip(grids,peak))
        score_map[grids]=np.where(dd<distance**2,-np.inf,score_map[grids])
    return coords,support,midpoint

def limit_volume(m,fine,cap):
    present=np.flatnonzero(m)
    if len(present)>cap:
        vals=fine.ravel()[present]
        keep=np.argpartition(vals,-cap)[-cap:]
        out=np.zeros(m.size,dtype=bool);out[present[keep]]=True
        m=out.reshape(m.shape)
    return m

def refine(fine,coordinates,support,cap):
    selected=np.zeros(fine.shape,dtype=bool)
    seen=np.zeros_like(selected)
    cover=[]
    for peak in coordinates:
        sl=box(peak,fine.shape)
        local=fine[sl]
        possible=support[sl]
        values=np.where(possible,local,-np.inf)
        at=np.unravel_index(np.argmax(values),local.shape)
        cutoff=max(np.log(.08/.92),float(values[at])-.85)
        binary=possible & (local>=cutoff)
        if binary.any():
            labels,n=nd.label(binary,np.ones([3]*3,bool))
            assert labels[at]!=0
            component=(labels==labels[at])
            if component.sum()>=4:selected[sl]|=component
        seen[sl]=True
        cover.append(seen.copy())
    return limit_volume(selected,fine,cap),cover

def develop(budget,seed):
    os.umask(0o077)
    training_file=ROOT/"pilot_private.json"
    lock=read(LOCK);assert sha(training_file)==lock["manifest_sha"]
    record=read(OUT/"protocol.json")
    assert record["sha"]==sha(Path(__file__)) and record["manifest"]==sha(training_file)
    cohort=read(training_file);train=cohort["selected" if budget=="b10" else "train"]
    previous=SLIDING/budget/str(seed);old=read(previous/"results.json")
    assert set(old["train_codes"])=={r["code"] for r in train}
    assert sha(previous/"B1.pt")==old["model_sha"]
    destination=OUT/budget/str(seed);destination.mkdir(parents=True,exist_ok=True)
    if (destination/"results.json").exists():return
    loc=fit(train,seed,destination)
    fine_model=DualScale().cuda().eval()
    fine_model.load_state_dict(torch.load(previous/"B1.pt",weights_only=True,map_location="cuda"))
    cap_ml=max(3.,3*float(np.median([r["coords"]*GRID**3/1000 for r in train])))
    cap_vox=int(cap_ml*1000/GRID**3)
    records=[]
    for row in cohort["development"]:
        start=time.monotonic()
        code=row["code"]
        ct,gt=load_row(row)
        assert side(row["query"])==row["side"]
        old_row=next(r for r in old["rows"] if r["code"]==code)
        old_pack=previous/(code+".npz")
        assert sha(old_pack)==old_row["prediction_sha"]
        with np.load(old_pack) as v:
            original=np.unpackbits(v["B1"],count=gt.size).reshape(gt.shape).astype(bool)
        fine,_,_=sliding(fine_model,ct,row["query"],torch.device("cuda"))
        assert np.array_equal(original,fine>=0)
        with torch.inference_mode():
            low=loc(coarse_image(ct).cuda(),[row["query"]])
            heat=F.interpolate(low[:,None],size=gt.shape,mode="trilinear",align_corners=False)[0,0].cpu().numpy()
        locs,support,midpoint=candidates(ct,row["query"],fine,heat)
        improved,fields=refine(fine,locs,support,cap_vox)
        methods=dict(B1=original,B1_0p2=fine>=np.log(.2/.8),B1_restricted=original&fields[-1],located_refined=improved)
        stats={k:score(v,gt) for k,v in methods.items()}
        for k,v in stats["B1"].items():assert v==old_row["metrics"]["B1"]["0.5"][k]
        recalls=[float(np.count_nonzero(r&gt)/gt.sum()) for r in fields]
        part=destination/(code+".npz")
        np.savez_compressed(part,shape=gt.shape,proposals=np.array(locs),
                            **{k:np.packbits(v.ravel()) for k,v in methods.items()})
        records.append(dict(code=code,metrics=stats,recall_of_proposals=recalls,search_voxels=int(fields[-1].sum()),
                            target_within_lung_support=float(np.count_nonzero(gt&support)/gt.sum()),
                            pred_sha=sha(part),prior_prediction_sha=sha(old_pack),cap_ml=cap_ml,
                            proposal_ijk=locs,seconds=time.monotonic()-start))
        print("scored",budget,seed,code,round(stats["located_refined"]["dice"],5),round(recalls[-1],3),round(records[-1]["seconds"],1),flush=True)
    write(destination/"results.json",dict(seed=seed,budget=budget,rows=records,protocol=sha(OUT/"protocol.json"),
         previous_weights=sha(previous/"B1.pt"),locator_weights=sha(destination/"locator.pt"),
         train_codes=sorted(r["code"] for r in train)))

if __name__=="__main__":
    p=ArgumentParser()
    p.add_argument("command",choices=("seal","develop"));p.add_argument("--budget",choices=("b10","all_labeled"),default="b10")
    p.add_argument("--seed",type=int,default=891);a=p.parse_args()
    if a.command=="seal":
        OUT.mkdir(exist_ok=True)
        seal(OUT/"protocol.json",dict(design=PROTOCOL,sha=sha(Path(__file__)),
              manifest=sha(ROOT/"pilot_private.json"),starting_protocol=sha(SLIDING/"protocol.json")))
    else:
        assert (a.budget,a.seed) in RUNS
        develop(a.budget,a.seed)

