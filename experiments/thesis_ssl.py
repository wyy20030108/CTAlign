"""Matched A-F text and semi-supervised training with two-class Adaptive UOT."""
from argparse import ArgumentParser
from copy import deepcopy
from pathlib import Path
import os,time
import numpy as np
import torch
from torch.nn import functional as F
from frozen_priors import read,sha,write
from diag_localize import seal
from finding_train import configure
from atelectasis_locate_refine import prior
from thesis_cohort import ROOT,MANIFEST,LOCK,PRIMARY
from thesis_model import (FindingNet,CaseStore,DEVICE,labeled_loss,slices,side,positive_region,
                          augment,case_eval,GRID,SIDE,THRESHOLDS)
from thesis_lobe_model import LobeStore

SEEDS=(891,1901,2707)
ARMS="ABCDEF"
WARM=448
STEPS=384
EMA=.992
OUT=ROOT/"controlled"
PROTOCOL=dict(seeds=list(SEEDS),arms=list(ARMS),warm_updates=WARM,matched_continuation=STEPS,ema=EMA,
   equality="Pair seeds, visible masks, supervised schedules, inference and threshold candidate list per budget. D and E receive the same unlabelled CT schedule, while E reads the frozen donor query. A/C see no text at either stage.",
   anatomy="All variants use image-only lung fields. Text variants also use the frozen LTRC-trained lungmask anatomy model, its query-selected lobe and a 6-mm soft margin. Full ipsilateral coverage is preserved. A/C have no query-selected lobe. No ReX lesion-trained checkpoint is used.",
   contrasts="B vs A; D vs B; D vs C; D vs E; F vs D. For the final primary comparison, select the stronger of B and D on development data only, retain all other comparisons.",
   unlabeled="36 planned report-only scans from patients absent from all releases. Four CT-only window proposals and one selected during the unlabeled step. Select stable foreground and background after two independent weak perturbations. A negative-only patch contributes zero.",
   transport="Half-relaxed two-class Sinkhorn, voxel row mass fixed, dynamic reference foreground from labelled-positive windows, entropy weight 0.5. Foreground evidence determines KL penalty. Teacher consistency acceptance and voxel indices computed before calibration.",
   fixed_eval="Top connected foreground component, trained positive volume cap, common threshold grid. Threshold and baseline remain selectable only on development data. No test scoring from this module.")

def load_cohort():
    manifest=read(MANIFEST)
    assert read(LOCK)["manifest_sha"]==sha(MANIFEST)
    return manifest
def exposure(budget,cohort):
    codes=cohort["labels"][budget]
    assert set(codes)<= {r["code"] for r in cohort["train"]}
    return [r for r in cohort["train"] if r["code"] in codes]
def make_schedule(n,steps,seed):
    g=np.random.default_rng(seed)
    return np.concatenate([g.permutation(n) for _ in range((steps+n-1)//n)])[:steps]
def update(model,optimizer,loss):
    optimizer.zero_grad(set_to_none=True);loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(),5,error_if_nonfinite=True)
    optimizer.step()
def checkpoint(path,model):
    tmp=path.with_suffix(".part")
    torch.save(model.state_dict(),tmp);tmp.replace(path)
    return sha(path)

@torch.no_grad()
def adaptive_uot(p,valid,running):
    # Paper objective: fixed uniform row marginal, KL-relaxed column marginal.
    selected=p[valid].clamp(1.e-5,1-1.e-5)
    evidence=float(selected.mean())
    eps=.5
    lam=.05+(2.-.05)*min(1,evidence/max(.003,3*running))
    source=torch.stack((1-selected,selected),dim=1)
    kernel=torch.exp(source.log()/eps).clamp_min(1.e-10)
    b=kernel.new_tensor([1-running,running])
    v=torch.ones_like(b)
    n=len(selected)
    for _ in range(8):
        u=1/(n*(kernel@v).clamp_min(1.e-12))
        v=(b/(kernel.T@u).clamp_min(1.e-12)).pow(lam/(lam+eps))
    q=kernel*v
    assert torch.isfinite(q).all()
    out=p.clone();out[valid]=q[:,1]/q.sum(1)
    return out,lam,evidence,float(out[valid].mean())

def possible_windows(case,query,rng,n=4):
    centerfield=case.domain(query)
    sign=side(query)
    anatomy=prior(query,case.anatomical[sign],case.ct.shape) if sign else None
    regions=[]
    for _ in range(96):
        center=tuple(rng.integers(np.array(case.ct.shape)))
        if not centerfield[center]:continue
        if anatomy is not None and rng.uniform()>.25+.75*anatomy[center[1:]]:continue
        begin=np.clip(np.array(center)-SIDE//2,0,np.array(case.ct.shape)-SIDE)
        region=slices(begin)
        if not any(region==prior for prior in regions):
            regions.append(region)
        if len(regions)==n:break
    if not regions:
        a=np.array(np.where(centerfield))
        index=int(rng.integers(a.shape[1]))
        begin=np.clip(a[:,index]-SIDE//2,0,np.array(case.ct.shape)-SIDE)
        regions.append(slices(begin))
    return regions

@torch.no_grad()
def pseudo_target(teacher,case,query,rng,quota,running,transport):
    best=None
    for region in possible_windows(case,query,rng):
        x=case.features(region,query)
        valid=torch.from_numpy(case.domain(query)[region][None].copy()).to(DEVICE)
        first=teacher(augment(x,rng),[query]).sigmoid()
        f=first[valid]
        confidence=f.topk(min(len(f),64)).values.mean().item() if len(f) else -1.
        if best is None or confidence>best[0]:
            best=(confidence,x,valid,first)
    _,x,valid,p=best
    second=teacher(augment(x,rng),[query]).sigmoid()
    f=.5*(p+second)
    stable=(p-second).abs()<=.14
    pos=valid & stable & (f>=.54) & (p>=.49) & (second>=.49)
    n=int(pos.sum())
    if n<12:return None,dict(accepted=0,proposed=n,pos=0)
    if n>quota:
        indexes=pos.flatten().nonzero(as_tuple=True)[0]
        take=f.flatten()[indexes].topk(quota).indices
        reduced=torch.zeros_like(pos);reduced.flatten()[indexes[take]]=True
        pos=reduced
    bg=valid & stable & (f<=.10)
    diagnostics=dict(accepted=1,proposed=n,pos=int(pos.sum()),evidence=float(f[valid].mean()))
    if transport:
        f,lam,before,after=adaptive_uot(f,valid,running)
        diagnostics.update(lam=lam,pre_transport=before,post_transport=after)
    return (x,pos,bg,f),diagnostics

def warm(budget,seed,kind,cohort,rows,store):
    folder=OUT/budget/str(seed)/"initial";folder.mkdir(exist_ok=True,parents=True)
    ck=folder/(kind+".pt")
    configure(seed);net=FindingNet().to(DEVICE)
    if ck.exists():
        r=read(ck.with_suffix(".json"))
        assert r["sha"]==sha(ck) and r["plan"]==sha(OUT/"protocol.json")
        return ck
    optimizer=torch.optim.AdamW(net.parameters(),lr=.001,weight_decay=.0001)
    log=[]
    for i,index in enumerate(make_schedule(len(rows),WARM,seed+917),1):
        row=rows[index];case=store.get(row,True);query=row["query"] if kind=="B" else ""
        rng=np.random.default_rng(seed*4093+i)
        loss=labeled_loss(net,row,case,query,rng);update(net,optimizer,loss)
        log.append(float(loss.detach()))
        if i%112==0: print("warm",budget,seed,kind,i,round(float(np.mean(log[-112:])),4),flush=True)
    write(ck.with_suffix(".json"),dict(sha=checkpoint(ck,net),plan=sha(OUT/"protocol.json"),loss=log))
    return ck

def fit_arm(budget,seed,arm):
    cohort=load_cohort()
    rows=exposure(budget,cohort)
    assert read(OUT/"protocol.json")["sha"]==sha(Path(__file__))
    root=OUT/budget/str(seed)/arm;root.mkdir(exist_ok=True,parents=True)
    out=root/"model.pt"
    uses_text=arm not in "AC"
    store=LobeStore(limit=64)
    ck=warm(budget,seed,("B" if uses_text else "A"),cohort,rows,store)
    configure(seed)
    model=FindingNet().to(DEVICE)
    cap=max(3.,3*float(np.median([r["volume_ml"] for r in rows])))
    quota=int(min(4.,.65*cap)*1000/GRID**3)
    if not out.exists():
        model.load_state_dict(torch.load(ck,weights_only=True,map_location=DEVICE))
        teacher=deepcopy(model).eval().requires_grad_(False)
        opt=torch.optim.AdamW(model.parameters(),lr=.0006,weight_decay=.0001)
        unlab=cohort["unlabelled"]
        by_code={r["code"]:r for r in unlab}
        permitted=[r["code"] for r in rows]
        ordered=make_schedule(len(rows),STEPS,seed+22591)
        uorder=make_schedule(len(unlab),STEPS,seed+8828)
        logs=[]; prior_mass=None
        for i,(ix,uix) in enumerate(zip(ordered,uorder),1):
            rng=np.random.default_rng(seed*16529+i)
            row=rows[ix];query=row["query"] if uses_text else ""
            case=store.get(row,True)
            ss=labeled_loss(model,row,case,query,rng)
            diagnostic=dict(accepted=0,proposed=0,pos=0)
            extra=ss.new_zeros(())
            if arm in "CDEF":
                u=unlab[uix];scan=store.get(u,False)
                partner=by_code[u["donor"]] if arm=="E" else u
                text=partner["query"] if uses_text else ""
                check=positive_region(case,np.random.default_rng(seed*18227+i))
                domain=case.domain(query)[check]
                label_volume=max(.00015,min(.06,(case.truth[check]&domain).sum()/max(1,domain.sum())))
                prior_mass=label_volume if prior_mass is None else .95*prior_mass+.05*label_volume
                rng_u=np.random.default_rng(seed*7321+i)
                targets,diagnostic=pseudo_target(teacher,scan,text,rng_u,quota,prior_mass,arm=="F")
                if targets is not None:
                    x,pos,bg,p=targets
                    z=model(augment(x,rng_u,strong=True),[text])
                    l=F.binary_cross_entropy_with_logits(z,p,reduction="none")
                    extra=(.65*l[pos].mean()+.35*l[bg].mean()) if bool(bg.any()) else l[pos].mean()
                diagnostic.update(u=u["code"],text=partner["code"] if uses_text else "empty",label_mass=prior_mass)
            objective=ss+.40*min(1,i/96)*extra
            update(model,opt,objective)
            with torch.no_grad():
                for a,b in zip(teacher.parameters(),model.parameters(),strict=True):a.lerp_(b,1-EMA)
            logs.append(dict(sup=float(ss.detach()),unsup=float(extra.detach()),**diagnostic))
            if i%96==0:print("arm",budget,seed,arm,i,round(float(np.mean([p["sup"] for p in logs[-96:]])),3),
                                 sum(p["accepted"] for p in logs[-96:]),flush=True)
        params=checkpoint(out,model)
        write(root/"training.json",dict(arm=arm,seed=seed,budget=budget,
            model_sha=params,base_sha=sha(ck),protocol=sha(OUT/"protocol.json"),
            train_codes=permitted,supervised_updates=STEPS,logs=logs))
    model.load_state_dict(torch.load(out,weights_only=True,map_location=DEVICE));model.eval()
    report=root/"development.json"
    if report.exists():
        assert read(report)["model_sha"]==sha(out);return
    results=[]
    for row in cohort["dev"]:
        case=store.get(row,True)
        x,tiles=case_eval(model,row,case,row["query"] if uses_text else "",cap,root/(row["code"]+".npz"))
        results.append(dict(code=row["code"],values=x,tiles=tiles,packed_sha=sha(root/(row["code"]+".npz"))))
    write(report,dict(seed=seed,arm=arm,budget=budget,cap_ml=cap,model_sha=sha(out),rows=results))
    print("dev complete",budget,seed,arm,round(float(np.mean([r["values"]["0.5"]["dice"] for r in results])),6),flush=True)

def main():
    os.umask(0o077)
    p=ArgumentParser();p.add_argument("mode",choices=("seal","run"))
    p.add_argument("--budget",choices=("5","10","20"),default=PRIMARY);p.add_argument("--seed",type=int,default=891)
    p.add_argument("--arms",default="AB")
    x=p.parse_args()
    if x.mode=="seal":
        load_cohort();OUT.mkdir(exist_ok=True)
        seal(OUT/"protocol.json",dict(design=PROTOCOL,cohort=sha(LOCK),sha=sha(Path(__file__)),
             model=sha(Path("thesis_model.py")),lobe_model=sha(Path("thesis_lobe_model.py"))))
    else:
        assert x.seed in SEEDS and all(a in ARMS for a in x.arms)
        for arm in x.arms:fit_arm(x.budget,x.seed,arm)
if __name__=="__main__":main()

