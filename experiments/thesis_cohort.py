"""Lock patient roles, nested equal-cost mask budgets and report-only text permutations."""
from collections import Counter,defaultdict
from itertools import combinations
from hashlib import sha256
from pathlib import Path
from typing import Iterable
import os
import numpy as np
from scipy.optimize import linear_sum_assignment

from atelectasis_b0 import GRID,ROOT as PILOT
from atelectasis_screen import anatomy
from audit_pairs import identity
from thesis_expand_screen import ROOT,SOURCES
from thesis_report_only import U
from frozen_priors import read,sha,write
from diag_localize import seal
from swap_data import side

MANIFEST=ROOT/"thesis_private.json"
LOCK=ROOT/"thesis_lock.json"
LABELED=ROOT/"labeled_roles.private.json"
PRIMARY="20"
BUDGETS=(5,10,20)

def rank(v,label):return sha256((label+"|"+v["code"]).encode()).hexdigest()
def site(row):
    lobes,_=anatomy(row["query"])
    return (side(row["query"]) or "unknown")+"_"+(next(iter(lobes)) if len(lobes)==1 else "unknown")
def choose_stratified(entries,n,tag):
    groups=defaultdict(list)
    for row in entries:groups[row["site"]].append(row)
    for k in groups:groups[k].sort(key=lambda r:rank(r,tag))
    order=sorted(groups,key=lambda k:(-len(groups[k]),k))
    result=[]
    while len(result)<n and any(groups[k] for k in order):
        for k in order:
            if groups[k] and len(result)<n:result.append(groups[k].pop(0))
    assert len(result)==n
    return result

def build(early=False):
    os.umask(0o077)
    old=read(PILOT/"pilot_private.json")
    srcs={s["code"]:s for s in read(PILOT/"screen.private.json")}
    conversion={}
    for group in ("train","development","holdout"):
        conversion[group]=[]
        for r in old[group]:
            a=dict(r,patient=":".join(identity(srcs[r["code"]]["name"])[1]),path=str(PILOT/"arrays"/(r["code"]+".npz")),
               source_style="single_sentence",volume_ml=r["coords"]*GRID**3/1000,site=site(r))
            conversion[group].append(a)
    candidates=read(SOURCES)
    assert len(list((ROOT/"receipts").glob("N???.json")))==len(candidates), "Candidate geometry audit is incomplete"
    counts=Counter();fresh=[]
    for r in candidates:
        q=ROOT/"receipts"/(r["code"]+".json")
        if not q.exists():
            counts["not_materialized"]+=1;continue
        receipt=read(q)
        if receipt["status"]!="eligible":
            counts[receipt["reason"]]+=1;continue
        if receipt["components"]!=1:
            counts["multiple_components"]+=1;continue
        if min(receipt["bounding_mm"])>=40:
            counts["broad_target"]+=1;continue
        patient=identity(r["name"])[1]
        a=dict(patient=":".join(patient),code=r["code"],query=r["query"],context=r["context"],published=r["published"],
            side=r["side"],site=site(r),source_style=r["matched_type"],path=str(ROOT/"arrays"/(r["code"]+".npz")),
            sha=receipt["sha"],volume_ml=receipt["voxels"]*GRID**3/1000,coords=receipt["voxels"],
            prior_exposure=r["prior_exposure"])
        fresh.append(a)
    unexposed=[r for r in fresh if not r["prior_exposure"]]
    test_goal=min(35,max(0,round(len(unexposed)*.55)))
    independ=choose_stratified(unexposed,test_goal,"independent-20261119")
    dev_pool=[r for r in fresh if r not in independ]
    ndev=min(8,round(len(dev_pool)*.33))
    dev=choose_stratified(dev_pool,ndev,"development-20261119")
    newtrain=[r for r in dev_pool if r not in dev]
    train=conversion["train"]+newtrain
    nested=choose_stratified(train,max(2,round(.20*len(train))),"mask-select-20261119")
    split=dict(train=train,dev=conversion["development"]+dev,test=conversion["holdout"]+independ)
    budgets={str(q):[r["code"] for r in nested[:max(2,round(q/100*len(train)))]] for q in BUDGETS}
    seal(LABELED,dict(**split,labels=budgets))
    if early:
        print("Labeled roles locked", {k:len(v) for k,v in split.items()}, {k:len(v) for k,v in budgets.items()},flush=True)
        return
    unlabeled=[]
    u_sources=read(U/"source.private.json")
    assert len(list((U/"receipts").glob("U???.json")))==len(u_sources), "Report-only downloads are incomplete"
    for r in u_sources:
        p=U/"receipts"/(r["code"]+".json")
        if not p.exists():continue
        receipt=read(p)
        assert receipt["contains_mask"] is False
        unlabeled.append(dict(code=r["code"],patient=":".join(identity(r["name"])[1]),
            query=r["query"],site=r["site"],path=str(U/"arrays"/(r["code"]+".npz")),sha=receipt["processed_sha"]))
    # A cost-minimizing derangement uses similar-length original text, preferring different anatomical sites.
    n=len(unlabeled)
    assert n>=2
    costs=np.zeros((n,n))
    for i,r in enumerate(unlabeled):
        for j,s in enumerate(unlabeled):
            costs[i,j]=(1000000 if i==j else (100 if r["site"]==s["site"] else 0)
                        +abs(len(r["query"].split())-len(s["query"].split())))
    ids,permutation=linear_sum_assignment(costs)
    assert len(ids)==n and np.all(ids!=permutation)
    for i,j in zip(ids,permutation):unlabeled[i]["donor"]=unlabeled[j]["code"]
    split["unlabelled"]=unlabeled
    for a,b in combinations(split,2):
        assert not ({r["patient"] for r in split[a]}&{r["patient"] for r in split[b]})
    split["labels"]=budgets
    seal(MANIFEST,split)
    census=dict(roles={k:len(split[k]) for k in ("train","dev","test","unlabelled")},
            budget={q:len(v) for q,v in split["labels"].items()},sites={k:dict(Counter(v["site"] for v in split[k]))
                for k in ("train","dev","test","unlabelled")}, primary=PRIMARY,
            expansion=dict(screened=len(candidates),eligible=len(fresh),exclusions=dict(counts),
                           newly_reserved=len(independ),newdev=len(dev),newtrain=len(newtrain)),
            manifest_sha=sha(MANIFEST),candidate_sha=sha(SOURCES),code_sha=sha(Path(__file__)),
            rationale="20% is declared primary before evaluating expanded models because 2 masks in the prior engineering split omitted the predominant development locations. 5% and 10% remain nested secondary budgets.")
    seal(LOCK,census)
    print(census,flush=True)
if __name__=="__main__":
    import sys
    build("--labeled-only" in sys.argv)

