"""Freeze a focal atelectasis cohort and train the full-volume low-resolution B0 baseline."""
import argparse, hashlib, json, os, time
from collections import Counter
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from atelectasis_screen import ROOT, side
from atelectasis_prepare import DATA, RECEIPTS, GRID
from diag_localize import seal
from finding_train import QueryNet, configure, photo, supervised_loss
from frozen_priors import read, sha, write

SHAPE=(96,96,96)
STEPS=384
LOCK=ROOT/"pilot_lock.json"
RULES=dict(input_grid=SHAPE,source_grid_mm=GRID,whole_CT=True,labeled_fraction=.10,
    seeds=[891,1901,2707],steps=STEPS,loss="fractional occupancy, soft Dice and foreground/background balancing",
    optimizer="AdamW 0.001, clipping at 5",threshold=.5,alternate_dev_thresholds=[.2,.35,.5,.65,.8],
    cohort="One CT/query/target per unique patient. Prior allocations cannot enter reserved test. This first run uses development patients only for scoring.",
    endpoint="Dice against full published finding channel on the 1.5 mm evaluation grid; all outside-target volume counts. No reference coordinates at inference.")
# Manual source review before observing model scores.
EXCLUDE={"L009":"English report has a corrupt linear-band term and the published wording silently repairs it.",
         "L040":"Published text describes plural findings at an unspecific shared lobar location.",
         "L047":"Source and release describe preferential involvement inside a lobe; target extent is ambiguous."}

def freeze():
    rows=read(ROOT/"screen.private.json")
    assert all((RECEIPTS/(r["code"]+".json")).exists() or (RECEIPTS/(r["code"]+".pending.json")).exists() for r in rows)
    inclusion=[]
    counts=Counter()
    for r in rows:
        code=r["code"];f=RECEIPTS/(code+".json")
        if not f.exists():counts["download_unresolved"]+=1;continue
        qc=read(f)
        if qc["status"]!="eligible":counts[qc["reason"]]+=1;continue
        if code in EXCLUDE:counts["source_review_exclusion"]+=1;continue
        if qc["components"]>1:
            counts["multiple_components_on_1p5mm_grid"]+=1;continue
        if qc["bounding_mm"] and min(qc["bounding_mm"])>=40:
            counts["broad_target_review"]+=1;continue
        inclusion.append(dict(code=code,query=r["query"],published=r["published"],side=r["side"],prior=r["prior"],
                              sha=qc["sha"],coords=qc["voxels"]))
    # Allocate before training and select a single stratified set of masks at the 10% budget.
    def order(items,key):return sorted(items,key=lambda r:hashlib.sha256((key+r["code"]).encode()).digest())
    untouched=order([r for r in inclusion if not r["prior"]],"holdout-v1")
    holdout_count=max(2,round(len(inclusion)*.20))
    assert len(untouched)>=holdout_count+2
    holdout=untouched[:holdout_count]
    remaining=order([r for r in inclusion if r not in holdout],"dev-v1")
    dev_count=max(2,round(len(inclusion)*.20))
    dev=remaining[:dev_count]; train=remaining[dev_count:]
    nominal=max(1,int(round(.1*len(train))))
    selected=[]
    for index,location in enumerate(("left","right")):
        if index>=nominal:break
        pool=[r for r in train if r["side"]==location]
        if pool:selected.append(order(pool,"budget-10-v1")[0])
    selected.extend(order([r for r in train if r not in selected],"budget-10-v1")[:nominal-len(selected)])
    assert len(selected)==nominal and len(train)+len(dev)+len(holdout)==len(inclusion)
    info=dict(screened=len(rows),technical_pass=sum(read(f)["status"]=="eligible" for f in RECEIPTS.glob("L???.json")),
              included=len(inclusion),exclusion_counts=dict(counts),manual_source_exclusions=EXCLUDE,
              split=dict(training_pool=len(train),labeled_10_percent=len(selected),development=len(dev),reserved=len(holdout)),
              retained_volume_ml=[round(x*GRID**3/1000,4) for x in sorted(r["coords"] for r in inclusion)],
              selected_codes=[r["code"] for r in selected],protocol=RULES)
    seal(ROOT/"pilot_private.json",dict(train=train,selected=selected,development=dev,holdout=holdout))
    seal(LOCK,dict(stats=info,manifest_sha=sha(ROOT/"pilot_private.json"),script_sha=sha(Path(__file__))))
    print(json.dumps(info),flush=True)

def load_row(row):
    path=DATA/(row["code"]+".npz")
    assert sha(path)==row["sha"]
    with np.load(path) as a:return a["ct"],a["mask"]
def coarse_image(ct):
    arr=torch.from_numpy(ct.astype(np.float32))[None,None]
    small=F.interpolate(arr,size=SHAPE,mode="trilinear",align_corners=False)[0]
    axis=torch.linspace(-1,1,SHAPE[0])
    xyz=torch.stack(torch.meshgrid(axis,axis,axis,indexing="ij"))
    return torch.cat((((small+1000)/1200).clamp(0,1),((small+150)/400).clamp(0,1),xyz))[None]
def train(seed,budget):
    lock=read(LOCK)
    assert lock["script_sha"]==sha(Path(__file__)) and lock["manifest_sha"]==sha(ROOT/"pilot_private.json")
    data=read(ROOT/"pilot_private.json")
    rows=data["selected"] if budget=="b10" else data["train"]
    assert len(rows)>0
    dest=ROOT/budget/str(seed)
    dest.mkdir(parents=True,exist_ok=True)
    done=dest/"results.json"
    if done.exists():
        assert read(done)["model_sha"]==sha(dest/"B0.pt")
        return
    configure(seed)
    rng=np.random.default_rng(seed)
    items=[]
    for row in rows:
        ct,m=load_row(row);x=coarse_image(ct).cuda()
        mask=F.adaptive_avg_pool3d(torch.from_numpy(m.astype(np.float32))[None,None],SHAPE)[0].cuda()
        assert mask.max()>0
        items.append((row,x,mask))
    net=QueryNet().cuda()
    optimizer=torch.optim.AdamW(net.parameters(),lr=.001,weight_decay=.0001)
    schedule=np.concatenate([rng.permutation(len(items)) for _ in range((STEPS+len(items)-1)//len(items))])[:STEPS]
    logs=[]
    for j,k in enumerate(schedule,1):
        row,x,y=items[k]
        logit=net(photo(x,rng),[row["query"]])
        loss=supervised_loss(logit,y)
        optimizer.zero_grad(set_to_none=True);loss.backward()
        grad=torch.nn.utils.clip_grad_norm_(net.parameters(),5,error_if_nonfinite=True)
        optimizer.step();logs.append(float(loss.detach()))
        if j%32==0:
            print(budget,seed,j,round(float(np.mean(logs[-32:])),5),flush=True)
    ck=dest/"B0.pt"
    torch.save(net.state_dict(),ck.with_suffix(".tmp"));ck.with_suffix(".tmp").replace(ck)
    net.eval()
    scores=[]
    with torch.inference_mode():
        for row in data["development"]:
            ct,truth=load_row(row)
            small=coarse_image(ct).cuda()
            coarse=net(small,[row["query"],""]).cpu().numpy()
            del small
            score=dict(code=row["code"],truth=int(truth.sum()),dice={},pred={},empty_text_dice={})
            for key,mat in zip(("correct","empty_text"),coarse):
                p=F.interpolate(torch.from_numpy(mat)[None,None],size=ct.shape,mode="trilinear",align_corners=False)[0,0].numpy()
                assert np.isfinite(p).all() and p.shape==truth.shape
                for prob in RULES["alternate_dev_thresholds"]:
                    cut=np.log(prob/(1-prob));mask=p>=cut
                    num=int(mask.sum());intersect=int(np.count_nonzero(mask&truth))
                    d=2*intersect/(num+score["truth"])
                    if key=="correct":
                        score["pred"][str(prob)]=num
                        score["dice"][str(prob)]=d
                    else: score["empty_text_dice"][str(prob)]=d
                if key=="correct":
                    name=dest/(row["code"]+".npz")
                    np.savez_compressed(name,probability_coarse_logits=mat,threshold_half=np.packbits((p>=0).ravel()),shape=ct.shape)
                    score.update(prediction_sha=sha(name))
                del p
            scores.append(score)
    write(done,dict(scores=scores,loss=logs,model_sha=sha(ck),train_codes=[x["code"] for x in rows],
        lock_sha=sha(LOCK),elapsed=time.time(),seed=seed,budget=budget))
    print("scored",budget,seed,len(scores),float(np.mean([r["dice"]["0.5"] for r in scores])),flush=True)
if __name__=="__main__":
    os.umask(0o077);p=argparse.ArgumentParser()
    p.add_argument("stage",choices=("freeze","train"))
    p.add_argument("--seed",type=int,default=891);p.add_argument("--budget",choices=("b10","all_labeled"),default="b10")
    args=p.parse_args()
    if args.stage=="freeze":freeze()
    else:train(args.seed,args.budget)

