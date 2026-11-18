"""Train and evaluate a text-conditioned 1.5 mm sliding-window extension of B0."""
import argparse
from copy import deepcopy
from itertools import product
import os
from pathlib import Path
from time import monotonic

import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree
import torch
from torch import nn
from torch.nn import functional as F

from atelectasis_b0 import ROOT, LOCK, SHAPE, STEPS, GRID, RULES as BASE_RULES, load_row, coarse_image
from diag_localize import seal
from finding_train import QueryNet, configure, supervised_loss
from frozen_priors import read, sha, write

OUT = ROOT / "sliding"
SIDE = 64
STRIDE = 40
BGRIDS = tuple(BASE_RULES["alternate_dev_thresholds"])
RUNS = (("b10",891), ("b10",1901), ("b10",2707), ("all_labeled",891))
DESIGN = {
    "cohort": "Reuse original development, training pool, 2 patient subset, queries and reference grid.",
    "training": "384 paired updates; same coarse architecture, optimizer, case order, seeds and HU perturbation. Add matched low resolution loss plus positive local and background local loss at 1.5 mm. Whole-volume and patch networks share text embedding and GRU.",
    "local_sampling": "One lesion-bearing crop with a random margin and one random no-overlap background patch per annotated patient update.",
    "fusion": "Fine text-conditioned logit plus learned bounded weight times tanh of sampled coarse logit. Auxiliary full CT global Dice and logistic loss. No evaluation ROI or GT input.",
    "inference": "64-cube (96 mm) at 1.5 mm; stride 40 with edge-aligned end, positive cosine overlap weights, full CT.",
    "metrics": "Fixed 0.5, same 5-threshold development sensitivity; patient and seed pairing; missed means zero true intersection; 10% recall secondary. Overprediction is extra-reference volume; 3 mm symmetric surface Dice on boundary voxels.",
    "decision": "Move toward pseudolabels only if development 2-mask results show non-accidental, useful localization with controlled FP volume, checked against 20-mask supervised anchor. Do not score reserve."
}

class DualScale(nn.Module):
    def __init__(self):
        super().__init__()
        self.global_net = QueryNet()
        self.local_net = deepcopy(self.global_net)
        self.local_net.emb = self.global_net.emb
        self.local_net.gru = self.global_net.gru
        self.fusion = nn.Parameter(torch.zeros(()))

    def combine(self, image, query, prior):
        fine = self.local_net(image, [query])
        return fine + self.fusion.sigmoid() * 2 * prior.tanh()

def windows(length):
    assert length >= SIDE
    points = list(range(0, length - SIDE + 1, STRIDE))
    if points[-1] != length - SIDE:
        points.append(length - SIDE)
    return points

def slices(start):
    return tuple(slice(int(a), int(a) + SIDE) for a in start)

def channels(ct):
    shape = ct.shape
    image = ct.astype(np.float32)
    axes = tuple(np.linspace(-1, 1, n, dtype=np.float32) for n in shape)
    return (np.clip((image+1000)/1200,0,1), np.clip((image+150)/400,0,1)), axes

def take(image, axes, region, device):
    xyz = np.meshgrid(*(a[s] for a,s in zip(axes,region)), indexing="ij")
    return torch.from_numpy(np.stack([*(a[region] for a in image), *xyz]))[None].to(device)

def crop_global(logits, shape, start):
    faces = [(torch.arange(a,a+SIDE,device=logits.device,dtype=logits.dtype)+.5)*2/n-1
             for a,n in zip(start,shape)]
    locs = torch.meshgrid(*faces,indexing="ij")
    query = torch.stack(locs[::-1],dim=-1)
    return F.grid_sample(logits[:,None],query[None],mode="bilinear",padding_mode="border",align_corners=False)[:,0]

def sample(item, rng):
    truth = item["truth"]
    shape = np.asarray(truth.shape)
    center = item["positive"][rng.integers(len(item["positive"]))]
    start = tuple(np.clip(center-rng.integers(16,49,size=3),0,shape-SIDE))
    assert truth[slices(start)].any()
    # Require a mask-free patch. The nearest of four eligible windows supplies anatomical hard negatives.
    empties = []
    for _ in range(64):
        candidate = tuple(rng.integers(0,shape-SIDE+1))
        if not truth[slices(candidate)].any():
            empties.append(candidate)
            if len(empties)==4:break
    assert empties, "CT contains no background window"
    other = min(empties,key=lambda loc:np.linalg.norm(np.asarray(loc)+(SIDE//2)-center))
    return start,other

def augment(tensor,gain,bias):
    return torch.cat(((tensor[:,:2]*gain+bias).clamp(0,1),tensor[:,2:]),1)

def patient_data(row, device):
    ct,truth=load_row(row)
    image,axes=channels(ct)
    target = F.adaptive_avg_pool3d(torch.from_numpy(truth.astype(np.float32))[None,None],SHAPE)[0].to(device)
    assert truth.any() and (target>0).any()
    return dict(name=row["code"],query=row["query"],truth=truth,image=image,axes=axes,positive=np.argwhere(truth),
                coarse=coarse_image(ct).to(device),target=target)

def score(pred,truth,surface=False):
    n=int(pred.sum());total=int(truth.sum());hit=int(np.count_nonzero(pred&truth))
    metrics = dict(true=total,pred=n,intersection=hit,dice=2*hit/(n+total),
                   recall=hit/total,fp_ml=(n-hit)*GRID**3/1000,
                   empty=(n==0),miss=(hit==0))
    if surface:
        metric=0.
        if n:
            boundary=ndimage.generate_binary_structure(3,1)
            a=np.argwhere(pred & ~ndimage.binary_erosion(pred,structure=boundary))
            b=np.argwhere(truth & ~ndimage.binary_erosion(truth,structure=boundary))
            close=(cKDTree(a).query(b)[0]<=3/GRID).sum()+(cKDTree(b).query(a)[0]<=3/GRID).sum()
            metric=float(close/(len(a)+len(b)))
        metrics["surface_dice_3mm"]=metric
    return metrics

@torch.inference_mode()
def sliding(net,ct,query,dev):
    full_tensor=coarse_image(ct).to(dev)
    base=net.global_net(full_tensor,[query])
    coarse=F.interpolate(base[:,None],size=ct.shape,mode="trilinear",align_corners=False)[0,0]
    image,axes = channels(ct)
    taper=torch.from_numpy((.2+.8*np.hanning(SIDE)).astype(np.float32)).to(dev)
    weights=taper[:,None,None]*taper[None,:,None]*taper[None,None,:]
    weighted = torch.zeros(ct.shape,device=dev)
    counts = torch.zeros_like(weighted)
    n=0
    for start in product(*(windows(d) for d in ct.shape)):
        region=slices(start)
        candidate=net.combine(take(image,axes,region,dev),query,coarse[region][None])[0]
        weighted[region] += candidate*weights
        counts[region] += weights
        n+=1
    assert counts.min() > 0 and torch.isfinite(weighted).all()
    fine=weighted/counts
    return fine.cpu().numpy(),coarse.cpu().numpy(),n

def train(budget,seed):
    os.umask(0o077);OUT.mkdir(exist_ok=True)
    pilot=read(LOCK);manifest=ROOT/"pilot_private.json"
    assert sha(manifest)==pilot["manifest_sha"]
    seal(OUT/"protocol.json",dict(design=DESIGN,code_sha=sha(Path(__file__)),pilot_lock=sha(LOCK),
                                  manifest_sha=sha(manifest),runs=[list(k) for k in RUNS],steps=STEPS))
    data=read(manifest)
    train_rows=data["selected" if budget=="b10" else "train"]
    target_codes={r["code"] for r in train_rows}
    dev_codes={r["code"] for r in data["development"]}
    assert len(train_rows)==(2 if budget=="b10" else 20) and len(dev_codes)==6 and not target_codes&dev_codes
    old_dir=ROOT/budget/str(seed)
    old_scores=read(old_dir/"results.json")
    assert target_codes==set(old_scores["train_codes"]) and old_scores["lock_sha"]==sha(LOCK)
    assert sha(old_dir/"B0.pt")==old_scores["model_sha"]
    assert set(s["code"] for s in old_scores["scores"])==dev_codes
    configure(seed)
    out=OUT/budget/str(seed);out.mkdir(exist_ok=True,parents=True)
    ck=out/"B1.pt"
    if (out/"results.json").exists():
        assert sha(ck)==read(out/"results.json")["model_sha"];return
    device=torch.device("cuda")
    start=monotonic()
    net=DualScale().to(device)
    params=sum(x.numel() for x in net.parameters())
    opt=torch.optim.AdamW(net.parameters(),lr=.001,weight_decay=.0001)
    rng=np.random.default_rng(seed)
    sampler=np.random.default_rng(seed+8707)
    schedule=np.concatenate([rng.permutation(len(train_rows)) for _ in range((STEPS+len(train_rows)-1)//len(train_rows))])[:STEPS]
    logs=[]
    if not ck.exists():
        items=[patient_data(row,device) for row in train_rows]
        print("training",budget,seed,"patients",len(items),"parameters",params,flush=True)
        for step,index in enumerate(schedule,1):
            item=items[index]
            gain=float(rng.uniform(.88,1.12)); bias=float(rng.uniform(-.045,.045))
            positive,background=sample(item,sampler)
            query=item["query"]
            glob=net.global_net(augment(item["coarse"],gain,bias),[query])
            global_loss=supervised_loss(glob,item["target"])
            local_losses=[]
            for k,loc in enumerate((positive,background)):
                region=slices(loc)
                scan=augment(take(item["image"],item["axes"],region,device),gain,bias)
                logits=net.combine(scan,query,crop_global(glob,item["truth"].shape,loc))
                if k==0:
                    ys=torch.from_numpy(item["truth"][region].astype(np.float32))[None].to(device)
                    local_losses.append(supervised_loss(logits,ys))
                else:
                    # Penalize the full background and the tail most likely to survive thresholding.
                    neg=F.softplus(logits.flatten())
                    local_losses.append(.5*(neg.mean()+neg.topk(max(1,neg.numel()//100)).values.mean()))
            loss=global_loss+.5*sum(local_losses)
            opt.zero_grad(set_to_none=True);loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(),5,error_if_nonfinite=True);opt.step()
            logs.append([float(global_loss.detach()),*[float(x.detach()) for x in local_losses]])
            if step%32==0:print("train",budget,seed,step, np.mean(logs[-32:],axis=0).round(5).tolist(),flush=True)
        del items
        write(out/"training.json",dict(loss=logs,steps=STEPS,cohort_hash=sha(manifest)))
        temp=ck.with_suffix(".tmp");torch.save(net.state_dict(),temp);temp.replace(ck)
    assert len(read(out/"training.json")["loss"])==STEPS
    net.load_state_dict(torch.load(ck,weights_only=True,map_location=device));net.eval()
    comparison=[]
    for row in data["development"]:
        c=row["code"];record=out/(c+".json");archive=out/(c+".npz")
        if record.exists():
            cached=read(record)
            assert cached["prediction_sha"]==sha(archive)
            comparison.append(cached)
            continue
        start_case=monotonic()
        ct,truth=load_row(row)
        fine,coarse,n=sliding(net,ct,row["query"],device)
        rounded=tuple(fine.shape)
        old_archive=old_dir/(c+".npz")
        old_row=next(s for s in old_scores["scores"] if s["code"]==c)
        assert old_row["prediction_sha"]==sha(old_archive)
        with np.load(old_archive) as a:
            assert tuple(a["shape"])==rounded
            old=F.interpolate(torch.from_numpy(a["probability_coarse_logits"])[None,None],size=rounded,
                              mode="trilinear",align_corners=False)[0,0].numpy()
            old_bits=a["threshold_half"].copy()
        matched={}
        modes=[("B0",old),("B1",fine),("B1_coarse",coarse)]
        bits={}
        for key,array in modes:
            metrics={}
            for p in BGRIDS:
                binary=array>=np.log(p/(1-p))
                metrics[str(p)]=score(binary,truth,surface=p==.5)
                if key=="B0":
                    assert np.isclose(metrics[str(p)]["dice"],old_row["dice"][str(p)])
                    if p==.5:assert np.array_equal(np.packbits(binary.ravel()),old_bits)
                if p==.5:bits[key]=np.packbits(binary.ravel())
            matched[key]=metrics
        occupancy=F.adaptive_avg_pool3d(torch.from_numpy(truth.astype(np.float32))[None,None],SHAPE)[0,0]
        np.savez_compressed(archive,**bits,shape=rounded)
        with np.load(archive) as a:
            for k,_ in modes:
                pred=np.unpackbits(a[k],count=truth.size).reshape(truth.shape).astype(bool)
                assert matched[k]["0.5"]==score(pred,truth,surface=True)
        case=dict(code=c,metrics=matched,tiles=n,covering_fraction=1.,grid_mm=GRID,
                  match_old=old_row["prediction_sha"],prediction_sha=sha(archive),
                  occupancy_cells=int(torch.count_nonzero(occupancy)),occupancy_half_cells=int(torch.count_nonzero(occupancy>=.5)),
                  seconds=monotonic()-start_case)
        write(record,case);comparison.append(case)
        print("scored",budget,seed,c,n,round(matched["B1"]["0.5"]["dice"],6),round(case["seconds"],1),flush=True)
    results=dict(budget=budget,seed=seed,cohort_hash=sha(manifest),model_sha=sha(ck),reference_sha=sha(old_dir/"B0.pt"),
                 fusion=float(net.fusion.sigmoid().detach()*2),parameters=params,coarse_parameters=sum(x.numel() for x in net.global_net.parameters()),
                 protocol=sha(OUT/"protocol.json"),train_codes=sorted(target_codes),elapsed_seconds=monotonic()-start,rows=comparison)
    write(out/"results.json",results)
if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("budget",choices=("b10","all_labeled"))
    parser.add_argument("seed",type=int)
    a=parser.parse_args();assert (a.budget,a.seed) in RUNS
    train(a.budget,a.seed)

