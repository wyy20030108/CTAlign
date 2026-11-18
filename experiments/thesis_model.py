"""Anatomically referenced supervised and teacher-student segmentation utilities."""
from collections import OrderedDict
from dataclasses import dataclass
from itertools import product
from pathlib import Path
import os,time

import numpy as np
from scipy import ndimage as nd
import torch
from torch import nn
from torch.nn import functional as F

from atelectasis_b0 import GRID,ROOT as PILOT
from atelectasis_locate_refine import lung_field,prior
from atelectasis_sliding import SIDE,windows,slices,score
from atelectasis_screen import anatomy
from finding_train import QueryNet,configure
from frozen_priors import read,sha,write
from thesis_expand_screen import ROOT
from swap_data import side

SIZE=64
THRESHOLDS=(.25,.4,.5,.6,.7,.8)
DEVICE=torch.device("cuda")
OUT=ROOT/"models"
class FindingNet(QueryNet):
    def __init__(self):
        super().__init__()
        self.a[0]=nn.Conv3d(8,12,3,padding=1)
        nn.init.constant_(self.base.bias,-2)

@dataclass
class Case:
    code:str
    ct:np.ndarray
    truth:np.ndarray|None
    detail:np.ndarray
    side_masks:dict
    axes:tuple
    anatomical:dict
    positive:np.ndarray|None

    def domain(self,text):
        which=side(text)
        return self.side_masks[which] if which else self.side_masks["union"]

    def features(self,region,text):
        hu=self.ct[region].astype(np.float32)
        xyz=np.meshgrid(*(a[s] for a,s in zip(self.axes,region)),indexing="ij")
        where=side(text)
        p=prior(text,self.anatomical[where],self.ct.shape)[region[1:]] if where else None
        guide=np.broadcast_to(p[None],hu.shape) if p is not None else np.zeros(hu.shape,np.float32)
        return torch.from_numpy(np.stack((np.clip((hu+1000)/1200,0,1), np.clip((hu+150)/400,0,1),
               *xyz, self.domain(text)[region].astype(np.float32),self.detail[region].astype(np.float32)/600, guide)))[None].to(DEVICE)

class CaseStore:
    def __init__(self,limit=10):
        self.cache=OrderedDict()
        self.limit=limit
    def get(self,row,labeled):
        key=(row["code"],labeled)
        if key not in self.cache:
            p=Path(row["path"]);assert sha(p)==row["sha"]
            with np.load(p) as a:
                assert labeled==("mask" in a.files)
                ct=a["ct"];m=a["mask"].astype(bool) if labeled else None
            lung,body,mid=lung_field(ct)
            rois={};bounds={}
            for k,v in lung.items():
                rois[k]=nd.binary_dilation(v,iterations=6)&body
                _,y,z=np.where(v)
                bounds[k]=int(y.min()),int(y.max()),int(z.min()),int(z.max())
            rois["union"]=rois["left"]|rois["right"]
            xyz=np.where(rois["union"])
            axes=[]
            for d,v in enumerate(xyz):
                if d==0:
                    width=max(mid-v.min(),v.max()-mid)
                    arr=(np.arange(ct.shape[d])-mid)/max(1,width)
                else:
                    arr=(np.arange(ct.shape[d])-v.min())*2/max(1,(v.max()-v.min()))-1
                axes.append(np.clip(arr,-1.25,1.25).astype(np.float32))
            soft=nd.uniform_filter(ct.astype(np.float32),size=7)
            detail=np.clip(ct-soft,-600,600).astype(np.int16)
            data=Case(row["code"],ct,m,detail,rois,tuple(axes),bounds,np.argwhere(m) if labeled else None)
            self.cache[key]=data
            if len(self.cache)>self.limit:self.cache.popitem(last=False)
        self.cache.move_to_end(key)
        return self.cache[key]

def positive_region(case,rng):
    point=case.positive[rng.integers(len(case.positive))]
    orig=np.clip(point-rng.integers(16,49,size=3),0,np.array(case.ct.shape)-SIDE)
    return slices(orig)

def negative_region(case,text,rng):
    mask=case.domain(text)
    for _ in range(36):
        center=tuple(rng.integers(np.array(case.ct.shape)))
        if not mask[center]:continue
        start=np.clip(np.array(center)-SIDE//2,0,np.array(case.ct.shape)-SIDE)
        region=slices(start)
        if not case.truth[region].any():return region
    for ix in product(*(windows(d) for d in case.ct.shape)):
        region=slices(ix)
        if mask[region].mean()>.02 and not case.truth[region].any():return region
    raise ValueError("No eligible verified background window")

def augment(x,rng,strong=False):
    width=.1 if strong else .05
    gain=float(rng.uniform(1-width,1+width));bias=float(rng.uniform(-width,width))
    z=x.clone()
    z[:,:2]=(z[:,:2]*gain+bias).clamp(0,1)
    if strong:z[:,:2]=(z[:,:2]+torch.randn_like(z[:,:2])*.02).clamp(0,1)
    return z

def hard_background(values):
    assert values.numel()>0
    return .45*values.mean()+.55*values.topk(max(1,values.numel()//100)).values.mean()

def supervised(z,y):
    p=z.sigmoid()
    neg=hard_background(F.softplus(z[~y]))
    if not y.any():return .45*neg
    tp=p[y].sum()
    fp=p[~y].sum()
    fn=y.sum()-tp
    tversky=(2+tp)/(2+tp+.35*fp+.65*fn)
    return (.65*(1-tversky)+.50*F.softplus(-z[y]).mean()+.45*neg)

def labeled_loss(net,row,case,text,rng):
    reg1=positive_region(case,rng)
    reg2=negative_region(case,text,rng)
    results=[]
    for reg in (reg1,reg2):
        x=augment(case.features(reg,text),rng)
        y=torch.from_numpy(case.truth[reg][None].copy()).to(DEVICE)
        results.append(supervised(net(x,[text]),y))
    return results[0]+.5*results[1]

@torch.inference_mode()
def volume_logits(net,case,text):
    shape=case.ct.shape
    domain=case.domain(text)
    weighted=torch.zeros(shape,device=DEVICE)
    counts=torch.zeros_like(weighted)
    ax=torch.from_numpy((.2+.8*np.hanning(SIDE)).astype(np.float32)).to(DEVICE)
    weight=ax[:,None,None]*ax[None,:,None]*ax[None,None,:]
    n=0
    for loc in product(*(windows(i) for i in shape)):
        reg=slices(loc)
        if not domain[reg].any():continue
        x=case.features(reg,text);p=net(x,[text])[0]
        weighted[reg]+=p*weight;counts[reg]+=weight;n+=1
    count_np=counts.cpu().numpy()
    assert np.all(count_np[domain]>0) and n>0
    result=(weighted/counts.clamp_min(1.e-6)).cpu().numpy()
    result[~domain]=-40
    return result,n

def segment(scores,threshold,cap_ml):
    high=scores>=np.log(threshold/(1-threshold))
    lab,n=nd.label(high,np.ones((3,3,3),bool))
    if not n:return high
    sizes=np.bincount(lab.ravel());sizes[0]=0
    ids=np.flatnonzero(sizes>=6)
    if not len(ids):return np.zeros_like(high)
    peak=nd.maximum(scores,labels=lab,index=ids)
    # This data set supplies a single annotated target per query.
    priority=peak+.14*np.log1p(sizes[ids])
    winner=ids[np.argmax(priority)]
    foreground=lab==winner
    cap=int(cap_ml*1000/GRID**3)
    if foreground.sum()>cap:
        indices=np.flatnonzero(foreground)
        best=np.argpartition(scores.ravel()[indices],-cap)[-cap:]
        foreground.fill(False)
        foreground.ravel()[indices[best]]=True
    return foreground

def case_eval(net,row,case,query,cap,save=None):
    logits,tiles=volume_logits(net,case,query)
    ret={}
    for t in THRESHOLDS:
        mask=segment(logits,t,cap)
        ret[str(t)]=score(mask,case.truth,surface=t==.5)
        if save is not None and t==.5:
            np.savez_compressed(save,shape=case.ct.shape,prediction=np.packbits(mask.ravel()))
    return ret,tiles

def cohort_pilot():
    source=read(PILOT/"pilot_private.json")
    def with_path(row):return dict(path=str(PILOT/"arrays"/(row["code"]+".npz")),**row)
    return {g:[with_path(q) for q in source[g]] for g in ("train","selected","development")}

def fit_pilot(budget,seed):
    configure(seed)
    store=CaseStore(limit=24)
    data=cohort_pilot()
    rows=data["selected" if budget=="two" else "train"]
    dest=OUT/"development_pilot"/budget/str(seed);dest.mkdir(exist_ok=True,parents=True)
    protocol=Path(__file__)
    model=FindingNet().to(DEVICE)
    opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
    rng=np.random.default_rng(seed)
    order=np.concatenate([rng.permutation(len(rows)) for _ in range((384+len(rows)-1)//len(rows))])[:384]
    cap=max(3.,3*np.median([r["coords"]*GRID**3/1000 for r in rows]))
    ck=dest/"model.pt"
    if not ck.exists():
        for step,i in enumerate(order,1):
            row=rows[i]
            case=store.get(row,True)
            loss=labeled_loss(model,row,case,row["query"],rng)
            opt.zero_grad(set_to_none=True);loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),5,error_if_nonfinite=True);opt.step()
            if step%48==0:print("pilot",budget,seed,step,float(loss.detach()),flush=True)
        temp=ck.with_suffix(".part");torch.save(model.state_dict(),temp);temp.replace(ck)
    model.load_state_dict(torch.load(ck,weights_only=True,map_location=DEVICE));model.eval()
    results=[]
    for row in data["development"]:
        ret,nt=case_eval(model,row,store.get(row,True),row["query"],float(cap),dest/(row["code"]+".npz"))
        results.append(dict(code=row["code"],thresholds=ret,tiles=nt,sha=sha(dest/(row["code"]+".npz"))))
        print("score",budget,seed,row["code"],ret["0.5"]["dice"],flush=True)
    write(dest/"results.json",dict(dev=results,seed=seed,train=[r["code"] for r in rows],budget=budget,
                                    code_sha=sha(protocol),checkpoint=sha(ck),cap_ml=float(cap)))
if __name__=="__main__":
    os.umask(0o077)
    import argparse
    p=argparse.ArgumentParser();p.add_argument("budget",choices=("two","all"));p.add_argument("seed",type=int)
    a=p.parse_args()
    fit_pilot(a.budget,a.seed)

