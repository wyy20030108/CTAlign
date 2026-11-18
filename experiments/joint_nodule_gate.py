"""Use the same CT-derived nodule proposals in every arm of the nodule experiment."""
from functools import lru_cache
from pathlib import Path
import re

import numpy as np
from scipy import ndimage as ndi
import torch

import thesis_model as image
from atelectasis_locate_refine import lung_field
from frozen_priors import read, sha, write
from joint_factorial import sharpen, transport
from joint_nodule_cohort import DEST
from joint_nodule_morphology import proposals
from swap_data import side
from thesis_cohort import site
from thesis_lobe_model import CODES


LOCATIONS=DEST/"shape_candidates"
ORIGINAL_LOGITS=image.volume_logits
MAXIMUM=16
NUMBER=r"\d+(?:\.\d+)?"
SIZE=re.compile(rf"\b({NUMBER}(?:\s*(?:[x×*]|by)\s*{NUMBER}){{0,2}})\s*(mm|cm)\b",re.I)


def report_volume(query):
    """Approximate an explicitly reported size as an ellipsoid in voxel units."""
    matches=SIZE.findall(query)
    if len(matches)!=1:
        return None
    values,unit=matches[0]
    dims=sorted(float(v)*(10 if unit.lower()=="cm" else 1) for v in re.findall(NUMBER,values))
    if dims[0]<2 or dims[-1]>36:
        return None
    axes=([dims[0]]*(3-len(dims)))+dims
    return float(np.pi/6*np.prod(axes)/(image.GRID**3))


def prepare(rows):
    LOCATIONS.mkdir(parents=True,exist_ok=True)
    for row in rows:
        path=LOCATIONS/(row["code"]+".json")
        if path.is_file():
            card=read(path)
            assert card["source"]==row["sha"] and card["schema"]==sha(Path(__file__))
            continue
        assert sha(Path(row["path"]))==row["sha"]
        with np.load(row["path"]) as a:ct=a["ct"]
        lungs,body,_=lung_field(ct)
        data={}
        for k,mask in lungs.items():
            domain=ndi.binary_dilation(mask,iterations=5)&body
            data[k]=sorted(proposals(ct,domain),key=lambda x:x[2],reverse=True)[:600]
        write(path,dict(source=row["sha"],schema=sha(Path(__file__)),raw=data))
        print("shape candidates",row["code"],flush=True)


@lru_cache(maxsize=96)
def raw_candidates(code):
    data=read(LOCATIONS/(code+".json"))
    assert data["schema"]==sha(Path(__file__))
    return data["raw"]


def candidate_points(case,query):
    points=raw_candidates(case.code)
    which=side(query)
    label=CODES.get(site({"query":query})) if query else None
    possible=points[which] if which else points["left"]+points["right"]
    if label is not None:
        margin=case.allowed.get(label)
        if margin is None:
            margin=ndi.binary_dilation(case.lobes==label,iterations=4)
            case.allowed[label]=margin
    else:
        margin=None
    def rank(row):
        return row[2]*(1.35 if margin is not None and margin[tuple(row[0])] else 1.)
    selected=[]
    for row in sorted(possible,key=rank,reverse=True):
        p=np.array(row[0],int)
        if case.domain(query)[tuple(p)] and all(np.linalg.norm(t-p)>=6 for t in selected):
            selected.append(p)
        if len(selected)==MAXIMUM:
            break
    return selected


def guidance(case,query):
    # Cache one exact text cue at a time per volume. Training uses the stored shape geometry for each case.
    key=side(query) or "both",site({"query":query}) if query else ""
    saved=getattr(case,"nodule_guidance",None)
    if saved is not None and saved[0]==key:
        return saved[1]
    maxima=np.zeros(case.ct.shape,np.float32)
    shape=np.array(maxima.shape)
    for i,p in enumerate(candidate_points(case,query)):
        lo=np.maximum(0,p-10)
        hi=np.minimum(shape,p+11)
        region=tuple(slice(a,b) for a,b in zip(lo,hi,strict=True))
        coords=np.ogrid[tuple(slice(a-b,c-b) for a,b,c in zip(lo,p,hi,strict=True))]
        strength=(.65+.35*np.exp(-i/5))*np.exp(-sum(a*a for a in coords)/18.)
        maxima[region]=np.maximum(maxima[region],strength)
    # Storing uint8 keeps the full training cache within the machine's RAM envelope.
    result=np.rint(255*maxima).astype(np.uint8)
    case.nodule_guidance=(key,result)
    return result


def offset(mask):
    return 3.0*(mask.astype(np.float32)/255.-.30)


def gated_volume(net,case,query):
    logits,tiles=ORIGINAL_LOGITS(net,case,query)
    logits += offset(guidance(case,query))
    return logits,tiles


@torch.no_grad()
def gated_target(teacher,case,query,origin,rng,prior,use_uot,mode,cap):
    region=image.slices(origin)
    x=case.features(region,query)
    valid=torch.from_numpy(case.domain(query)[region].copy()).to(image.DEVICE)
    anatomy=guidance(case,query)[region]
    uplift=torch.from_numpy(offset(anatomy)).to(image.DEVICE)
    a=(teacher(image.augment(x,rng),[query])[0]+uplift).sigmoid()
    b=(teacher(image.augment(x,rng),[query])[0]+uplift).sigmoid()
    p=.5*(a+b)
    stable=valid&((a-b).abs()<=.15)
    support=stable&torch.from_numpy((anatomy>=51).copy()).to(image.DEVICE)
    if int(support.sum())<16:
        return None,(0,0,None,None,None)
    q=p.clone()
    strength=before=after=None
    if use_uot:
        local_prior=float(np.clip(prior*int(stable.sum())/int(support.sum()),.0001,.15))
        stated=report_volume(query)
        if stated is not None:
            text_prior=float(np.clip(stated/int(support.sum()),local_prior/4,local_prior*4))
            local_prior=float(np.clip(np.sqrt(local_prior*text_prior),.0001,.15))
        q[support],strength,before,after=transport(p[support],local_prior,mode)
    soft=sharpen(q)
    pos=support&(soft>=.72)&(p>=.40)
    n=int(pos.sum())
    if n<12:
        return None,(0,n,strength,before,after)
    if n>cap:
        ids=pos.flatten().nonzero(as_tuple=True)[0]
        chosen=ids[soft.flatten()[ids].topk(cap).indices]
        pos.fill_(False)
        pos.flatten()[chosen]=True
    neg=stable&(soft<=.10)&(p<=.20)
    return (x,pos,neg,soft),(1,int(pos.sum()),strength,before,after)


def activate(experiment=None):
    image.volume_logits=gated_volume
    if experiment is not None:
        experiment.teacher_target=gated_target
