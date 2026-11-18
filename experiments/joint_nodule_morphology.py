"""Check whether compact high-density nodule candidates improve spatial recall."""
from argparse import ArgumentParser
from pathlib import Path
import os

import numpy as np
from scipy import ndimage as ndi

from atelectasis_locate_refine import lung_field
from frozen_priors import read, sha, write
from joint_nodule_cohort import CASE, DEST


SCALES=(1.,1.7,2.6)
MAXIMA=(1,4,8,16)
OUT=DEST/"morphology_dev"


def proposals(ct, domain):
    points=np.where(domain)
    limits=[slice(max(0,int(p.min())-10),min(d,int(p.max())+11)) for p,d in zip(points,ct.shape,strict=True)]
    start=np.array([a.start for a in limits])
    slab=np.clip(ct[tuple(limits)].astype(np.float32),-1000,400)/1000
    eligible=domain[tuple(limits)]&(slab>-.75)
    rows=[]
    components=((0,0),(1,1),(2,2),(0,1),(0,2),(1,2))
    for sigma in SCALES:
        diagonal=[ndi.gaussian_filter(slab,sigma,order=tuple(2 if k==j else 0 for k in range(3))) for j in range(3)]
        amplitude=-sum(diagonal)*sigma**2
        steep=amplitude==ndi.maximum_filter(amplitude,size=5)
        pos=np.flatnonzero(steep&eligible&(amplitude>0))
        if not len(pos):
            continue
        n=min(len(pos),3000)
        where=pos[np.argpartition(amplitude.ravel()[pos],-n)[-n:]]
        hessian=np.zeros((n,3,3),np.float32)
        for i,j in components:
            if i==j:
                value=diagonal[i]
            else:
                value=ndi.gaussian_filter(slab,sigma,order=tuple(int(k in (i,j)) for k in range(3)))
            hessian[:,i,j]=hessian[:,j,i]=value.ravel()[where]
        eig=-np.linalg.eigvalsh(hessian)*sigma**2
        # Reject tubes whose third eigenvalue is close to zero. Keep a separate unfiltered contrast rank.
        sphere=np.maximum(eig[:,2],0)*np.sqrt(np.maximum(eig[:,2],0)/np.maximum(eig[:,0],1e-5))
        coords=np.stack(np.unravel_index(where,slab.shape),1)+start
        rows.extend((loc.tolist(),float(a),float(b)) for loc,a,b in zip(coords,amplitude.ravel()[where],sphere,strict=True))
    assert rows
    return rows


def retain(rows,rank):
    chosen=[]
    for row in sorted(rows,key=lambda r:r[rank],reverse=True):
        coord=np.asarray(row[0])
        if any(np.linalg.norm(prior-coord)<6 for prior in chosen):
            continue
        chosen.append(coord)
        if len(chosen)==max(MAXIMA):
            break
    return chosen


def run(shard,shards):
    OUT.mkdir(exist_ok=True,parents=True)
    for t in read(CASE)["dev"][shard::shards]:
        code=t["code"]
        done=OUT/(code+".json")
        if done.exists():
            continue
        physical=read(DEST/"receipts"/(code+".json"))
        if physical["status"]!="eligible":
            write(done,dict(status=physical["status"]))
            continue
        disk=DEST/"arrays"/(code+".npz")
        assert sha(disk)==physical["sha"]
        with np.load(disk) as f:
            ct=f["ct"];target=f["mask"]
        lungs,body,_=lung_field(ct)
        domain=ndi.binary_dilation(lungs[t["side"]],iterations=5)&body
        rows=proposals(ct,domain)
        distance=ndi.distance_transform_edt(~target,sampling=1.5)
        result={}
        for rank,name in ((1,"contrast"),(2,"sphericity")):
            selected=retain(rows,rank)
            nearest=[float(distance[tuple(point)]) for point in selected]
            result[name]=dict(distances_mm=nearest,near_reference={n:min(nearest[:n])<=6 for n in MAXIMA})
        write(done,dict(code=code,sha=physical["sha"],metrics=result,status="eligible",program=sha(Path(__file__))))
        print(code,{k:v["near_reference"] for k,v in result.items()},flush=True)


if __name__=="__main__":
    os.umask(0o077)
    a=ArgumentParser()
    a.add_argument("--shard",type=int,default=0)
    a.add_argument("--shards",type=int,default=1)
    x=a.parse_args()
    run(x.shard,x.shards)
