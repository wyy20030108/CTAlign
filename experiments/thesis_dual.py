"""Prepare a paired target subset for true text exchange on one untouched CT."""
from argparse import ArgumentParser
from pathlib import Path
import os,tempfile,time
from collections import Counter
import nibabel as nib
import numpy as np
from scipy import ndimage as nd

from thesis_expand_screen import ROOT,SOURCES,REX,select_span,TYPE,PATTERN
from thesis_cohort import LABELED,site
from thesis_transfer import download
from atelectasis_screen import ROOT as OLD,FIBROSIS,QUESTIONABLE
from replay_controls import read_reports
from frozen_priors import read,sha,write
from diag_localize import seal
from fetch_manifest import FILES
from prepare_unlabeled import release_path
from swap_data import reduce_mask,side
from atelectasis_b0 import GRID

OUT=ROOT/"dual_reference"
LOCK=OUT/"candidates.private.json"

def screen():
    OUT.mkdir(exist_ok=True)
    d=read(LABELED)["test"]
    raw={r["code"]:r for path in (SOURCES,OLD/"screen.private.json") for r in read(path)}
    names={raw[r["code"]]["name"] for r in d}
    published={r["name"]:r for r in read(REX)["train"] if r["name"] in names}
    reports=read_reports(names);rows=[]
    for row in d:
        f=raw[row["code"]];name=f["name"]
        release=published[name]
        for k,q in sorted(release["findings"].items(),key=lambda x:int(x[0])):
            if int(k)==f["channel"] or release["categories"].get(k)!="2a" or release["entity_counts"].get(k)!=1:continue
            if not(TYPE.search(q) and PATTERN.search(q) and side(q)) or QUESTIONABLE.search(q) or FIBROSIS.search(q):continue
            m=select_span(q,reports[name])
            if not m or site(dict(query=m[1]))==row["site"]:continue
            rows.append(dict(code=row["code"],released_volume=name,first=f["channel"],second=int(k),
                              primary=row["query"],alternative=m[1],context=m[2],side=side(q),
                              pixels=release["pixels"][k],path=row["path"],path_sha=row["sha"]))
            break
    seal(LOCK,rows)
    print("Independent potential dual-finding cases",len(rows),flush=True)

def fetch(shard,shards):
    for row in read(LOCK)[shard::shards]:
        f=OUT/(row["code"]+".json")
        if f.exists():continue
        for attempt in range(3):
            try:
                with tempfile.TemporaryDirectory(prefix="dual-",dir=OUT) as temp:
                    r=Path(row["path"]);assert sha(r)==row["path_sha"]
                    with np.load(r) as z:affine=z["affine"];primary=z["mask"].astype(bool)
                    name=row["released_volume"]
                    repo,revision,_=FILES["ct_rate"]
                    img,sct=download(repo,revision,release_path(name),temp)
                    repo,revision,_=FILES["rex"]
                    seg,sref=download(repo,revision,"segmentations/"+name,temp)
                    native=nib.load(img)
                    reference=nib.load(seg)
                    assert reference.shape[1:]==native.shape and nib.aff2axcodes(native.affine)==("L","P","S")
                    check=reduce_mask(np.asarray(reference.dataobj[row["first"]])>0,native.affine,(primary.shape,affine))
                    assert np.array_equal(check,primary)
                    mask=np.asarray(reference.dataobj[row["second"]])>0
                    assert int(mask.sum())==row["pixels"]
                    alt=reduce_mask(mask,native.affine,(primary.shape,affine)).astype(bool)
                    nn=int(alt.sum());volume=nn*GRID**3/1000
                    retention=nn*GRID**3/(mask.sum()*abs(np.linalg.det(native.affine[:3,:3])))
                    overlap=int(np.count_nonzero(primary&alt))
                    result=dict(code=row["code"],ct_sha=sct,published_sha=sref,volume_ml=volume,retention=retention,
                                overlap=overlap,status="excluded")
                    components=nd.label(alt,structure=np.ones((3,)*3))[1]
                    opposite=(float(np.argwhere(alt)[:,0].mean())>(primary.shape[0]-1)/2) if nn else None
                    if nn>=8 and .60<=retention<=1.45 and overlap<=.05*min(int(primary.sum()),nn) and components==1 and opposite==(row["side"]=="right"):
                        out=f.with_suffix(".npz")
                        np.savez_compressed(out,mask=alt)
                        result.update(status="eligible",mask_sha=sha(out))
                    write(f,result)
                    print(row["code"],result["status"],volume,retention,overlap,flush=True)
                    break
            except Exception as e:
                print(row["code"],attempt,type(e).__name__,str(e).split("http",1)[0][:60],flush=True)
                time.sleep(2)
                if attempt==2:write(f.with_suffix(".pending.json"),dict(code=row["code"],error=type(e).__name__))

if __name__=="__main__":
    os.umask(0o077)
    p=ArgumentParser();p.add_argument("stage",choices=("screen","fetch"));p.add_argument("--shard",type=int,default=0)
    p.add_argument("--shards",type=int,default=2);a=p.parse_args()
    if a.stage=="screen":screen()
    else:fetch(a.shard,a.shards)

