"""Cache external LTRC-derived lung lobes without consulting lesion masks."""
import argparse
import os
from pathlib import Path
import sys
import numpy as np
from scipy import ndimage as nd
from thesis_expand_screen import ROOT
from thesis_cohort import MANIFEST,LABELED,site
from thesis_report_only import U
from thesis_model import cohort_pilot
from frozen_priors import read,sha,write

sys.path.insert(0,"/tmp/thesis_lobes_deps")
CACHE=ROOT/"lobe_anatomy"
WEIGHT=Path("/tmp/thesis_lobes_weights/unet_ltrclobes-3a07043d.pth")
CODES={"left_upper":1,"left_lingular":1,"left_lower":2,"right_upper":3,"right_middle":4,"right_lower":5}

def prepare(rows):
    from lungmask import LMInferer
    import torch
    CACHE.mkdir(exist_ok=True)
    checksum=sha(WEIGHT);assert checksum.startswith("3a07043d")
    torch.set_num_threads(2)
    inferer=LMInferer(modelpath=str(WEIGHT),batch_size=12,tqdm_disable=True)
    for row in rows:
        code=row["code"];result=CACHE/(code+".npz");rec=CACHE/(code+".json")
        if rec.exists():
            assert sha(result)==read(rec)["prediction_sha"];continue
        src=Path(row["path"]);assert sha(src)==row["sha"]
        with np.load(src) as a:
            ct=a["ct"]
        # LTRC expects axial arrays in z, anterior-to-posterior, right-to-left order.
        inp=ct[::-1,::-1,:].transpose(2,1,0).copy()
        labeled=inferer.apply(inp).transpose(2,1,0)[::-1,::-1,:].copy().astype(np.uint8)
        assert labeled.shape==ct.shape and set(np.unique(labeled))<=set(range(6))
        midpoint=(ct.shape[0]-1)/2
        left=np.argwhere((labeled==1)|(labeled==2))[:,0]
        right=np.argwhere(labeled>=3)[:,0]
        assert left.size and right.size and left.mean()<midpoint<right.mean()
        temp=result.with_suffix(".part.npz")
        np.savez_compressed(temp,lobes=labeled);temp.replace(result)
        write(rec,dict(code=code,weights_sha=checksum,source_sha=row["sha"],prediction_sha=sha(result),
                       counts=np.bincount(labeled.ravel(),minlength=6).tolist(),model="lungmask LTRCLobes",affine_order="RAS"))
        print(code,round(float(left.mean()),1),round(float(right.mean()),1),flush=True)

def audit(rows):
    results=[]
    for row in rows:
        code=row["code"]
        r=read(CACHE/(code+".json"))
        assert r["source_sha"]==row["sha"]
        s=site(row);label=CODES[s]
        with np.load(row["path"]) as a:
            y=a["mask"].astype(bool)
        with np.load(CACHE/(code+".npz")) as a:
            m=a["lobes"]==label
        assert y.shape==m.shape
        margin=nd.binary_dilation(m,iterations=4)
        results.append(dict(code=code,site=s,exact_coverage=float(m[y].mean()),margin6mm_coverage=float(margin[y].mean())))
    write(CACHE/"development_coverage.json",results)
    print(results,flush=True)

if __name__=="__main__":
    os.umask(0o077)
    p=argparse.ArgumentParser();p.add_argument("stage",choices=("pilot","expanded","expanded_labels","report_available","audit_pilot","test"));p.add_argument("--shard",type=int,default=0);p.add_argument("--shards",type=int,default=1)
    a=p.parse_args()
    if a.stage=="audit_pilot":audit(cohort_pilot()["development"])
    else:
        if a.stage=="pilot":
            d=cohort_pilot();rows=d["development"]+d["train"]
        elif a.stage=="expanded":
            d=read(MANIFEST);rows=d["train"]+d["dev"]+d["unlabelled"]
        elif a.stage=="expanded_labels":
            d=read(LABELED);rows=d["train"]+d["dev"]
        elif a.stage=="report_available":
            rows=[dict(code=f.stem,sha=read(f)["processed_sha"],path=str(U/"arrays"/(f.stem+".npz"))) for f in sorted((U/"receipts").glob("U???.json"))]
        else:
            fixed=read(ROOT/"evaluation_lock.json")
            assert fixed["manifest"]==sha(MANIFEST)
            rows=read(MANIFEST)["test"]
        prepare(rows[a.shard::a.shards])

