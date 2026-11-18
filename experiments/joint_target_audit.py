"""Count pseudo foreground before and after PRCV transport on development references."""
from argparse import ArgumentParser
from pathlib import Path
import os

import numpy as np

import thesis_model as image
from frozen_priors import read, sha, write
from joint_factorial import teacher_target
from joint_roi import AUDIT, ROOT, warm_models
from thesis_cohort import MANIFEST
from thesis_lobe_model import LobeStore
from thesis_ssl import SEEDS


DEST = ROOT / "target_audit"


def annotated_prior(rows,store,use_text):
    values=[]
    for i,row in enumerate(rows):
        x=store.get(row,True)
        q=row["query"] if use_text else ""
        patch=image.positive_region(x,np.random.default_rng(5316+i))
        v=x.domain(q)[patch]
        values.append(float(np.count_nonzero(x.truth[patch]&v)/max(1,v.sum())))
    return float(np.clip(np.median(values),.0001,.08))


def run(kind):
    DEST.mkdir(exist_ok=True,parents=True)
    cohort=read(MANIFEST)
    use_text=kind=="B"
    labels=[r for r in cohort["train"] if r["code"] in cohort["labels"]["20"]]
    store=LobeStore(limit=32)
    prior=annotated_prior(labels,store,use_text)
    cap=max(3.,3*float(np.median([r["volume_ml"] for r in labels])))
    quota=int(min(4.,.65*cap)*1000/image.GRID**3)
    models=warm_models(kind)
    result=[]
    for i,row in enumerate(cohort["dev"]):
        x=store.get(row,True)
        q=row["query"] if use_text else ""
        receipt=AUDIT/(kind+"_"+row["code"]+".json")
        pos=read(receipt)["choices"]["agreement"]["origin"]
        actual=x.truth[image.slices(pos)]
        for seed,teacher in zip(SEEDS,models,strict=True):
            modes={}
            for m in ("none","direct","conditioned"):
                rng=np.random.default_rng(seed*7321+i+1)
                candidate, diag=teacher_target(teacher,x,q,pos,rng,prior,m!="none",m,quota)
                p=None if candidate is None else candidate[1].cpu().numpy()
                modes[m]=dict(accepted=diag[0],voxels=0 if p is None else int(p.sum()),
                              hits=0 if p is None else int((p&actual).sum()),lambda_value=diag[2],
                              raw=diag[3],calibrated=diag[4])
            result.append(dict(code=row["code"],seed=seed,ref=int(x.truth.sum()),patch_ref=int(actual.sum()),modes=modes))
    pooled={}
    for key in ("none","direct","conditioned"):
        counts={s:sum(r["modes"][key][s] for r in result) for s in ("accepted","voxels","hits")}
        pooled[key]=dict(**counts,precision=counts["hits"]/max(1,counts["voxels"]),
                         recall=counts["hits"]/sum(r["ref"] for r in result),
                         qualified=sum(r["modes"][key]["hits"]>0 for r in result))
    write(DEST/(kind+".json"),dict(prior=prior,cases=result,pooled=pooled,code=sha(Path(__file__))))
    print(kind,prior,pooled,flush=True)


if __name__=="__main__":
    os.umask(0o077)
    p=ArgumentParser()
    p.add_argument("modality",choices=("A","B"))
    run(p.parse_args().modality)
