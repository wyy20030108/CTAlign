"""Measure pseudo-label precision/recall for the stronger detector-localized teacher."""
from argparse import ArgumentParser
from pathlib import Path
import os

import numpy as np
from scipy import ndimage as ndi

import joint_external_refine as local
from joint_external_residual import base_contour
import joint_factorial as trainer
import joint_nodule_gate as gate
import joint_nodule_growth_refine as growth
import joint_roi as proposal
from frozen_priors import read,sha,write
from joint_nodule_run import COHORT,DEST
from joint_nodule_size import get_calibrator
from joint_target_audit import annotated_prior
from thesis_lobe_model import LobeStore
from thesis_ssl import SEEDS


OUT=DEST/"local_pseudo_diagnostic"


def run(kind):
    local.install()
    co=read(COHORT)
    OUT.mkdir(exist_ok=True,parents=True)
    store=LobeStore(limit=12)
    proposal.PREVIOUS=local.WARM
    models=proposal.warm_models(kind)
    labels=[r for r in co["train"] if r["code"] in co["labels"]["20"]]
    prior=annotated_prior(labels,store,kind=="B")
    gate.report_volume=get_calibrator(COHORT)
    records=[]
    transport=trainer.transport
    prior_shape=growth.prior_shape
    for i,row in enumerate(co["dev"]):
        query=row["query"] if kind=="B" else ""
        case=store.get(row,True)
        candidate=base_contour(case,query)
        if candidate is None:
            continue
        _,pos,box,mask=candidate
        start=local.origin(case,pos)
        window=local.slices(start)
        whole=np.zeros(case.ct.shape,bool)
        whole[box]=mask
        centre=whole[window]
        if not centre.any():
            continue
        d=ndi.distance_transform_edt(~centre)
        focus=d<=2
        hint=(2.*np.exp(-d**2/2.25)).astype(np.float32)
        density=float((centre&case.domain(query)[window]).sum()/max(1,(focus&case.domain(query)[window]).sum()))
        density=float(np.clip(density,.0001,.35))
        growth.prior_shape=lambda mode,code:(focus,hint)
        for seed,teacher in zip(SEEDS,models,strict=True):
            measures={}
            for option in ("raw","conditioned","contour_anchored"):
                trainer.transport=transport
                if option=="contour_anchored":
                    trainer.transport=lambda p,q,mode:transport(p,float(np.clip(np.sqrt(q*density),.0001,.35)),mode)
                rng=np.random.default_rng(seed*7321+i+1)
                fake,diag=growth.target(teacher,case,query,start,rng,prior,option!="raw","conditioned",640)
                count=hit=0
                if fake is not None:
                    y=fake[1].cpu().numpy()
                    count=int(y.sum())
                    hit=int((y&case.truth[window]).sum())
                measures[option]=dict(accepted=diag[0],voxels=count,hit=hit)
            records.append(dict(code=case.code,seed=seed,truth=int(case.truth.sum()),results=measures))
    trainer.transport=transport
    growth.prior_shape=prior_shape
    study={}
    for option in ("raw","conditioned","contour_anchored"):
        counts={m:sum(r["results"][option][m] for r in records) for m in ("accepted","voxels","hit")}
        study[option]=dict(**counts,precision=counts["hit"]/max(1,counts["voxels"]),recall=counts["hit"]/sum(r["truth"] for r in records))
    write(OUT/(kind+".json"),dict(results=study,prior=prior,details=records,program=sha(Path(__file__))))
    print(kind,study,flush=True)


if __name__=="__main__":
    os.umask(0o077)
    p=ArgumentParser()
    p.add_argument("kind",choices=("A","B"))
    run(p.parse_args().kind)
