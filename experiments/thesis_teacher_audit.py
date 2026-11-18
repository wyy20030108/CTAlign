"""Audit the initial teacher on development scans as if their masks were unseen."""
from pathlib import Path
import os,json
import numpy as np
from scipy.ndimage import binary_dilation
import torch
from thesis_cohort import ROOT,MANIFEST,PRIMARY
from thesis_model import FindingNet,DEVICE,GRID
from thesis_lobe_model import LobeStore,CODES
from thesis_ssl import OUT,SEEDS,pseudo_target
from frozen_priors import read,write,sha
from finding_train import configure

def run():
    os.umask(0o077)
    cohort=read(MANIFEST);store=LobeStore(limit=20)
    visible={x for x in cohort["labels"][PRIMARY]}
    cap=max(3.,3*float(np.median([r["volume_ml"] for r in cohort["train"] if r["code"] in visible])))
    quota=int(min(4.,.65*cap)*1000/GRID**3)
    result=[]
    for seed in SEEDS:
        path=OUT/PRIMARY/str(seed)/"initial/B.pt"
        assert sha(path)==read(path.with_suffix(".json"))["sha"]
        configure(seed);model=FindingNet().to(DEVICE).eval()
        model.load_state_dict(torch.load(path,map_location=DEVICE,weights_only=True))
        for i,row in enumerate(cohort["dev"]):
            d=store.get(row,True)
            coords={}
            original=d.features
            def watched(region,query):
                tensor=original(region,query)
                coords[id(tensor)]=(tensor,region)
                return tensor
            d.features=watched
            try:
                target,diag=pseudo_target(model,d,row["query"],np.random.default_rng(seed*7321+i+1),quota,.01,False)
            finally:
                d.features=original
            data=dict(code=row["code"],seed=seed,teacher=sha(path),truth=int(d.truth.sum()),**diag)
            if target is None:
                data.update(pos_tp=0,pos_fp=0,lobe_tp=0,lobe_fp=0,reference_seen=0)
            else:
                x,pos,bg,probability=target
                region=coords[id(x)][1];m=d.truth[region]
                pred=pos.cpu().numpy()[0]
                q=CODES[row["site"]]
                fair=d.allowed[q][region] if q in d.allowed else binary_dilation(d.lobes==q,iterations=4)[region]
                restricted=pred&fair
                data.update(pos_tp=int(np.count_nonzero(m&pred)),pos_fp=int(np.count_nonzero(~m&pred)),
                            lobe_tp=int(np.count_nonzero(m&restricted)),lobe_fp=int(np.count_nonzero(~m&restricted)),
                            reference_seen=int(m.sum()))
            result.append(data)
        print(seed,sum(x["pos_tp"]>0 for x in result if x["seed"]==seed),flush=True)
    aggregate={k:sum(r[k] for r in result) for k in ("accepted","pos_tp","pos_fp","lobe_tp","lobe_fp")}
    aggregate["num_evaluated"]=len(result)
    write(ROOT/"development_teacher_audit.json",dict(rows=result,counts=aggregate,
          scope="Development masks only opened for diagnostic scoring after teacher selected a patch and stable foreground. No change to the predetermined primary comparison."))
    print(aggregate,flush=True)
if __name__=="__main__":run()

