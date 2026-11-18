"""Freeze the complete comparison on development then evaluate untouched patients."""
from argparse import ArgumentParser
from pathlib import Path
import os,time
import numpy as np
from scipy import ndimage as nd
from scipy.spatial import cKDTree
import torch

from diag_localize import seal
from frozen_priors import sha,read,write
from thesis_cohort import ROOT,MANIFEST,LOCK,PRIMARY,site
from thesis_lobe_model import LobeStore
from thesis_model import FindingNet,DEVICE,score,volume_logits,segment,GRID,THRESHOLDS
from thesis_ssl import OUT,SEEDS,ARMS
from finding_train import configure

EVAL=ROOT/"evaluation_lock.json"
RESULTS=ROOT/"independent_evaluation"
MEASURES=("dice","recall","fp_ml","surface_dice_3mm","hd95","empty","miss")

def train_file(budget,seed,arm):
    start=OUT/budget/str(seed)/arm
    results=read(start/"development.json");fit=read(start/"training.json")
    assert results["model_sha"]==fit["model_sha"]==sha(start/"model.pt")
    assert results["arm"]==fit["arm"]==arm and len(fit["logs"])==fit["supervised_updates"]==384
    assert fit["protocol"]==sha(OUT/"protocol.json")
    return start,results,fit

def freeze():
    cohort=read(MANIFEST)
    test_codes={r["code"] for r in cohort["test"]}
    dev_codes={r["code"] for r in cohort["dev"]}
    assert not test_codes&dev_codes and read(LOCK)["manifest_sha"]==sha(MANIFEST)
    detailed={}
    for budget,arms in ((PRIMARY,ARMS),("5","BF"),("10","BF")):
        detailed[budget]={}
        for arm in arms:
            thresholds={str(t):[] for t in THRESHOLDS}
            records=[]
            for seed in SEEDS:
                start,r,fit=train_file(budget,seed,arm)
                assert set(fit["train_codes"])==set(cohort["labels"][budget])
                assert {x["code"] for x in r["rows"]}==dev_codes
                for t,vals in thresholds.items():
                    vals.extend(x["values"][t]["dice"] for x in r["rows"])
                accepted=sum(p["accepted"] for p in fit["logs"])
                records.append(dict(seed=seed,model=fit["model_sha"],development=sha(start/"development.json"),
                                    training=sha(start/"training.json"),accepted=accepted,
                                    pos=sum(u["pos"] for u in fit["logs"])))
            pooled={t:float(np.mean(vals)) for t,vals in thresholds.items()}
            best=max(pooled,key=lambda t:(pooled[t],-abs(float(t)-.5),float(t)))
            detailed[budget][arm]=dict(chosen=best,development_dice=pooled[best],all_thresholds=pooled,checkpoints=records)
    primary_base=max(("B","D"), key=lambda k:(detailed[PRIMARY][k]["development_dice"],k=="B"))
    seal(EVAL,dict(manifest=sha(MANIFEST),training_protocol=sha(OUT/"protocol.json"),evaluator=sha(Path(__file__)),
                   primary_budget=PRIMARY,primary="F",comparator=primary_base,arms=detailed,
                   endpoint="Patient paired macro whole-volume Dice; report recall and extra-reference volume. Wrong anatomical query is a prompt-dependence check on the original reference, not a negative/no-lesion assertion.",
                   comparators_provenance="Identical lesion label budget; B and D are the eligible direct CT+text baselines. ReX-trained generic checkpoints would have seen the reserved training-release patients."))
    print("Locked",PRIMARY,primary_base,["%s:%s"%(k,detailed[PRIMARY][k]["chosen"]) for k in ARMS],flush=True)

def compare_visuals(mask,truth):
    d=score(mask,truth,surface=True)
    hd=None
    if mask.any():
        conn=nd.generate_binary_structure(3,1)
        a=np.argwhere(mask & ~nd.binary_erosion(mask,structure=conn))*GRID
        b=np.argwhere(truth & ~nd.binary_erosion(truth,structure=conn))*GRID
        hd=max(float(np.quantile(cKDTree(a).query(b)[0],.95)),float(np.quantile(cKDTree(b).query(a)[0],.95)))
    d["hd95"]=hd
    return d

def donors(cohort):
    u=cohort["unlabelled"]
    out={}
    for r in cohort["test"]:
        s=r["site"]
        order=sorted([x for x in u if x["site"]!=s],
                     key=lambda x:(int(x["site"].split("_")[0]==s.split("_")[0]),
                                   abs(len(x["query"].split())-len(r["query"].split())),x["code"]))
        assert order
        out[r["code"]]=order[0]
    return out

SITE_LANGUAGE={"right_middle":"right middle lobe","right_lower":"right lower lobe","right_upper":"right upper lobe",
               "left_lingular":"lingula of the left upper lobe","left_lower":"left lower lobe","left_upper":"left upper lobe"}
def evaluate(budget,seed,arm):
    fixed=read(EVAL)
    assert fixed["manifest"]==sha(MANIFEST) and fixed["evaluator"]==sha(Path(__file__))
    assert fixed["training_protocol"]==sha(OUT/"protocol.json")
    run,check,fit=train_file(budget,seed,arm);cfg=fixed["arms"][budget][arm]
    matched=next(q for q in cfg["checkpoints"] if q["seed"]==seed)
    assert matched["model"]==sha(run/"model.pt")
    assert matched["development"]==sha(run/"development.json") and matched["training"]==sha(run/"training.json")
    cohort=read(MANIFEST);swap=donors(cohort)
    folder=RESULTS/budget/str(seed)/arm;folder.mkdir(exist_ok=True,parents=True)
    if (folder/"finished.json").exists():
        assert read(folder/"finished.json")["lock"]==sha(EVAL);return
    configure(seed)
    net=FindingNet().to(DEVICE).eval()
    net.load_state_dict(torch.load(run/"model.pt",map_location=DEVICE,weights_only=True))
    store=LobeStore(limit=2)
    offered=[]
    for row in cohort["test"]:
        code=row["code"];result=folder/(code+".json")
        save=folder/(code+".npz")
        if result.exists():
            cache=read(result);assert sha(save)==cache["prediction_sha"] and cache["locked"]==sha(EVAL)
            offered.append(cache);continue
        case=store.get(row,True)
        text=(arm not in "AC")
        queries={"correct":row["query"] if text else ""}
        if text and budget==PRIMARY and arm in {"B","D","F"}:
            queries.update(wrong_location=swap[code]["query"],empty="",
                           site_template="linear atelectasis in the "+SITE_LANGUAGE.get(row["site"],row["site"]))
        outcome={};masks={}
        for name,q in queries.items():
            logits,windows=volume_logits(net,case,q)
            mask=segment(logits,float(cfg["chosen"]),check["cap_ml"])
            outcome[name]=compare_visuals(mask,case.truth)
            outcome[name]["tiles"]=windows
            masks[name]=np.packbits(mask.ravel())
        np.savez_compressed(save,shape=case.ct.shape,**masks)
        rec=dict(code=code,locked=sha(EVAL),prediction_sha=sha(save),n_voxels=int(case.truth.sum()),
                 site=row["site"],source_style=row["source_style"],outcomes=outcome,
                 wrong_location_code=swap[code]["code"] if "wrong_location" in queries else None)
        write(result,rec);offered.append(rec)
        print("independent",budget,arm,seed,code,round(outcome["correct"]["dice"],5),flush=True)
    write(folder/"finished.json",dict(lock=sha(EVAL),budget=budget,seed=seed,arm=arm,rows=offered,threshold=cfg["chosen"]))

if __name__=="__main__":
    os.umask(0o077)
    p=ArgumentParser();p.add_argument("stage",choices=("lock","run"))
    p.add_argument("--budget",default=PRIMARY);p.add_argument("--seed",type=int,default=891);p.add_argument("--arms",default="AB")
    args=p.parse_args()
    if args.stage=="lock":freeze()
    else:
        for arm in args.arms:evaluate(args.budget,args.seed,arm)

