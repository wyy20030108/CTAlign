"""Compute prespecified matched contrasts without letting test data choose the model."""
from argparse import ArgumentParser
from collections import defaultdict
from pathlib import Path

import numpy as np

from frozen_priors import read, sha, write
from joint_roi import ROOT
from thesis_model import THRESHOLDS
from thesis_ssl import SEEDS


ARM = tuple(k+a for k in "AB" for a in "SNUP")
CONTRASTS = {
    "report_only_effect": {"BN": 1, "AN": -1},
    "transport_only_effect": {"AU": 1, "AN": -1},
    "add_transport_to_report": {"BU": 1, "BN": -1},
    "add_report_to_transport": {"BU": 1, "AU": -1},
    "joint_over_main_baseline": {"BU": 1, "AN": -1},
    "transport_report_interaction": {"BU": 1, "BN": -1, "AU": -1, "AN": 1},
    "unlabelled_report_over_supervised": {"BN": 1, "BS": -1},
    "joint_transport_over_supervised": {"BU": 1, "BS": -1},
    "prototype_increment": {"BP": 1, "BU": -1},
    "full_framework_without_report": {"AP": 1, "AN": -1},
    "full_framework_with_report": {"BP": 1, "BN": -1},
    "full_report_beyond_framework": {"BP": 1, "AP": -1},
    "joint_full_over_supervised": {"BP": 1, "BS": -1},
    "full_framework_report_interaction": {"BP": 1,"BN": -1,"AP": -1,"AN": 1},
}


def summary(values):
    x = np.asarray(values,float)
    sample = np.random.default_rng(20261119).integers(len(x),size=(20000,len(x)))
    ci = np.quantile(x[sample].mean(1),(.025,.975)).tolist()
    return dict(delta=float(x.mean()),ci95=ci,improvement=int((x>0).sum()),patients=len(x))


def compare(scores):
    out={}
    for key, powers in CONTRASTS.items():
        if not all(arm in scores for arm in powers):
            continue
        common=sorted(next(iter(scores.values())))
        assert all(set(scores[arm])==set(common) for arm in powers)
        changes={metric:summary([sum(number*scores[arm][p][metric] for arm,number in powers.items())
                                for p in common]) for metric in ("dice","recall","fp_ml","miss")}
        out[key]=changes
    return out


def development(root):
    patient={}
    thresholds={}
    pinned={}
    for arm in ARM:
        k,a=arm
        paths=[root/k/str(seed)/a/"development.json" for seed in SEEDS]
        if not all(p.is_file() for p in paths):
            continue
        complete=[read(f) for f in paths]
        for r, path in zip(complete,paths,strict=True):
            assert r["model"]==sha(path.with_name("model.pt")) and r["protocol"]==sha(root/"protocol.json")
        grid=[str(t) for t in THRESHOLDS]
        mean={th:float(np.mean([r["value"][th]["dice"] for answer in complete for r in answer["score"]])) for th in grid}
        value=max(grid,key=lambda q:(mean[q],-abs(float(q)-.50)))
        grouped=defaultdict(list)
        for response in complete:
            for row in response["score"]:
                grouped[row["code"]].append(row["value"][value])
        assert all(len(g)==len(SEEDS) for g in grouped.values())
        patient[arm]={k:{m:float(np.mean([row[m] for row in v])) for m in ("dice","recall","fp_ml","miss")}
                      for k,v in grouped.items()}
        thresholds[arm]=dict(threshold=float(value),dice=mean[value],all=mean)
        pinned[arm]=[dict(seed=s,weight=r["model"],receipt=sha(f),cap_ml=r["cap_ml"])
                     for s,r,f in zip(SEEDS,complete,paths,strict=True)]
    values=dict(thresholds=thresholds,weights=pinned,contrast=compare(patient),patient=patient,
                protocol=sha(root/"protocol.json"),stage="Development results for optimization only")
    write(root/"development_study.json",values)
    print({k:(round(v["dice"]*100,2),v["threshold"]) for k,v in thresholds.items()},flush=True)
    print({k:round(v["dice"]["delta"]*100,2) for k,v in values["contrast"].items()},flush=True)


if __name__=="__main__":
    p=ArgumentParser()
    p.add_argument("path",type=Path,default=ROOT/"factorial_matched",nargs="?")
    development(p.parse_args().path)
