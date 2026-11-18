"""Recount paired full-volume outputs and archive the verified development comparison."""
from collections import defaultdict
from pathlib import Path
import json
import os
import zipfile

import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
import numpy as np

from atelectasis_sliding import OUT, ROOT, RUNS, BGRIDS, GRID, load_row, score
from frozen_priors import read, sha, write

MODES=("B0","B1","B1_coarse")
FIELDS=("dice","recall","surface_dice_3mm","fp_ml")
def avg(values):return float(np.mean(values))
def median(values):return float(np.median(values))

def summarized(cases,threshold):
    by_code=defaultdict(list)
    for item in cases:by_code[item["code"]].append(item)
    counts={};patients={}
    for model in MODES:
        entries=[row["metrics"][model][threshold] for row in cases]
        patients[model]={k:{m:avg([r["metrics"][model][threshold][m] for r in v])
                            for m in FIELDS if m in entries[0]} for k,v in sorted(by_code.items())}
        counts[model]=dict(n_patients=len(by_code),n_evaluations=len(cases),
            missed=sum(p["miss"] for p in entries), empty=sum(p["empty"] for p in entries),
            hit10=sum(p["recall"]>=.1 for p in entries),
            patients_hit10_majority=sum(sum(r["metrics"][model][threshold]["recall"]>=.1 for r in group)>len(group)/2 for group in by_code.values()),
            patient_mean_dice=avg([x["dice"] for x in patients[model].values()]),
            patient_mean_recall=avg([x["recall"] for x in patients[model].values()]),
            fp_mean_ml=avg([x["fp_ml"] for x in patients[model].values()]),
            fp_median_ml=median([p["fp_ml"] for p in entries]),
            pred_median_ml=median([p["pred"]*GRID**3/1000 for p in entries]))
        if threshold=="0.5":
            counts[model]["patient_mean_surface_dice_3mm"]=avg([x["surface_dice_3mm"] for x in patients[model].values()])
    return counts,patients

def paired_ci(patients,field):
    dr=[patients["B1"][p][field]-patients["B0"][p][field] for p in patients["B0"]]
    indices=np.random.default_rng(89190).integers(len(dr),size=(10000,len(dr)))
    bounds=np.quantile(np.asarray(dr)[indices].mean(axis=1),[.025,.975]).tolist()
    return {"mean":avg(dr),"ci95":bounds,"patients_improved":sum(x>0 for x in dr)}

def plot_development(dev,budget,seed):
    # Display the same maximum reference-area axial slice for all model overlays.
    # Its coordinates are for visualization after scoring.
    fig,axs=plt.subplots(6,3,figsize=(10,16),squeeze=False)
    for row_index,row in enumerate(dev):
        ct,truth=load_row(row); z=int(np.argmax(truth.sum(axis=(0,1))))
        with np.load(OUT/budget/str(seed)/(row["code"]+".npz")) as f:
            pred=[np.unpackbits(f[k],count=truth.size).reshape(truth.shape)[:,:,z].astype(bool) for k in ("B0","B1")]
        for j,ax in enumerate(axs[row_index]):
            ax.imshow(ct[:,:,z].T,cmap="gray",origin="lower",vmin=-1000,vmax=180)
            if j and pred[j-1].any():ax.contour(pred[j-1].T,levels=[.5],colors=["#52c9ff"],origin="lower",linewidths=.8)
            ax.contour(truth[:,:,z].T,levels=[.5],colors=["#ffbb4c"],origin="lower",linewidths=1.1)
            ax.set_title(row["code"]+" "+("reference only","low-res B0","sliding B1")[j],fontsize=9)
            ax.set_axis_off()
    label="2 masks" if budget=="b10" else "20 masks"
    fig.suptitle(f"Paired development overlays | {label}, seed {seed} | Yellow reference, blue prediction",fontsize=12)
    fig.tight_layout(rect=(0,0,1,.98))
    path=OUT/(budget+"_paired_overlays.png")
    fig.savefig(path,dpi=125)
    plt.close(fig)
    return path

def release():
    os.umask(0o077)
    dev=read(ROOT/"pilot_private.json")["development"]
    dev_ids={r["code"] for r in dev}
    cases=[]; files=[OUT/"protocol.json"]
    for budget,seed in RUNS:
        run=OUT/budget/str(seed)
        result=read(run/"results.json")
        old=read(ROOT/budget/str(seed)/"results.json")
        assert set(result["train_codes"])==set(old["train_codes"])
        assert result["reference_sha"]==old["model_sha"]
        assert result["protocol"]==sha(OUT/"protocol.json")
        assert len(read(run/"training.json")["loss"])==384
        assert result["model_sha"]==sha(run/"B1.pt")
        assert {r["code"] for r in result["rows"]}==dev_ids
        for row in dev:
            code=row["code"];file=run/(code+".npz")
            value=next(x for x in result["rows"] if x["code"]==code)
            assert value["prediction_sha"]==sha(file)
            ct,truth=load_row(row);del ct
            with np.load(file) as z:
                assert tuple(z["shape"])==truth.shape and value["covering_fraction"]==1 and value["tiles"]>0
                for mode in MODES:
                    pred=np.unpackbits(z[mode],count=truth.size).reshape(truth.shape).astype(bool)
                    raw=score(pred,truth,surface=False)
                    match=value["metrics"][mode]["0.5"]
                    assert all(match[k]==v for k,v in raw.items())
            cases.append(dict(budget=budget,seed=seed,**value))
            files.append(file)
        files.extend(run/f for f in ("results.json","B1.pt","training.json"))
    assert len(cases)==24
    groups={}
    for budget in ("b10","all_labeled"):
        pairs=[r for r in cases if r["budget"]==budget]
        assert len(pairs)==(18 if budget=="b10" else 6)
        threshold={str(t):summarized(pairs,str(t))[0] for t in BGRIDS}
        stats,patients=summarized(pairs,"0.5")
        truth_ml=lambda r:r["metrics"]["B0"]["0.5"]["true"]*GRID**3/1000
        strata={label:summarized(sub,"0.5")[0] for label,sub in (
                ("below_1ml",[r for r in pairs if truth_ml(r)<1]),
                ("at_least_1ml",[r for r in pairs if truth_ml(r)>=1])) if sub}
        groups[budget]=dict(fixed=stats,patient=patients,dice_delta=paired_ci(patients,"dice"),fp_delta_ml=paired_ci(patients,"fp_ml"),
                            surface_dice_delta=paired_ci(patients,"surface_dice_3mm"),by_threshold=threshold,by_size=strata,
                            reference_volume_median_ml=median([truth_ml(r) for r in pairs]))
    inputs=[r for r in cases if r["budget"]=="b10" and r["seed"]==891]
    detail=dict(empty_half_occupancy=sum(r["occupancy_half_cells"]==0 for r in inputs),
                occupancy_cells={r["code"]:r["occupancy_cells"] for r in inputs},
                majority_occupancy_cells={r["code"]:r["occupancy_half_cells"] for r in inputs})
    training_check=[]
    for seed in (891,1901,2707):
        p=OUT/"b10"/str(seed)/"training_field_check.json"
        chk=read(p)
        assert chk["checkpoint"]==sha(p.parent/"B1.pt") and len(chk["results"])==2
        training_check.extend(dict(seed=seed,**r) for r in chk["results"])
        files.append(p)
    write(OUT/"summary.json",dict(groups=groups,cases=cases,coarse_support=detail,training_check=training_check,
                                  active_holdout_scored=False))
    files.extend([OUT/"summary.json",plot_development(dev,"b10",2707),plot_development(dev,"all_labeled",891)])
    brief={k:{"B0":v["fixed"]["B0"],"B1":v["fixed"]["B1"],"dice_delta":v["dice_delta"]} for k,v in groups.items()}
    print(json.dumps(brief,indent=2),flush=True)
    source=[Path(n) for n in ("atelectasis_sliding.py","atelectasis_sliding_release.py","atelectasis_sliding_traincheck.py","atelectasis_b0.py")]
    archive=OUT/"sliding_evidence.zip"
    tmp=archive.with_suffix(".part")
    with zipfile.ZipFile(tmp,"w",zipfile.ZIP_DEFLATED) as z:
        combined={str(p.relative_to(OUT)):p for p in files}
        combined.update({"code/"+p.name:p for p in source})
        for path,f in combined.items():z.write(f,path)
        z.writestr("checksums.json",json.dumps({path:sha(f) for path,f in combined.items()}))
    tmp.replace(archive)
    print("archive",sha(archive))
if __name__=="__main__":release()

