"""Independent native-grid recount and patient-paired descriptive inference."""
import argparse,hashlib,json,os,shutil
from pathlib import Path
from zipfile import ZipFile,ZIP_DEFLATED
import numpy as np
from finding_train import OUT,SRC,SEEDS,ARMS,DESIGN,read,write,sha

def average(values):return float(np.mean(values))
def difference(vec):
    x=np.asarray(vec,float);draw=np.random.default_rng(891).integers(len(x),size=(10000,len(x)))
    return dict(mean_pp=average(x),patient_bootstrap95=np.quantile(x[draw].mean(axis=1),[.025,.975]).tolist(),
                positive_equal_negative=[int((x>0).sum()),int((x==0).sum()),int((x<0).sum())])

def run(variant):
    root=OUT/variant if variant=="volume_term" else OUT
    groups=("fallback_val","swap_dev")
    source={}
    for group in groups:
        for r in read(SRC/(group+".private.json")):
            assert r["code"] not in source and sha(Path(r["path"]))==r["sha256"]
            with np.load(r["path"]) as v:source[r["code"]]=(group,v["masks"].astype(bool),v["affine"])
    assert len(source)==18
    rows=[];checks=[];manifest=[];initial={}
    for arm in (*ARMS,"unpaired"):
        for seed in SEEDS:
            base=root/"unpaired_U"/str(seed) if arm=="unpaired" else root/str(seed)
            name="semi" if arm=="unpaired" else arm
            step=read(base/(name+".json"));sign=sha(base/(name+".pt"))
            assert step["sha256"]==sign and step["steps"]==256 and step["n_U"]==(32 if name=="semi" else 0)
            assert initial.setdefault(seed,step["initial"])==step["initial"]
            if arm=="unpaired":
                intervention=read(base/"intervention.json")
                assert sorted(intervention["unlabelled_CT_permutation"])==list(range(32)) and intervention["fixed_points"]==0
            if variant=="volume_term":
                other=(OUT/"unpaired_U"/str(seed)) if arm=="unpaired" else OUT/str(seed)
                assert read(other/(name+".json"))["initial"]==step["initial"]
            manifest.append(dict(seed=seed,arm=arm,checkpoint=sign,initial=step["initial"],n_U=step["n_U"],updates=step["steps"]))
            folder=base/(name+"_dev")
            assert {p.stem for p in folder.glob("*.json")}==set(source)
            for code,(group,m,aff) in source.items():
                audit=read(folder/(code+".json"));f=folder/(code+".npz")
                assert audit["checkpoint_sha256"]==sign and audit["map_sha256"]==sha(f)
                metrics={};volumes=[]
                with np.load(f) as v:
                    assert np.array_equal(v["affine"],aff) and tuple(v["shape"])==m.shape
                    for mode in set(v.files)-{"affine","shape"}:
                        pred=np.unpackbits(v[mode],bitorder="little",count=m.size).reshape(m.shape)
                        for key,pair in [(mode,pred)]+([] if mode=="empty" else [(mode+"_swapped",pred[::-1])]):
                            dices=[]
                            for j,(p,t) in enumerate(zip(pair,m)):
                                n=int(np.count_nonzero(p));s=int(np.count_nonzero(t));i=int(np.count_nonzero(p&t));d=2*i/(n+s)
                                recorded=audit["scores"][key][j]
                                assert (n,s,i,d)==tuple(recorded[k] for k in ("pred_voxels","true_voxels","intersection","dice"))
                                if key=="paired":volumes.append([s*.008,n*.008])
                                dices.append(d)
                            metrics[key]=100*average(dices)
                assert set(metrics)==set(audit["scores"])
                rows.append(dict(arm=arm,seed=seed,code=code,group=group,values=metrics,volumes=volumes,indexed=audit["location_index"]))
            if variant=="base":
                cal=read(base/(name+"_threshold.json"));sel=cal["selected"]
                assert sel==int(np.argmax(np.mean([r["correct"] for r in cal["rows"]["fallback_train"]],axis=0)))
                for g in groups:
                    for r in cal["rows"][g]:
                        existing=next(z for z in reversed(rows) if z["arm"]==arm and z["seed"]==seed and z["code"]==r["code"])
                        assert abs(100*r["correct"][9]-existing["values"]["paired"])<1e-5
                checks.append(dict(arm=arm,seed=seed,probability=cal["thresholds"][sel],train_dice05=100*average([r["correct"][9] for r in cal["rows"]["fallback_train"]]),
                        train_dice_cal=100*average([r["correct"][sel] for r in cal["rows"]["fallback_train"]]),
                        development_dice_cal=100*average([r["correct"][sel] for g in groups for r in cal["rows"][g]])))
    lookup={(r["arm"],r["seed"],r["code"]):r for r in rows};stats={}
    for g in (*groups,"combined"):
        codes=[c for c,v in source.items() if g=="combined" or v[0]==g]
        def values(a,m):
            return np.asarray([average([lookup[a,s,c]["values"][m] for s in SEEDS]) for c in codes])
        arms={a:{m:average(values(a,m)) for m in ("paired","paired_swapped","empty","template")} for a in (*ARMS,"unpaired")}
        diffs={k:difference(values(a,m)-values(b,n)) for k,a,m,b,n in [
            ("semi_minus_supervised","semi","paired","supervised","paired"),("semi_minus_unpaired_U","semi","paired","unpaired","paired"),
            ("semi_minus_image","semi","paired","image","paired"),("semi_correct_minus_swapped","semi","paired","semi","paired_swapped"),
            ("semi_correct_minus_template","semi","paired","semi","template")]}
        stats[g]=dict(n=len(codes),arms=arms,differences=diffs)
    cset=[c for c in source if "clinical" in lookup["semi",SEEDS[0],c]["values"]]
    clinical=difference([average([lookup["semi",s,c]["values"]["clinical"]-lookup["semi",s,c]["values"]["clinical_swapped"] for s in SEEDS]) for c in cset])
    seed_stats={}
    for s in SEEDS:
        rs=[r for r in rows if r["seed"]==s and r["arm"]=="semi"]
        volumes=np.array([pair for r in rs for pair in r["volumes"]])
        seed_stats[s]=dict(dice=average([r["values"]["paired"] for r in rs]),true_median_ml=float(np.median(volumes[:,0])),
                           predicted_median_ml=float(np.median(volumes[:,1])),empty=int((volumes[:,1]==0).sum()))
    result=dict(variant=variant,design=DESIGN,scope="Specified finding segmentation, 18 reused development patients, 36 targets. No final blind set.",
                n_clinical=len(cset),clinical=clinical,summary=stats,seeds=seed_stats,checkpoints=manifest,patient_results=rows,
                training_only_calibration=checks,verified_patient_maps=len(rows),code_sha256=sha(Path(__file__)))
    write(OUT/("results_"+variant+".json"),result)
    print("verified",variant,len(rows),flush=True)

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("variant",choices=("base","volume_term"));a=p.parse_args();run(a.variant)
