"""Independently recount the locked thesis results and prepare the evidence package."""
from collections import defaultdict
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
import ast,json,os,sys,zipfile
import numpy as np
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt

from thesis_cohort import ROOT,LOCK,MANIFEST,PRIMARY
from thesis_evaluate import EVAL,RESULTS,MEASURES,train_file
from thesis_dual import OUT as DUAL
from thesis_dual_eval import PAIR_LOCK
from thesis_ssl import OUT as CONTROL,ARMS,SEEDS
from thesis_model import score
from frozen_priors import read,sha,write
from thesis_soft_eval import OUT as SOFT, LOCK as SOFT_LOCK, RESULTS as SOFT_RESULTS


def source_dependencies(sources):
    """Carry project-local Python imports alongside the experiment entry points."""
    queue=list(sources)
    visited=set()
    while queue:
        path=queue.pop()
        if path in visited:
            continue
        visited.add(path)
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8-sig"))):
            if isinstance(node,ast.Import):
                names=[item.name for item in node.names]
            elif isinstance(node,ast.ImportFrom) and node.module and node.level==0:
                names=[node.module]
            else:
                continue
            for name in names:
                dependency=Path(name.split(".")[0]+".py")
                if dependency.is_file():
                    queue.append(dependency)
    return sorted(visited)


def reproduction_notes():
    packages=("torch","numpy","scipy","nibabel","matplotlib","requests","huggingface-hub","lungmask","SimpleITK")
    versions={}
    for name in packages:
        try:
            versions[name]=version(name)
        except PackageNotFoundError:
            continue
    return dict(python=sys.version.split()[0],packages=versions,
                source="Original CT, report text, case identity mapping and reference arrays are authorized, server-only inputs.",
                main=["python thesis_evaluate.py lock","python thesis_test_pipeline.py"],
                validation="Compare the archived evaluation lock, weights, evidence and per-patient tables with SHA256.json.",
                location="Source scripts assume the original relative server work directory and frozen lobe weights.")

def paired(a,b,metric="dice"):
    assert set(a)==set(b)
    vals=np.array([a[k][metric]-b[k][metric] for k in sorted(a)],dtype=float)
    idx=np.random.default_rng(191102).integers(len(vals),size=(20000,len(vals)))
    ci=np.quantile(vals[idx].mean(1),[.025,.975]).tolist()
    return dict(delta=float(vals.mean()),ci95=ci,n=len(vals),n_positive=int(np.count_nonzero(vals>0)))

def average(vals):
    present=[v for v in vals if v is not None]
    return None if not present else float(np.mean(present))

def collapse(rows,key):
    patients=defaultdict(list)
    for r in rows:
        if key in r["outcomes"]:patients[r["code"]].append(r["outcomes"][key])
    result={code:{m:average([x[m] for x in rr]) for m in MEASURES} for code,rr in patients.items()}
    summary={m:average([r[m] for r in result.values()]) for m in MEASURES}
    summary.update(patients=len(result),n=len(rows),nonempty_hd95=sum(x["hd95"] is not None for rr in patients.values() for x in rr))
    return summary,result

def figures(package,cohort):
    p=package["primary_budget"];base=package["reference"]
    result=package["independent"];a=package["patient"][p]
    arms=list(ARMS)
    fig,axs=plt.subplots(1,3,figsize=(12.8,4.2),layout="constrained")
    y=[100*result[p][k]["summary"]["dice"] for k in arms]
    colors=["#567080" if x not in (base,"F") else "#255e68" if x=="F" else "#916847" for x in arms]
    axs[0].bar(arms,y,color=colors)
    for i,v in enumerate(y):axs[0].text(i,v+.15,f"{v:.1f}",ha="center",fontsize=8)
    axs[0].set(xlabel="Locked comparison",ylabel="Mean Dice (%)",title="Independent cohort")
    codes=sorted(a[base],key=lambda k:a["F"][k]["dice"]-a[base][k]["dice"])
    delta=[100*(a["F"][k]["dice"]-a[base][k]["dice"]) for k in codes]
    axs[1].bar(range(len(codes)),delta,color=["#255e68" if v>0 else "#a17c66" for v in delta])
    axs[1].axhline(0,color="black",linewidth=.65)
    axs[1].set(xlabel="Patients ordered by paired difference",ylabel="Dice difference (pp)",title=f"F versus {base}")
    actual=[package["cohort"]["budget"][k] for k in ("5","10","20")]
    for arm,color in (("B","#916847"),("F","#255e68")):
        axs[2].plot(actual,[100*result[k][arm]["summary"]["dice"] for k in ("5","10","20")],
                    marker="o",label=arm,color=color)
    axs[2].set(xticks=actual,xlabel="Exposed masks",ylabel="Mean Dice (%)",title="Matched label budgets")
    axs[2].legend(frameon=False)
    image=ROOT/"independent_summary.png";fig.savefig(image,dpi=180);plt.close(fig)
    selected=list(dict.fromkeys((codes[0],codes[len(codes)//2],codes[-1])))
    chart,axs=plt.subplots(len(selected),3,figsize=(9,3.0*len(selected)),squeeze=False,layout="constrained")
    lookup={r["code"]:r for r in cohort["test"]}
    seed=SEEDS[0]
    for i,c in enumerate(selected):
        row=lookup[c]
        with np.load(row["path"]) as v:ct=v["ct"];truth=v["mask"]
        axial=int(np.argmax(truth.sum((0,1))))
        masks=[]
        for arm in (base,"F"):
            with np.load(RESULTS/p/str(seed)/arm/(c+".npz")) as v:
                masks.append(np.unpackbits(v["correct"],count=truth.size).reshape(truth.shape)[:,:,axial])
        for j,axis in enumerate(axs[i]):
            axis.imshow(ct[:,:,axial].T,origin="lower",cmap="gray",vmin=-1000,vmax=150)
            axis.contour(truth[:,:,axial].T,levels=[.5],colors=["#ffbb4c"],origin="lower",linewidths=.8)
            if j and masks[j-1].any():axis.contour(masks[j-1].T,levels=[.5],colors=["#54c6fa"],origin="lower",linewidths=.8)
            axis.set_title(c+" "+("reference",base,"F")[j]);axis.axis("off")
    chart.suptitle(f"Selected extremes and median of paired differences | Seed {seed}")
    overlays=ROOT/"independent_overlays.png";chart.savefig(overlays,dpi=140);plt.close(chart)
    return image,overlays

def run():
    os.umask(0o077)
    f=read(EVAL);census=read(LOCK);cohort=read(MANIFEST)
    assert f["manifest"]==census["manifest_sha"]==sha(MANIFEST)
    assert f["training_protocol"]==sha(CONTROL/"protocol.json")
    cases={row["code"]:row for row in cohort["test"]}
    truth={}
    for code,row in cases.items():
        path=Path(row["path"]);assert sha(path)==row["sha"]
        with np.load(path) as a:truth[code]=a["mask"].astype(bool)
    reports={};patients={}
    files=[LOCK,EVAL,CONTROL/"protocol.json",PAIR_LOCK,ROOT/"component_equivalence.json",
           ROOT/"screen.json",ROOT/"report_only/cohort.json",ROOT/"expanded_lobe_audit.json",
           ROOT/"development_teacher_audit.json"]
    checked=[]
    for budget,arms in f["arms"].items():
        reports[budget]={};patients[budget]={}
        for arm,design in arms.items():
            rows=[]
            pseudo=[]
            for seed in SEEDS:
                model,dev,fit=train_file(budget,seed,arm)
                selected=next(x for x in design["checkpoints"] if x["seed"]==seed)
                assert selected["model"]==sha(model/"model.pt")
                d=RESULTS/budget/str(seed)/arm
                completed=read(d/"finished.json")
                assert completed["lock"]==sha(EVAL) and completed["threshold"]==design["chosen"]
                assert {r["code"] for r in completed["rows"]}==set(cases)
                for r in completed["rows"]:
                    code=r["code"];m=truth[code];p=d/(code+".npz")
                    assert sha(p)==r["prediction_sha"] and r["locked"]==sha(EVAL)
                    with np.load(p) as a:
                        for key,expected in r["outcomes"].items():
                            y=np.unpackbits(a[key],count=m.size).reshape(m.shape).astype(bool)
                            assert all(expected[k]==v for k,v in score(y,m).items())
                    rows.append(r)
                    files.extend([p,d/(code+".json")])
                files.extend([d/"finished.json",model/"training.json",model/"model.pt",model/"development.json"])
                logs=fit["logs"]
                pseudo.append(dict(seed=seed,accepted=sum(x["accepted"] for x in logs),steps=len(logs),
                    scans_seen=len({x["u"] for x in logs if "u" in x}),
                    accepted_scans=len({x["u"] for x in logs if x["accepted"]}),
                    pos=int(sum(x["pos"] for x in logs)),foreground_prior=average([x["label_mass"] for x in logs if "label_mass" in x]),
                    transported_fg_before=average([x["pre_transport"] for x in logs if "pre_transport" in x]),
                    transported_fg_after=average([x["post_transport"] for x in logs if "post_transport" in x])))
            report,data=collapse(rows,"correct")
            by_site={k:collapse([r for r in rows if r["site"]==k],"correct")[0] for k in sorted({p["site"] for p in rows})}
            by_volume={k:collapse([r for r in rows if pred(r["n_voxels"]*1.5**3/1000)],"correct")[0]
                        for k,pred in (("lt_1ml",lambda v:v<1),("ge_1ml",lambda v:v>=1))}
            query={}
            for option in ("wrong_location","site_template","empty"):
                overall,values=collapse(rows,option)
                if values:query[option]=dict(metrics=overall,dice_diff=paired(data,values))
            reports[budget][arm]=dict(summary=report,threshold=design["chosen"],by_site=by_site,by_volume=by_volume,
                                       pseudolabel=pseudo,query_controls=query,dev_dice=design["development_dice"])
            patients[budget][arm]=data
    bars=patients[PRIMARY];comparison=[("B","A"),("D","B"),("D","C"),("D","E"),("F","D"),("F",f["comparator"])]
    contrast={a+"_"+b:{m:paired(bars[a],bars[b],m) for m in ("dice","recall","fp_ml")} for a,b in comparison}
    estimate=contrast["F_"+f["comparator"]]["dice"]
    pairs=read(PAIR_LOCK);dual_results={}
    refs={}
    for row in pairs["pairs"]:
        code=row["code"];p=DUAL/(code+".npz");assert sha(p)==row["mask_sha"]
        with np.load(p) as a:refs[code]=(truth[code],a["mask"].astype(bool))
    for arm in "BDF":
        real=defaultdict(list)
        false=defaultdict(list)
        for seed in SEEDS:
            d=DUAL/"scores"/str(seed)/arm
            got=read(d/"finished.json")
            assert got["lock"]==sha(PAIR_LOCK) and {r["code"] for r in got["rows"]}==set(refs)
            files.append(d/"finished.json")
            for row in got["rows"]:
                code=row["code"];p=d/(code+".npz")
                assert sha(p)==row["sha"]
                with np.load(p) as out:
                    a,b=[np.unpackbits(out[k],count=refs[code][0].size).reshape(refs[code][0].shape).astype(bool)
                         for k in ("A","B")]
                matrix=row["matrix"]
                for i,y in enumerate((a,b)):
                    for j,r in enumerate(refs[code]):
                        assert all(matrix[i][j][k]==v for k,v in score(y,r).items())
                real[code].append(.5*(matrix[0][0]["dice"]+matrix[1][1]["dice"]))
                false[code].append(.5*(matrix[0][1]["dice"]+matrix[1][0]["dice"]))
                files.extend([p,d/(code+".json")])
        if real:
            correct={k:{"dice":average(v)} for k,v in real.items()}
            swapped={k:{"dice":average(v)} for k,v in false.items()}
            dual_results[arm]=dict(n=len(real),correct=average([x["dice"] for x in correct.values()]),
                                   swapped=average([x["dice"] for x in swapped.values()]),
                                   delta=paired(correct,swapped))
    soft=read(SOFT/"decision.json")
    files.extend([SOFT/"decision.json",SOFT/"protocol.json"])
    for seed in SEEDS:
        q=SOFT/PRIMARY/str(seed)/"F"
        fit=read(q/"training.json")
        assert fit["model_sha"]==sha(q/"model.pt") and fit["protocol"]==sha(SOFT/"protocol.json")
        files.extend(q/p for p in ("model.pt","development.json","training.json"))
    if soft["auxiliary_independent"]:
        files.append(SOFT_LOCK)
        r=[]
        for seed in SEEDS:
            q=SOFT_RESULTS/PRIMARY/str(seed)/"F"
            run_result=read(q/"finished.json")
            assert run_result["lock"]==sha(SOFT_LOCK) and {p["code"] for p in run_result["rows"]}==set(cases)
            for row in run_result["rows"]:
                c=row["code"];m=truth[c];p=q/(c+".npz")
                assert sha(p)==row["prediction_sha"]
                with np.load(p) as a:
                    for option,wanted in row["outcomes"].items():
                        y=np.unpackbits(a[option],count=m.size).reshape(m.shape).astype(bool)
                        assert all(wanted[k]==v for k,v in score(y,m).items())
                files.extend([p,q/(c+".json")])
            r.extend(run_result["rows"]);files.append(q/"finished.json")
        s,pts=collapse(r,"correct")
        soft["independent"]=dict(metrics=s,dice=paired(pts,bars[f["comparator"]]))
    package=dict(cohort=census,independent=reports,patient=patients,contrasts=contrast,reference=f["comparator"],
                 primary_budget=PRIMARY,primary_advance=estimate["delta"]>=.03 and estimate["ci95"][0]>0,
                 genuine_within_ct_dual=dual_results,softened_transport=soft,primary_n=len(cases),lock=sha(EVAL),
                 teacher_audit=read(ROOT/"development_teacher_audit.json")["counts"])
    dest=ROOT/"completed_study.json";write(dest,package);files.append(dest)
    files.extend(figures(package,cohort))
    archive=ROOT/"thesis_study_evidence.zip"
    sources=[Path(f) for f in ("thesis_expand_screen.py","thesis_expand_fetch.py","thesis_transfer.py",
              "thesis_report_only.py","thesis_cohort.py","thesis_lobes.py","thesis_lobe_model.py","thesis_model.py",
              "thesis_ssl.py","thesis_pipeline.py","thesis_test_pipeline.py","thesis_evaluate.py","thesis_dual.py",
              "thesis_dual_eval.py","thesis_release.py","thesis_component_fast.py","thesis_accelerated.py",
              "thesis_soft_transport.py","thesis_soft_eval.py","thesis_teacher_audit.py")]
    notes=ROOT/"reproduction.json";write(notes,reproduction_notes());files.append(notes)
    temp=archive.with_suffix(".part")
    with zipfile.ZipFile(temp,"w",zipfile.ZIP_DEFLATED) as z:
        entries={str(p.relative_to(ROOT)):p for p in files}
        entries.update({"code/"+p.name:p for p in source_dependencies(sources)})
        for k,p in entries.items():z.write(p,k)
        z.writestr("SHA256.json",json.dumps({n:sha(p) for n,p in entries.items()}))
    temp.replace(archive)
    print("Primary",PRIMARY,"F against",f["comparator"],estimate,flush=True)
    print("QA file",archive,sha(archive),flush=True)
if __name__=="__main__":run()

