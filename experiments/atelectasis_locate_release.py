"""Verify each full-volume mask and summarize the paired supervised comparison."""
import io,json,os
from collections import defaultdict
from pathlib import Path
import zipfile

import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
import numpy as np

from atelectasis_locate_refine import OUT,ROOT,RUNS,SLIDING,load_row,score,box
from frozen_priors import read,sha,write

ARMS=("B1","B1_0p2","B1_restricted","located_refined")

def group(rows):
    codes=sorted({r["code"] for r in rows})
    per_model={}
    patients={}
    for arm in ARMS:
        data=[r["metrics"][arm] for r in rows]
        patients[arm]={}
        for code in codes:
            items=[r["metrics"][arm] for r in rows if r["code"]==code]
            patients[arm][code]={k:float(np.mean([i[k] for i in items])) for k in ("dice","recall","fp_ml")}
            patients[arm][code]["missed"]=sum(i["miss"] for i in items)
            patients[arm][code]["n"]=len(items)
        means={k:float(np.mean([i[k] for i in patients[arm].values()])) for k in ("dice","recall","fp_ml")}
        per_model[arm]=dict(**means,miss=sum(x["miss"] for x in data),empty=sum(x["empty"] for x in data),
                               hit10=sum(x["recall"]>=.10 for x in data),
                               sustained_hit10=sum(sum(x["metrics"][arm]["recall"]>=.10 for x in rows if x["code"]==c)>
                                                  patients[arm][c]["n"]/2 for c in codes),
                               fp_median=float(np.median([x["fp_ml"] for x in data])),
                               evals=len(data),patients=len(codes))
    paired={}
    for k in ("dice","recall","fp_ml"):
        vals=np.array([patients["located_refined"][c][k]-patients["B1"][c][k] for c in codes])
        indexes=np.random.default_rng(30841).integers(len(vals),size=(10000,len(vals)))
        paired[k]={"mean":float(vals.mean()),"ci95":np.quantile(vals[indexes].mean(1),(.025,.975)).tolist(),
                   "patients_increased":int(np.count_nonzero(vals>0))}
    cov1=[r["recall_of_proposals"][0] for r in rows]
    cov4=[r["recall_of_proposals"][-1] for r in rows]
    info=dict(top1_50=sum(v>=.5 for v in cov1),top4_50=sum(v>=.5 for v in cov4),top4_any=sum(v>0 for v in cov4),
              top4_mean_coverage=float(np.mean(cov4)),mean_lung_support=float(np.mean([r["target_within_lung_support"] for r in rows])),
              mean_search_ml=float(np.mean([r["search_voxels"]*1.5**3/1000 for r in rows])),
              missed_despite_coverage50=sum(c>=.5 and r["metrics"]["located_refined"]["miss"] for c,r in zip(cov4,rows)),
              below10_despite_coverage50=sum(c>=.5 and r["metrics"]["located_refined"]["recall"]<.10 for c,r in zip(cov4,rows)),
              n=len(rows))
    return dict(model=per_model,patient=patients,paired=paired,localization=info)

def draw(development,budget,seed):
    fig, axes=plt.subplots(6,3,figsize=(10,16),squeeze=False)
    folder=OUT/budget/str(seed)
    for i,row in enumerate(development):
        ct,gt=load_row(row);code=row["code"]
        z=int(np.argmax(gt.sum((0,1))))
        pack=folder/(code+".npz")
        with np.load(pack) as v:
            a,b=[np.unpackbits(v[x],count=gt.size).reshape(gt.shape)[:,:,z].astype(bool) for x in ("B1","located_refined")]
            centers=v["proposals"].copy()
        for j,(ax,pred) in enumerate(zip(axes[i],(None,a,b))):
            ax.imshow(ct[:,:,z].T,origin="lower",cmap="gray",vmin=-1000,vmax=180)
            ax.contour(gt[:,:,z].T,levels=[.5],colors=["#ffbb4c"],origin="lower",linewidths=1.1)
            if pred is not None and pred.any():ax.contour(pred.T,levels=[.5],colors=["#53c2ff"],origin="lower",linewidths=1)
            if j==2:
                from matplotlib.patches import Rectangle
                for xyz in centers:
                    x,y,plane=box(xyz,ct.shape)
                    if plane.start<=z<plane.stop:
                        ax.add_patch(Rectangle((x.start,y.start),x.stop-x.start,y.stop-y.start,fill=False,
                                               edgecolor="#f197a7",linewidth=.55,linestyle="--"))
            ax.set_title(code+" "+("Reference","Sliding B1","Locate then refine")[j],fontsize=8)
            ax.axis("off")
    fig.suptitle(f"Matched CT slices | {budget}, seed {seed} | Yellow reference, blue prediction, pink candidate",fontsize=10)
    fig.tight_layout(rect=(0,0,1,.98))
    dest=OUT/(budget+"_overlays.png")
    fig.savefig(dest,dpi=120);plt.close(fig);return dest

def main():
    os.umask(0o077)
    cohort=read(ROOT/"pilot_private.json");dev=cohort["development"];by_code={r["code"]:r for r in dev}
    assert set(by_code).isdisjoint(r["code"] for r in cohort["holdout"])
    checked=[]; entries=[OUT/"protocol.json"]
    for budget,seed in RUNS:
        folder=OUT/budget/str(seed);f=folder/"results.json";r=read(f)
        training=read(folder/"training.json")
        assert len(training["loss"])==384 and r["locator_weights"]==sha(folder/"locator.pt")==training["sha"]
        old=read(SLIDING/budget/str(seed)/"results.json")
        assert r["train_codes"]==old["train_codes"] and r["previous_weights"]==old["model_sha"]
        assert r["protocol"]==sha(OUT/"protocol.json") and {u["code"] for u in r["rows"]}==set(by_code)
        for u in r["rows"]:
            code=u["code"];path=folder/(code+".npz")
            assert sha(path)==u["pred_sha"]
            ct,m=load_row(by_code[code]);del ct
            with np.load(path) as a:
                assert tuple(a["shape"])==m.shape and a["proposals"].shape==(4,3)
                proposal_union=np.zeros(m.shape,dtype=bool)
                for i,center in enumerate(a["proposals"]):
                    proposal_union[box(center,m.shape)]=True
                    assert np.isclose(np.count_nonzero(m&proposal_union)/m.sum(),u["recall_of_proposals"][i])
                for key in ARMS:
                    q=np.unpackbits(a[key],count=m.size).reshape(m.shape).astype(bool)
                    assert score(q,m)==u["metrics"][key]
                    if key=="located_refined":
                        assert not np.any(q&~proposal_union)
                        assert q.sum()*1.5**3/1000<=u["cap_ml"]
                previous=SLIDING/budget/str(seed)/(code+".npz")
                assert sha(previous)==u["prior_prediction_sha"]
                with np.load(previous) as old_mask:assert np.array_equal(a["B1"],old_mask["B1"])
            checked.append(dict(budget=budget,seed=seed,**u))
            entries.append(path)
        entries.extend([f,folder/"locator.pt",folder/"training.json"])
    assert len(checked)==24
    groups={k:group([x for x in checked if x["budget"]==k]) for k in ("b10","all_labeled")}
    summary=OUT/"summary.json"
    write(summary,dict(groups=groups,rows=checked,n=24,holdout_evaluated=False,
                       same_prior_mask_verified=True,protocol=sha(OUT/"protocol.json")))
    entries.extend([summary,draw(dev,"b10",891),draw(dev,"all_labeled",891)])
    archive=OUT/"locate_evidence.zip";tmp=archive.with_suffix(".part")
    with zipfile.ZipFile(tmp,"w",zipfile.ZIP_DEFLATED) as z:
        values={str(f.relative_to(OUT)):f for f in entries}
        values.update({f"code/{n}":Path(n) for n in ("atelectasis_locate_refine.py","atelectasis_locate_release.py")})
        for name,path in values.items():z.write(path,name)
        z.writestr("checksums.json",json.dumps({k:sha(v) for k,v in values.items()}))
    tmp.replace(archive)
    for k,v in groups.items():
        print(k,json.dumps(dict(model=v["model"],paired=v["paired"],localization=v["localization"])),flush=True)
    print("archive",sha(archive))
if __name__=="__main__":main()

