"""Check hard predictions against authorized original arrays, then summarize by patient."""
import json,zipfile,re,shutil
from pathlib import Path
import numpy as np
from frozen_priors import read,write,sha
from matched6 import OUT,GROUPS,SEEDS,SRC
from diag_localize import score_mask

def mean(x):return float(np.mean(x))
def ci(x):
    x=np.asarray(x);ix=np.random.default_rng(21043).integers(len(x),size=(10000,len(x)))
    return np.quantile(x[ix].mean(1),[.025,.975]).tolist()
old=read(Path("phase5_diag/localization/supervised/refine_results.json"))
old={r["code"]:r for r in old["rows"] if r["region"]=="oracle_2" and r["mode"]=="finding"}
official=read(Path("phase5_diag/localization/supervised/results.json"))
official={r["code"]:r["oracle_boxes"] for r in official["cases"]}
locks=read(OUT/"development_lock.json")
for f,h in {**locks["first_scores"],**locks["models"]}.items():assert sha(Path(f))==h
models=[(str(s),OUT/str(s)) for s in SEEDS]+[("exclude_S002_A",OUT/"sensitivity/891")]
groups={}
records=[]
eligible={}
def clean(t):return " ".join(re.findall(r"[a-z0-9]+",t.lower()))
for group in GROUPS:
    rows=read(SRC/(group+".private.json"));assert len(rows)==dict(zip(GROUPS,[16,10,8]))[group]
    scores={name:{r["code"]:r for r in read(folder/group/"scores.json")} for name,folder in models}
    patients=[];size=[];training_targets=[]
    cf=Path("phase3_swap/clinical")/(group+".private.json");clinical=read(cf) if cf.exists() else {}
    for row in rows:
        c=row["code"];p=Path(row["path"]);assert sha(p)==row["sha256"]
        with np.load(p) as z:m=z["masks"].astype(bool)
        per=[];text=clean(" ".join(row["report"].values()))
        source_ok=c not in clinical or all(clean(t) in text for t in clinical[c])
        assert source_ok
        for name,folder in models:
            r=scores[name][c];fp=folder/group/(c+".npz");assert sha(fp)==r["prediction"]
            with np.load(fp) as a:
                assert list(a["shape"])==list(m.shape[1:])
                loc=a["starts"];roi=score_mask(m.shape[1:],loc);truth=m[:,roi]
                pp=np.unpackbits(a["pred"],axis=1,count=int(a["width"])).astype(bool)
                assert np.array_equal(np.unpackbits(a["mask"],axis=1,count=int(a["width"])),truth)
                if c in official: assert {tuple(v) for v in loc}=={tuple(v) for v in official[c]}
                n=pp.sum(1);total=m.sum((1,2,3));inside=truth.sum(1)
                tp=np.array([[np.count_nonzero(x&y) for y in truth] for x in pp]);mat=2*tp/(n[:,None]+total)
                assert np.array_equal(tp,r["intersect"]) and np.allclose(mat,r["dice"],rtol=0,atol=1e-14)
                assert np.allclose(2*tp/(n[:,None]+inside),r["local_dice"])
                rec=dict(code=c,model=name,group=group,true=mean([mat[0,0],mat[1,1]]),swap=mean([mat[1,0],mat[0,1]]),
                    empty=mean(mat[2]),template=mean([mat[3,0],mat[4,1]]),template_swap=mean([mat[3,1],mat[4,0]]),
                    local=mean([r["local_dice"][0][0],r["local_dice"][1][1]]),empty_pred=mean(n[:2]==0))
                rec["clinical"]=mean([mat[5,0],mat[6,1]]) if len(mat)==7 else None
                rec["clinical_swap"]=mean([mat[6,0],mat[5,1]]) if len(mat)==7 else None
                records.append(rec)
                if name!="exclude_S002_A":
                    per.append(rec)
                    for j in (0,1):
                        category="small" if total[j]*.008<1 else "large"
                        size.append((category,c,j,mat[j,j],mat[1-j,j]))
                        if group==GROUPS[0]: training_targets.append((c,j,mat[j,j]))
        pat={k:mean([r[k] for r in per]) for k in ["true","swap","empty","template","template_swap","local","empty_pred"]}
        pat.update(code=c,delta=pat["true"]-pat["swap"],sides=scores["891"][c]["sides"],clinical=per[0]["clinical"] is not None)
        if pat["clinical"]:
            pat["clinical_true"]=mean([r["clinical"] for r in per]);pat["clinical_swap"]=mean([r["clinical_swap"] for r in per])
        if c in old:pat["historic"]=mean([v["dice_whole"] for v in old[c]["scores"]["correct"]])
        patients.append(pat)
    keys=["true","swap","empty","template","template_swap","local","empty_pred","delta"]
    out={k:mean([p[k] for p in patients]) for k in keys}
    out["n"]=len(patients);out["ci"]=ci([p["delta"] for p in patients])
    out["wins"]=sum(p["delta"]>0 for p in patients)
    out["loo_min"]=min(mean([s["delta"] for s in patients if s!=p]) for p in patients)
    for mode in ("opposite","other","clinical"):
        sub=[p for p in patients if (p["clinical"] if mode=="clinical" else ((set(p["sides"])=={"left","right"})==(mode=="opposite")))]
        out[mode]={"n":len(sub)}
        if sub:
            a,b=("clinical_true","clinical_swap") if mode=="clinical" else ("true","swap")
            out[mode].update(true=mean([p[a] for p in sub]),swap=mean([p[b] for p in sub]))
    for cat in ("small","large"):
        sub=[r for r in size if r[0]==cat];out[cat]=dict(targets=len(sub)//3,true=mean([r[3] for r in sub]),swap=mean([r[4] for r in sub]))
    if group!=GROUPS[0]:
        out["historical"]=mean([p["historic"] for p in patients]);out["advance"]=(out["true"]-out["historical"]>=.10 and out["ci"][0]>0)
    else:
        out["train_except_flagged"]=mean([d for c,j,d in training_targets if c!="S030" and (c,j)!=("S002",0)])
    groups[group]=dict(summary=out,patients=patients)
write(OUT/"summary.json",dict(groups=groups,records=records))
for n in ["matched6.py","matched6_pipeline.py","matched6_sensitivity.py","matched6_collect.py"]:
    shutil.copyfile(n,OUT/n)
files=[f for f in sorted(OUT.rglob("*")) if f.is_file() and f.name!="hashes.json"]
write(OUT/"hashes.json",{str(f.relative_to(OUT)):sha(f) for f in files})
with zipfile.ZipFile("matched6_evidence.zip","w",zipfile.ZIP_DEFLATED,compresslevel=2) as z:
    for f in files+[OUT/"hashes.json"]:z.write(f,str(f.relative_to(OUT)))
print("verified",len(files),"files",sha(Path("matched6_evidence.zip")))

