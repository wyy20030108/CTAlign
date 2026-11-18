"""Build a patient-partitioned set of independently identifiable ReX finding pairs."""
import csv, hashlib, itertools, json, os, re, sys, tempfile, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import nibabel as nib
import numpy as np
from fetch_four import sha256, write_json
from fetch_manifest import FILES
from prepare_unlabeled import obtain_file, release_path
from replay_controls import read_reports
sys.path.insert(0,str(Path("phase1/geometry").resolve()))
from geometry import reduce_ct, reduce_mask

out=Path("phase3_swap")
def read(p): return json.loads(Path(p).read_text())
def digest(v): return hashlib.sha256(v.encode()).hexdigest()
def norm(v): return " ".join(re.findall(r"[a-z0-9]+",v.lower()))
def side(v):
    s=set(re.findall(r"\b(left|right|bilateral|bilaterally)\b",v.lower()))
    return next(iter(s)) if len(s)==1 and s<={"left","right"} else None
def template(v):
    words=re.findall(r"\b(?:effusions?|nodul(?:e|es|ar)|consolidations?|atelecta\w*|emphysema\w*|calcifi\w*|opacit(?:y|ies)|thickening|bronchiectasis|fibro\w*|masses|mass|cysts?|fractures?|cardiomegaly|aneurysm|pneumothorax|herni\w*)\b",v.lower())
    return " ".join(dict.fromkeys(words)) or "abnormal finding"

def screen():
    with Path("phase1/cohort/split_v1/b10/T.csv").open() as f: rows=list(csv.DictReader(f))
    assert len(rows)==len({r["patient"] for r in rows})==190
    allowed={r["volume"] for r in rows}
    used=set(read("data/four_train/pilot_T_reservation.json")["volumes"])
    partial=Path("phase1/data/T_partial")
    used.update(p.stem+".nii.gz" for p in partial.glob("*.npz"))
    used.update(r["name"] for r in read(partial/"native_review.private.json"))
    good={r["volume"] for r in (json.loads(s) for s in Path("phase1/silver_audit/b10/paid_T_finding_candidates.private.jsonl").read_text().splitlines()) if r["data_check"]=="listed_calibration_finite"}
    names=(allowed-used)&good
    reports=read_reports(names)
    result=[]
    for row in read("metadata/rex/dataset.json")["train"]:
        n=row["name"]
        if n not in names: continue
        fs=row["findings"]
        candidates=[]
        for a,b in itertools.combinations(sorted(fs,key=int),2):
            if any(row["entity_counts"].get(k)!=1 or row["pixels"].get(k,0)<=0 or not fs[k].strip() for k in (a,b)): continue
            if norm(fs[a])==norm(fs[b]): continue
            opposite={side(fs[a]),side(fs[b])}=={"left","right"}
            candidates.append((not opposite,row["categories"][a]!=row["categories"][b],digest(n+a+"|"+b),a,b))
        if candidates:
            _,_,_,a,b=min(candidates)
            doc={k:norm(v) for k,v in reports[n].items()}
            result.append(dict(name=n,channels=[int(a),int(b)],queries=[fs[a],fs[b]],templates=[template(fs[a]),template(fs[b])],
                categories=[row["categories"][a],row["categories"][b]],sides=[side(fs[a]),side(fs[b])],native_counts=[row["pixels"][a],row["pixels"][b]],
                verbatim_in_clinical_report=[any(norm(fs[k]) in v for v in doc.values()) for k in (a,b)]))
    result.sort(key=lambda r:digest("swap-20261118"+r["name"]))
    # Interleave roles by deterministic hash order. All are part of the original paid T budget.
    for i,r in enumerate(result):
        r.update(code=f"S{i+1:03}",role=("swap_dev","fallback_train","fallback_train","fallback_val")[i%4])
    lock=dict(initial=190,prior_excluded=len(allowed&used),fresh_calibrated=len(names),pairs=result)
    index=out/"index.private.json"
    if index.exists(): assert read(index)==lock
    else: write_json(index,lock)
    return result

def prepare(r):
    n=r["name"]; code=r["code"]; record=out/(code+".json"); pack=out/(code+".npz")
    if record.exists():
        v=read(record)
        if v["status"]=="ok": assert sha256(pack)==v["sha256"]
        return v
    with tempfile.TemporaryDirectory(dir="/tmp") as tmp:
        cr,cv,_=FILES["ct_rate"]; mr,mv,_=FILES["rex"]
        a,ah=obtain_file(cr,cv,release_path(n),tmp); b,bh=obtain_file(mr,mv,"segmentations/"+n,tmp)
        img=nib.load(a); seg=nib.load(b)
        assert img.shape==seg.shape[1:] and nib.aff2axcodes(img.affine)==("L","P","S")
        ct,aff=reduce_ct(img,2.)
        masks=[]
        for k,num in zip(r["channels"],r["native_counts"]):
            raw=np.asarray(seg.dataobj[k,:,:,:])>0
            # Released counts may refer to per-entity summaries; require exact native support here.
            if int(raw.sum())!=num:
                v=dict(code=code,status="native_count_mismatch"); write_json(record,v); return v
            masks.append(reduce_mask(raw,img.affine,(ct.shape,aff)).astype(bool))
        x,y=masks
        v=dict(code=code,role=r["role"],source_ct=ah,source_rex=bh)
        if min(x.sum(),y.sum())<8 or (x&y).sum()>0.05*min(x.sum(),y.sum()):
            v.update(status="small_or_overlapping")
        else:
            tmpout=pack.with_suffix(".tmp.npz")
            np.savez_compressed(tmpout,ct=ct,masks=np.stack(masks),affine=aff)
            tmpout.replace(pack)
            v.update(status="ok",sha256=sha256(pack),voxels=[int(x.sum()),int(y.sum())])
        write_json(record,v);return v

def work(r):
    for attempt in range(3):
        try: return prepare(r)
        except Exception as e:
            print(r["code"],type(e).__name__,str(e).split("?")[0][:130],flush=True)
            if attempt==2: raise
            time.sleep(2)

if __name__=="__main__":
    os.umask(0o077);out.mkdir(exist_ok=True)
    rows=screen();print("screened",len(rows),flush=True)
    with ThreadPoolExecutor(max_workers=4) as ex:
        for v in ex.map(work,sorted(rows,key=lambda r:r["role"]!="swap_dev")):
            print(v["code"],v["status"],flush=True)
    write_json(out/"prepared.json",{"records":[read(out/(r["code"]+".json")) for r in rows]})
