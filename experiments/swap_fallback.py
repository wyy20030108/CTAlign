"""Build ready-to-load labelled and report-only manifests; retain separate QC review items."""
import itertools,os,sys,tempfile
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
import nibabel as nib
from swap_data import out,read,sha256,write_json,obtain_file,release_path,FILES,reduce_ct,reduce_mask,side,template,norm,digest,read_reports
from audit_pairs import identity

dest=out/"fallback"
def recover(args):
    r,src=args
    n=r["name"]; p=dest/(r["code"]+".npz"); receipt=dest/(r["code"]+".json")
    if receipt.exists():return read(receipt)
    f=src["findings"]
    keys=[k for k in f if src["entity_counts"].get(k)==1 and src["pixels"].get(k,0)>0 and f[k].strip()]
    candidates=[(a,b) for a,b in itertools.combinations(sorted(keys,key=int),2) if norm(f[a])!=norm(f[b]) and [int(a),int(b)]!=r["channels"]]
    candidates.sort(key=lambda ab:({side(f[ab[0]]),side(f[ab[1]])}!={"left","right"},src["categories"][ab[0]]!=src["categories"][ab[1]],digest(n+"|".join(ab))))
    result={"status":"no_alternate"}
    if candidates:
        with tempfile.TemporaryDirectory() as tmp:
            repo,rev,_=FILES["ct_rate"];a,ah=obtain_file(repo,rev,release_path(n),tmp)
            repo,rev,_=FILES["rex"];b,bh=obtain_file(repo,rev,"segmentations/"+n,tmp)
            img=nib.load(a);seg=nib.load(b);assert img.shape==seg.shape[1:] and nib.aff2axcodes(img.affine)==("L","P","S")
            ct,aff=reduce_ct(img,2.); channels={}
            for k in keys:
                native=np.asarray(seg.dataobj[int(k)])>0
                if int(native.sum())==src["pixels"][k]:channels[k]=reduce_mask(native,img.affine,(ct.shape,aff)).astype(bool)
            for i,j in candidates:
                if i not in channels or j not in channels:continue
                x,y=channels[i],channels[j];smaller=min(x.sum(),y.sum())
                if smaller<8 or (x&y).sum()>.05*smaller:continue
                temp=p.with_suffix(".tmp.npz")
                np.savez_compressed(temp,ct=ct,affine=aff,masks=np.stack([x,y]));temp.replace(p)
                result=dict(status="ok",channels=[int(i),int(j)],queries=[f[i],f[j]],path=str(p),source_ct=ah,source_rex=bh,sha256=sha256(p))
                break
    write_json(receipt,result);return result

def main():
    os.umask(0o077); dest.mkdir(exist_ok=True);rows=read(out/"index.private.json")["pairs"]
    ids={r["name"] for r in rows};raw={r["name"]:r for r in read("metadata/rex/dataset.json")["train"] if r["name"] in ids}
    need=[r for r in rows if read(out/(r["code"]+".json"))["status"]!="ok"]
    with ThreadPoolExecutor(max_workers=3) as pool:
        recovered=dict(zip([r["code"] for r in need],pool.map(recover,[(r,raw[r["name"]]) for r in need])))
    docs=read_reports(ids);roles={s:[] for s in ("fallback_train","fallback_val","swap_dev","review")}
    totals=Counter()
    for r in rows:
        code=r["code"]; rec=read(out/(code+".json"))
        t=dict(r)
        if rec["status"]=="ok":
            t.update(path=str(out/(code+".npz")),sha256=rec["sha256"],source_ct=rec["source_ct"],source_rex=rec["source_rex"])
        else:
            alt=recovered[code]
            if alt["status"]!="ok":continue
            t.update(alt); totals["alternate_pairs"]+=1
        assert sha256(Path(t["path"]))==t["sha256"]
        queries=t["queries"]; t["templates"]=[template(q) for q in queries];t["sides"]=[side(q) for q in queries]
        t["categories"]=[raw[t["name"]]["categories"][str(k)] for k in t["channels"]]
        t["native_counts"]=[raw[t["name"]]["pixels"][str(k)] for k in t["channels"]]
        t["report"]=docs[t["name"]];t["verbatim_in_clinical_report"]=[any(norm(q) in norm(v) for v in t["report"].values()) for q in queries]
        with np.load(t["path"]) as v:
            ct,m,aff=v["ct"],v["masks"],v["affine"];assert m.shape==(2,*ct.shape) and np.allclose(aff[:3,:3],np.eye(3)*2)
            t["side_qc_fraction"]=[]
            for s,mask in zip(t["sides"],m):
                xs=np.nonzero(mask)[0]-(ct.shape[0]-1)/2
                t["side_qc_fraction"].append(float(((xs>0) if s=="right" else (xs<0)).mean()) if s else None)
        bad=any(z is not None and z<.20 for z in t["side_qc_fraction"])
        roles["review" if bad else r["role"]].append(t)
    for role,items in roles.items():write_json(dest/(role+".private.json"),items)
    u=read("phase1/data/U_report_strata32/manifest.private.json")["rows"]
    udocs=read_reports({r["name"] for r in u});names={identity(n)[1] for n in ids}
    assert len(names)==len(ids)
    for r in u:
        assert identity(r["name"])[1] not in names and sha256(Path(r["file"]))==r["npz_sha256"]
        with np.load(r["file"]) as v:assert set(v.files)=={"ct_hu","affine"}
        r["report"]=udocs[r["name"]]
    write_json(dest/"unlabelled_train.private.json",u)
    counts={k:len(v) for k,v in roles.items()}
    write_json(dest/"receipt.json",dict(labelled_patients_by_role=counts,report_only_U=len(u),alternate_pairs=totals["alternate_pairs"],
             clinical_report_and_rex_query_separate=True,grid_mm=2,sha256_lists={p.name:sha256(p) for p in dest.glob("*.private.json")}))
    print(counts,"U",len(u),"alternate",totals["alternate_pairs"],flush=True)
if __name__=="__main__":main()
