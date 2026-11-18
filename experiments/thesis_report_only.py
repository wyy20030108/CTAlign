"""Freeze and materialize a report-only patient cohort from CT-RATE."""
from argparse import ArgumentParser
from collections import Counter,defaultdict
from hashlib import sha256
from pathlib import Path
import csv,json,os,re,shutil,tempfile,time
import nibabel as nib
import numpy as np
from thesis_expand_screen import ROOT,REX,CHALLENGE,REPORTS,TYPE,PATTERN,REPORT_BAD,FIBROSIS,QUESTIONABLE
from atelectasis_screen import anatomy,norm,side
from audit_pairs import identity,split_names
from fetch_manifest import FILES
from frozen_priors import read,write,sha
from diag_localize import seal
from prepare_unlabeled import release_path
from swap_data import reduce_ct
import thesis_transfer as transfers

U=ROOT/"report_only"
COUNT=36

def rank(k):return sha256(("thesis-U-v1|"+k).encode()).hexdigest()
def screen():
    U.mkdir(exist_ok=True)
    annotated=set()
    for file in (REX,CHALLENGE):
        annotated.update(identity(name)[1] for files in split_names(file).values() for name in files)
    avoid=set()
    for f in Path("data/four_train/ct_rate/dataset/metadata").glob("no_chest_*.txt"):
        avoid.update(Path(v.strip()).name for v in f.read_text().splitlines())
    cases=defaultdict(list)
    path=next(f for f in REPORTS if f.name=="train_reports.csv")
    with path.open(encoding="utf-8-sig",newline="") as file:
        for row in csv.DictReader(file):
            name,patient,_=identity(row["VolumeName"])
            if patient in annotated or name in avoid or not name.endswith("_1.nii.gz"):continue
            text=(row["Findings_EN"] or "").strip()
            matches=[]
            for sentence in re.split(r"(?<=[.!?])\s+|\n+",text):
                sentence=sentence.strip()
                if not (TYPE.search(sentence) and PATTERN.search(sentence) and side(sentence)):
                    continue
                if REPORT_BAD.search(sentence) or QUESTIONABLE.search(sentence) or FIBROSIS.search(sentence):
                    continue
                lobes,_=anatomy(sentence)
                if len(lobes)==1:matches.append((sentence,lobes.pop()))
            if len({norm(q) for q,_ in matches})!=1:continue
            query,lobe=matches[0]
            assert query in text
            cases[patient].append(dict(name=name,query=query,site=side(query)+"_"+lobe))
    selected_by_site=defaultdict(list)
    for entries in cases.values():
        r=min(entries,key=lambda item:rank(item["name"]))
        selected_by_site[r["site"]].append(r)
    batches={k:sorted(v,key=lambda r:rank(r["name"])) for k,v in selected_by_site.items()}
    main=("right_middle","left_lingular","right_lower","left_lower")
    chosen=[]
    while len(chosen)<COUNT and any(batches.get(k) for k in main):
        for k in main:
            if batches.get(k) and len(chosen)<COUNT:chosen.append(batches[k].pop(0))
    if len(chosen)<COUNT:
        other=sorted([row for rs in batches.values() for row in rs],key=lambda r:rank(r["name"]))
        chosen.extend(other[:COUNT-len(chosen)])
    for i,r in enumerate(chosen,1):r["code"]=f"U{i:03d}"
    seal(U/"source.private.json",chosen)
    seal(U/"cohort.json",dict(eligible=len(cases),selected=len(chosen),sites=dict(Counter(r["site"] for r in chosen)),
                             exclusion="ReX release and challenge patients in every split excluded. No masks requested.",
                             source_sha=sha(U/"source.private.json")))
    print(read(U/"cohort.json"),flush=True)

def fetch(shard,shards):
    (U/"arrays").mkdir(exist_ok=True);(U/"receipts").mkdir(exist_ok=True)
    transfers.ROOT=ROOT
    for r in read(U/"source.private.json")[shard::shards]:
        code=r["code"];rec=U/"receipts"/(code+".json")
        out=U/"arrays"/(code+".npz")
        if rec.exists():
            assert sha(out)==read(rec)["processed_sha"];continue
        for trial in range(3):
            try:
                if shutil.disk_usage(ROOT).free < 2*1024**3:
                    raise OSError("data disk has under 2 GiB free")
                with tempfile.TemporaryDirectory(prefix="report-only-",dir=U) as stage:
                    repo,rev,_=FILES["ct_rate"]
                    source,sh=transfers.download(repo,rev,release_path(r["name"]),stage)
                    ct,aff=reduce_ct(nib.load(source),1.5)
                    assert np.isfinite(ct).all() and ct.min()<-400 and ct.max()>50
                    temp=out.with_suffix(".part.npz")
                    np.savez_compressed(temp,ct=ct,affine=aff);temp.replace(out)
                write(rec,dict(code=code,source_sha=sh,processed_sha=sha(out),shape=ct.shape,contains_mask=False))
                print("complete",code,flush=True)
                break
            except Exception as exc:
                print(code,trial,type(exc).__name__,str(exc).split("http",1)[0][:45],flush=True)
                if trial==2:write(rec.with_suffix(".pending.json"),dict(code=code,error=type(exc).__name__))
                time.sleep(3)
if __name__=="__main__":
    os.umask(0o077)
    p=ArgumentParser();p.add_argument("stage",choices=("screen","fetch"));p.add_argument("--shard",type=int,default=0);p.add_argument("--shards",type=int,default=4)
    a=p.parse_args()
    if a.stage=="screen":screen()
    else:fetch(a.shard,a.shards)

