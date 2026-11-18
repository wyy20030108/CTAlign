"""Prospectively reserve a larger non-overlapping nodule cohort for fixed-model replication."""
from argparse import ArgumentParser
from collections import Counter,defaultdict
import csv
from pathlib import Path
import os
import re
import time

import atelectasis_prepare as prep
import thesis_transfer as transfer
from atelectasis_screen import CHALLENGE,REX
from audit_pairs import identity,split_names
from diag_localize import seal
from frozen_priors import read,sha,write
from joint_nodule_cohort import focal,query
from joint_nodule_run import COHORT,usable_lungs
from joint_roi import ROOT
from replay_controls import read_reports
from swap_data import side
from thesis_cohort import choose_stratified,site


DEST=ROOT/"nodule_replication"
STORAGE=Path("/tmp/joint_nodule_replication_data")
CANDIDATE=DEST/"candidate.private.json"
LOCK=DEST/"replication_manifest.private.json"
TARGET=112
PATIENT=re.compile(r"(?:train|valid|validation)_\d+_[a-z]+_\d+\.nii\.gz",re.I)


def previous_sources():
    roots=[Path(r) for r in ("phase1/data","phase2","phase3_swap","phase4_finding","phase5_diag","phase6_matched",
                             "phase7_atelectasis","phase8_thesis",str(ROOT/"nodule_v2"))]
    paths=[]
    for root in roots:
        paths.extend(root.rglob("*.private.json"))
    paths.extend([COHORT,Path("phase8_thesis/thesis_private.json")])
    seen=set()
    for p in sorted(set(paths)):
        for name in PATIENT.findall(p.read_text(encoding="utf-8-sig")):
            seen.add(identity(name)[1])
    for p in (COHORT,Path("phase8_thesis/thesis_private.json")):
        for rows in read(p).values():
            if isinstance(rows,list):
                seen.update(tuple(r["patient"].split(":")) for r in rows)
    for p in (REX,CHALLENGE):
        invalid=split_names(p)
        for k,v in invalid.items():
            if k!="train":seen.update(identity(n)[1] for n in v)
    return seen


def screen():
    DEST.mkdir(exist_ok=True,parents=True)
    STORAGE.mkdir(exist_ok=True,parents=True)
    seen=previous_sources()
    pending=[]
    for row in read(REX)["train"]:
        name,patient,_=identity(row["name"])
        if patient in seen:continue
        sites=Counter(site({"query":s}) for s in row["findings"].values() if focal(s))
        for k,s in row["findings"].items():
            if focal(s) and row["entity_counts"][k]==1 and row["pixels"][k]>0 and sites[site({"query":s})]==1:
                pending.append((name,patient,k,s,row["pixels"][k]))
    texts=read_reports({p[0] for p in pending})
    selected=defaultdict(list)
    for name,patient,k,pure,pixels in pending:
        matched=query(pure,texts[name]["Findings_EN"])
        if not matched:continue
        exact,context=matched
        assert exact in context and context in texts[name]["Findings_EN"]
        selected[patient].append(dict(name=name,patient=":".join(patient),channel=int(k),native_count=pixels,side=side(exact),
                                      query=exact,context=context,published=pure,site=site({"query":exact}),code=name+"-"+k))
    ready=[min(q,key=lambda r:r["code"]) for q in selected.values()]
    assert len(ready)>=TARGET
    cohort=choose_stratified(ready,TARGET,"externally-fixed-nodule-confirmation-v1")
    for i,row in enumerate(cohort,1):row["code"]=f"Y{i:03d}"
    assert len({r["patient"] for r in cohort})==TARGET
    seal(CANDIDATE,cohort)
    original=Path("phase9_joint/nodule_v2/external_matched/external_lock.json")
    seal(DEST/"design.json",dict(participants=TARGET,eligible_metadata=len(ready),past_patient_count=len(seen),
              candidate_sha=sha(CANDIDATE),frozen_model_and_thresholds=sha(original),
              scientific_question="Independently repeat all original contrasts, including the Adaptive-UOT module and full framework, unchanged. Primary result is the size and patient-level uncertainty of both additions to independently chosen stronger controls."))
    print("Candidates",len(cohort),"eligible",len(ready),"excluded prior and official partitions",len(seen),flush=True)


def fetch(shard,shards):
    prep.ROOT=STORAGE
    prep.DATA=STORAGE/"arrays"
    prep.RECEIPTS=DEST/"receipts"
    transfer.ROOT=STORAGE
    prep.obtain_file=transfer.download
    for p in (prep.DATA,prep.RECEIPTS):p.mkdir(exist_ok=True,parents=True)
    for row in read(CANDIDATE)[shard::shards]:
        code=row["code"]
        for trial in range(3):
            try:
                receipt=prep.convert(row,False)
                print(code,receipt["status"],receipt.get("reason"),flush=True)
                break
            except Exception as err:
                print(code,trial,type(err).__name__,flush=True)
                if trial==2:write(prep.RECEIPTS/(code+".pending.json"),dict(error=type(err).__name__))
                else:time.sleep(3)


def seal_references():
    refs=[]
    excluded=Counter()
    for row in read(CANDIDATE):
        code=row["code"]
        p=DEST/"receipts"/(code+".json")
        if not p.is_file():
            assert p.with_suffix(".pending.json").is_file()
            excluded["publisher_transfer"]+=1
            continue
        d=read(p)
        if d["status"]!="eligible":
            excluded[d["reason"]]+=1
            continue
        if max(d["bounding_mm"])>36 or d["components"]!=1:
            excluded["reference_geometry"]+=1
            continue
        disk=STORAGE/"arrays"/(code+".npz")
        assert sha(disk)==d["sha"]
        if not usable_lungs(disk):
            excluded["lung_field"]+=1
            continue
        refs.append(dict(row,path=str(disk),sha=d["sha"]))
    seal(LOCK,refs)
    seal(DEST/"receipt.json",dict(references=len(refs),excluded=dict(excluded),manifest=sha(LOCK),cohort=sha(DEST/"design.json")))
    print(len(refs),dict(excluded),flush=True)


if __name__=="__main__":
    os.umask(0o077)
    p=ArgumentParser()
    p.add_argument("stage",choices=("screen","fetch","seal"))
    p.add_argument("--shard",type=int,default=0)
    p.add_argument("--shards",type=int,default=8)
    a=p.parse_args()
    if a.stage=="screen":screen()
    elif a.stage=="fetch":fetch(a.shard,a.shards)
    else:seal_references()
