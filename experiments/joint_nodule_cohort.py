"""Prepare a second CT/report phenotype with prospective patient-level separation."""
from argparse import ArgumentParser
from collections import Counter, defaultdict
import csv
import os
from pathlib import Path
import re
import time

import atelectasis_prepare as prep
import thesis_report_only as unlabelled
import thesis_transfer as transfer
from atelectasis_screen import CHALLENGE, QUESTIONABLE, REPORTS, REX, anatomy
from audit_pairs import identity, split_names
from diag_localize import seal
from frozen_priors import read, sha, write
from joint_reserve import exposed_patients
from joint_roi import ROOT
from replay_controls import read_reports
from swap_data import norm, side
from thesis_cohort import choose_stratified, site
from thesis_expand_screen import REPORT_BAD, SPLIT


DEST = ROOT / "nodule_v2"
CASE = DEST / "candidate.private.json"
REGISTRY = DEST / "cohort.private.json"
NODULE = re.compile(r"\bnodule\b|\bnodular (?:lesion|density)\b", re.I)
EXCLUDE = re.compile(r"\b(?:multiple nodules|numerous nodules|scattered nodules|nodular consolidation|no nodule\w*|no pulmonary nodule\w*)\b", re.I)


def focal(text):
    lobes, _ = anatomy(text)
    if not (NODULE.search(text) and side(text) and len(lobes)==1):
        return False
    if EXCLUDE.search(text) or QUESTIONABLE.search(text) or REPORT_BAD.search(text):
        return False
    return True


def sections(text):
    for sentence in re.split(r"(?<=[.!?])\s+|\n+",text):
        sentence=sentence.strip()
        if not (NODULE.search(sentence) and not REPORT_BAD.search(sentence)):
            continue
        if focal(sentence):
            yield sentence, sentence
        else:
            for part in SPLIT.split(sentence):
                part=part.strip()
                if focal(part):
                    yield part,sentence


def query(q,text):
    reg, seg=anatomy(q)
    matches={}
    for part, sentence in sections(text):
        lobe, locations=anatomy(part)
        if lobe==reg and side(part)==side(q) and (not seg or not locations or bool(seg & locations)):
            matches[norm(part)]=(part,sentence)
    if len(matches)==1:
        return next(iter(matches.values()))
    return None


def screen():
    DEST.mkdir(parents=True,exist_ok=True)
    history=exposed_patients()
    former_test={tuple(r["patient"].split(":")) for r in read(Path("phase8_thesis/thesis_private.json"))["test"]}
    base,canonical=read(REX),read(CHALLENGE)
    reserved={identity(r["name"])[1] for d in (base,canonical) for k in ("val","test") for r in d[k]}
    groups={"official":canonical["val"],"available":base["train"]}
    candidate=[]
    for group,rows in groups.items():
        for row in rows:
            name,patient,_=identity(row["name"])
            if (group=="official" and patient in history) or (group=="available" and patient in reserved|former_test):
                continue
            for k,q in row["findings"].items():
                if focal(q) and row["entity_counts"][k]==1 and row["pixels"][k]>0:
                    candidate.append((name,patient,group,int(k),q,row["pixels"][k]))
    reports=read_reports({r[0] for r in candidate})
    matched=defaultdict(dict)
    for name,patient,group,channel,published,native_count in candidate:
        q=query(published,reports[name]["Findings_EN"])
        if not q:
            continue
        part,context=q
        assert part in context and context in reports[name]["Findings_EN"]
        item=dict(name=name,patient=":".join(patient),query=part,context=context,published=published,
                  native_count=native_count,channel=channel,side=side(part),site=site({"query":part}),
                  code=f"{group}-{name}-{channel}")
        before=matched[group].get(patient)
        if before is None or item["code"]<before["code"]:
            matched[group][patient]=item
    potential=dict((k,len(v)) for k,v in matched.items())
    all_labeled=list(matched["available"].values())
    training=choose_stratified(all_labeled,min(23,len(all_labeled)//2),"nodule-labelled-v1")
    seen={r["patient"] for r in training}
    candidates=[r for r in all_labeled if r["patient"] not in seen]
    development=choose_stratified(candidates,min(18,len(candidates)),"nodule-development-v1")
    official=sorted(matched["official"].values(),key=lambda r:r["code"])
    data={"train":training,"dev":development,"reserved":official}
    for idx,(role,rows) in enumerate(data.items()):
        for j,row in enumerate(rows,1):
            row.update(code=f"X{'TDR'[idx]}{j:03d}",role=role)
    seal(CASE,data)
    history.update(tuple(r["patient"].split(":")) for rows in data.values() for r in rows)
    for path in (REX,CHALLENGE):
        history.update(identity(name)[1] for names in split_names(path).values() for name in names)
    entries=defaultdict(list)
    with next(p for p in REPORTS if p.name=="train_reports.csv").open(encoding="utf-8-sig",newline="") as f:
        for row in csv.DictReader(f):
            name,patient,_=identity(row["VolumeName"])
            if patient in history or not name.endswith("_1.nii.gz"):
                continue
            found={norm(q):q for q,_ in sections(row["Findings_EN"] or "")}
            if len(found)==1:
                q=next(iter(found.values()))
                entries[patient].append(dict(name=name,query=q,site=site({"query":q}),code=name))
    pool=[min(rows,key=lambda r:r["name"]) for rows in entries.values()]
    selected=choose_stratified(pool,min(56,len(pool)),"nodule-unlabelled-v1")
    for j,row in enumerate(selected,1):
        row.update(code=f"XU{j:03d}")
    loc=DEST / "report_only"
    loc.mkdir(exist_ok=True)
    seal(loc/"source.private.json", selected)
    summary=dict(candidates={k:len(v) for k,v in data.items()},report_only=len(selected),available=potential,
                 labelled_lock=sha(CASE),unlabelled_lock=sha(loc/"source.private.json"),
                 source={str(p):sha(p) for p in (REX,CHALLENGE,*REPORTS)},
                 finding="single traceable unilateral pulmonary nodule; source and resampling audit prospectively fixed")
    seal(DEST/"census.json",summary)
    print(summary["candidates"],summary["report_only"],summary["available"],flush=True)


def fetch(role,shard,shards):
    transfer.ROOT=DEST
    if role=="unlabelled":
        unlabelled.U=DEST/"report_only"
        unlabelled.ROOT=DEST
        return unlabelled.fetch(shard,shards)
    prep.ROOT=DEST
    prep.DATA=DEST/"arrays"
    prep.RECEIPTS=DEST/"receipts"
    prep.obtain_file=transfer.download
    for p in (prep.DATA,prep.RECEIPTS):p.mkdir(exist_ok=True)
    for row in read(CASE)[role][shard::shards]:
        for attempt in range(3):
            try:
                r=prep.convert(row,False)
                print(role,row["code"],r["status"],r.get("reason"),flush=True)
                break
            except Exception as error:
                print(role,row["code"],type(error).__name__,flush=True)
                if attempt==2:write(prep.RECEIPTS/(row["code"]+".pending.json"),dict(code=row["code"],error=type(error).__name__))
                else:time.sleep(3)


if __name__=="__main__":
    os.umask(0o077)
    p=ArgumentParser()
    p.add_argument("stage",choices=("screen","fetch"))
    p.add_argument("--role",choices=("train","dev","reserved","unlabelled"),default="reserved")
    p.add_argument("--shard",type=int,default=0)
    p.add_argument("--shards",type=int,default=1)
    cli=p.parse_args()
    if cli.stage=="screen":screen()
    else:fetch(cli.role,cli.shard,cli.shards)
