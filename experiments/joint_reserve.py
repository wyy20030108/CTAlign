"""Build a genuinely new reference cohort without looking at segmentation predictions."""
from argparse import ArgumentParser
from collections import Counter, defaultdict
from hashlib import sha256
from pathlib import Path
import csv
import os
import re
import time

import requests
from huggingface_hub import get_token, hf_hub_url

import atelectasis_prepare as prep
import thesis_transfer as fetcher
from atelectasis_screen import CHALLENGE, FIBROSIS, PATTERN, QUESTIONABLE, REX, TYPE, anatomy
from audit_pairs import identity, split_names
from diag_localize import seal
from fetch_manifest import FILES
from frozen_priors import read, sha, write
from joint_roi import ROOT
from replay_controls import read_reports
from swap_data import norm, side
from thesis_cohort import MANIFEST
from thesis_expand_screen import REPORT_BAD, SPLIT, select_span


RESERVE = ROOT / "reserve"
CANDIDATES = RESERVE / "candidate.private.json"


def exposed_patients():
    result = set()
    for rows in read(MANIFEST).values():
        if isinstance(rows, list):
            result.update(tuple(r["patient"].split(":")) for r in rows)
    inputs = [Path("phase7_atelectasis/screen.private.json"), Path("phase8_thesis/candidate_private.json"),
              *Path("phase3_swap/fallback").glob("*.private.json"),
              *Path("phase1/data").glob("*/manifest.private.json")]
    for path in inputs:
        rows = read(path)
        if isinstance(rows, dict):
            rows = rows.get("rows", [])
        result.update(identity(r["name"])[1] for r in rows if r.get("name"))
    for path in Path("phase1/cohort/split_v1").rglob("*.csv"):
        with path.open() as f:
            result.update(identity(r["volume"])[1] for r in csv.DictReader(f) if r.get("volume"))
    return result


def raw_rule(row, channel, q):
    return (row["categories"].get(channel)=="2a" and row["entity_counts"].get(channel)==1
            and row["pixels"].get(channel,0)>0 and TYPE.search(q) and side(q)
            and len(anatomy(q)[0])==1 and not QUESTIONABLE.search(q) and not FIBROSIS.search(q))


def primary_span(q, report):
    original = report["Findings_EN"]
    primary = select_span(q, report) if PATTERN.search(q) else None
    if primary:
        return "exact_linear",primary[1],primary[2]
    lobes, segments = anatomy(q)
    laterality = side(q)
    matching = {}
    for sentence in re.split(r"(?<=[.!?])\s+|\n+", original):
        sentence = sentence.strip()
        if not TYPE.search(sentence) or REPORT_BAD.search(sentence) or FIBROSIS.search(sentence):
            continue
        for clause in SPLIT.split(sentence):
            clause = clause.strip()
            region, present = anatomy(clause)
            if not TYPE.search(clause) or QUESTIONABLE.search(clause) or side(clause)!=laterality:
                continue
            if region!=lobes or not segments.issubset(present) or len(present-segments)>1:
                continue
            # Require the same shape family whenever the reference specifies linear/band morphology.
            if PATTERN.search(q) and not PATTERN.search(clause):
                continue
            matching[norm(clause)] = (clause, sentence)
    if len(matching)==1:
        clause, sentence = next(iter(matching.values()))
        kind = "exact_linear" if PATTERN.search(q) else "additional_focal"
        return kind,clause,sentence
    return None


def screen():
    RESERVE.mkdir(parents=True,exist_ok=True)
    known = exposed_patients()
    base, challenge = read(REX), read(CHALLENGE)
    hidden = {identity(r["name"])[1] for d in (base,challenge) for r in d["test"]}
    owned = defaultdict(list)
    potential = []
    supply = (("official_validation",challenge["val"]),("unseen_release_train",base["train"]))
    for source,rows in supply:
        for row in rows:
            name, patient, _ = identity(row["name"])
            if patient in known or patient in hidden:
                continue
            for channel,q in row["findings"].items():
                if raw_rule(row,channel,q):
                    potential.append((source, row, int(channel), q, name, patient))
    texts = read_reports({v[4] for v in potential})
    for source,row,k,q,name,patient in potential:
        match = primary_span(q,texts[name])
        if not match:
            continue
        kind, query, ctx = match
        assert query in ctx and ctx in texts[name]["Findings_EN"]
        owned[patient].append(dict(name=name,channel=k,query=query,context=ctx,published=q,
                                   native_count=row["pixels"][str(k)],side=side(q),source=source,kind=kind))

    def rank(r):
        digest = sha256(("joint-reserve-v1|"+r["name"]+"|"+str(r["channel"])).encode()).hexdigest()
        return (r["source"]!="official_validation", r["kind"]!="exact_linear", digest)

    chosen = sorted((min(rows,key=rank) for rows in owned.values()),key=rank)
    for index,r in enumerate(chosen,1):
        r["code"] = f"J{index:03d}"
    seal(CANDIDATES,chosen)
    counts = {s:dict(Counter(r["kind"] for r in chosen if r["source"]==s)) for s,_ in supply}
    seal(RESERVE/"census.json",dict(candidates=counts,manifest=sha(CANDIDATES),original_cohort=sha(MANIFEST),
                                    excluded_history=len(known),scope="no prediction scores; unique patients"))
    print(counts,flush=True)


def verify_paths(index):
    sample = next(r for r in read(CANDIDATES) if r["code"]==index)
    repo,revision,_ = FILES["rex"]
    headers = {"Authorization":"Bearer "+get_token()}
    for folder in ("segmentations", "challenge_segmentations", "MICCAI_segmentations"):
        target = hf_hub_url(repo,folder+"/"+sample["name"],repo_type="dataset",revision=revision)
        try:
            response=requests.head(target,headers=headers,timeout=(8,8),allow_redirects=False)
            print(sample["code"],folder,response.status_code,flush=True)
        except requests.RequestException as exc:
            print(sample["code"],folder,type(exc).__name__,flush=True)


def fetch(shard, shards, group, kind, folder):
    prep.ROOT = RESERVE
    prep.DATA = RESERVE / "arrays"
    prep.RECEIPTS = RESERVE / "receipts"
    fetcher.ROOT = RESERVE
    for p in (prep.DATA,prep.RECEIPTS):
        p.mkdir(exist_ok=True)
    def retrieve(repo, revision, rel, temp):
        if rel.startswith("segmentations/"):
            rel=folder+"/"+rel.split("/",1)[1]
        return fetcher.download(repo,revision,rel,temp)
    prep.obtain_file = retrieve
    cases = [r for r in read(CANDIDATES) if r["source"]==group and r["kind"]==kind]
    for row in cases[shard::shards]:
        for trial in range(3):
            try:
                out = prep.convert(row,False)
                print(row["code"],out["status"],out.get("reason"),flush=True)
                break
            except Exception as e:
                print(row["code"],trial,type(e).__name__,flush=True)
                if trial==2:
                    write(prep.RECEIPTS/(row["code"]+".pending.json"),dict(code=row["code"],error=type(e).__name__))
                else:
                    time.sleep(2)


if __name__ == "__main__":
    os.umask(0o077)
    parser=ArgumentParser()
    parser.add_argument("stage",choices=("screen","paths","fetch"))
    parser.add_argument("--index",default="J001")
    parser.add_argument("--group",choices=("official_validation","unseen_release_train"),default="official_validation")
    parser.add_argument("--kind",choices=("exact_linear","additional_focal"),default="exact_linear")
    parser.add_argument("--folder",default="segmentations")
    parser.add_argument("--shard",type=int,default=0)
    parser.add_argument("--shards",type=int,default=1)
    a=parser.parse_args()
    if a.stage=="screen":screen()
    elif a.stage=="paths":verify_paths(a.index)
    else:fetch(a.shard,a.shards,a.group,a.kind,a.folder)
