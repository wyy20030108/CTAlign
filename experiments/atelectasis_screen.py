"""Select ReX linear/band atelectasis with a traceable, target-specific CT-RATE passage."""
import json, os, re
from collections import Counter, defaultdict
from pathlib import Path
from audit_pairs import identity, run, split_names
from frozen_priors import sha, write
from replay_controls import read_reports
from swap_data import norm, side

ROOT=Path("phase7_atelectasis")
REX=Path("metadata/rex/dataset.json")
CHALLENGE=REX.with_name("MICCAI_challenge_dataset.json")
REPORTS=list(sorted(Path("metadata/ct_rate/dataset/radiology_text_reports").glob("*reports.csv")))
PATTERN=re.compile(r"\b(?:linear|band(?:[- ]?like)?|strip(?:e|like)?|plate[- ]like|discoid|laminar)\b",re.I)
TYPE=re.compile(r"\batelecta\w*\b",re.I)
QUESTIONABLE=re.compile(r"\b(?:left and right|right and left|both lungs|both lobes|bilateral\w*|diffuse\w*|widespread|throughout|scattered|total\w*|complet\w*|entire|lobar collapse|lingular and|more prominent|except|cannot exclude|suspect\w*|possible|previously|no evidence of|no atelecta\w*|resolved)\b",re.I)
FIBROSIS=re.compile(r"\b(?:fibroatelecta\w*|fibrotic\w*|fibrosis|scarring)\b",re.I)
LOBES={"upper":r"\bupper (?:lung )?lobe\b","middle":r"\bmiddle (?:lung )?lobe\b","lower":r"\blower (?:lung )?lobe\b"}
SEG={"medial":r"\bmedial\b","lateral":r"\blateral\b","posterior":r"\b(?:posterior|postero)\b","posterobasal":r"\bpostero\s*basal\b",
     "laterobasal":r"\blatero\s*basal\b","anterobasal":r"\bantero\s*basal\b","apical":r"\bapic\w*\b","superior":r"\bsuperior\b"}
def anatomy(txt):
    lower=norm(txt); lobes=[k for k,p in LOBES.items() if re.search(p,lower)]
    if re.search(r"\blingul\w*\b",lower):
        lobes=[k for k in lobes if k!="upper"]+["lingular"]
    return set(lobes),{k for k,p in SEG.items() if re.search(p,lower)}
def consistent(q,s):
    if not TYPE.search(s) or not PATTERN.search(s) or QUESTIONABLE.search(s) or FIBROSIS.search(s):return False
    if side(q)!=side(s):return False
    ql,qs=anatomy(q);sl,ss=anatomy(s)
    if len(sl)!=1 or (ql and ql!=sl) or (qs and not qs.issubset(ss)):return False
    if len(ss-qs)>1 or re.search(r"\b(?:segments|lobes)\b|\b(?:in|of)\s+the\s+other\b",s,re.I):return False
    return True
def passage(q, report):
    text=report.get("Findings_EN","")
    sentence=[s.strip() for s in re.split(r"(?<=[.!?])\s+|\n+",text) if consistent(q,s)]
    if len({norm(s) for s in sentence})==1:
        return sentence[0]
    return None
def screen():
    os.umask(0o077);ROOT.mkdir(exist_ok=True)
    allowed=[];audit=run(REX,REPORTS,[CHALLENGE],matched_names=allowed)
    assert audit["report_matched_rex_train"]["unique_patients"]>=2600
    allowed=set(allowed)
    document=json.loads(REX.read_text()); shortlist=[]; counts=Counter()
    for row in document["train"]:
        name=row["name"]
        if name not in allowed:continue
        for key,q in row["findings"].items():
            if not TYPE.search(q) or not PATTERN.search(q):continue
            counts["lexical"]+=1
            if row["categories"].get(key)!="2a" or row["entity_counts"].get(key)!=1 or row["pixels"].get(key,0)<=0:continue
            counts["single_entity_category"]+=1
            if not side(q) or QUESTIONABLE.search(q) or FIBROSIS.search(q) or len(anatomy(q)[0])>1:continue
            counts["focal_text"]+=1
            shortlist.append((name,key,q,row["pixels"][key]))
    reports=read_reports({r[0] for r in shortlist})
    grouped=defaultdict(list)
    for name,key,q,pixels in shortlist:
        query=passage(q,reports[name])
        if query:
            n,patient,_=identity(name)
            grouped[patient].append(dict(name=n,channel=int(key),query=query,published=q,native_count=pixels,side=side(q)))
            counts["matching_source_passage"]+=1
    import hashlib
    def priority(row):
        return hashlib.sha256(("linear-atelectasis-891|"+row["name"]+"|"+str(row["channel"])).encode()).hexdigest()
    chosen=[min(entries,key=priority) for entries in grouped.values()]
    chosen.sort(key=priority)
    import csv
    prior=set()
    # Previous allocation includes all prior in-project calibrated and development groups.
    for p in Path("phase1/cohort/split_v1/b20").glob("*.csv"):
        if p.stem not in {"T","R1","R2"}:continue
        with p.open() as f:prior.update(identity(r["volume"])[1] for r in csv.DictReader(f) if r.get("volume"))
    for n in ("fallback_train","fallback_val","swap_dev","unlabelled_train"):
        p=Path("phase3_swap/fallback")/(n+".private.json")
        prior.update(identity(r["name"])[1] for r in json.loads(p.read_text()))
    for i,row in enumerate(chosen,1):
        row.update(code=f"L{i:03}",prior=identity(row["name"])[1] in prior)
    write(ROOT/"screen.private.json",chosen)
    write(ROOT/"screen.json",dict(counts=dict(counts),selected_unique_patients=len(chosen),
        previously_allocated=sum(r["prior"] for r in chosen),source_hashes={str(p):sha(p) for p in [REX,CHALLENGE,*REPORTS]},
        selection="Release train only, both official holdouts excluded by patient; single published entity in category 2a, focal unilateral explicit linear/band wording, one traceable and compatible Findings_EN sentence, one examination/channel per patient."))
    print((ROOT/"screen.json").read_text(),flush=True)
if __name__=="__main__":screen()

