"""Expand the exact-source cohort with explicitly selected report spans."""
from collections import Counter,defaultdict
from hashlib import sha256
from pathlib import Path
import os,re,csv,json

from atelectasis_screen import (ROOT as PREVIOUS,REX,CHALLENGE,REPORTS,PATTERN,TYPE,
                                QUESTIONABLE,FIBROSIS,anatomy,passage,norm,side)
from audit_pairs import run,identity
from replay_controls import read_reports
from frozen_priors import read,write,sha
from diag_localize import seal

ROOT=Path("phase8_thesis")
SOURCES=ROOT/"candidate_private.json"
REPORT_BAD=re.compile(r"\b(?:no evidence|no (?:linear|band|atelecta\w*)|cannot exclude|possible|suspected|previously|resolved|history of)\b",re.I)
SPLIT=re.compile(r"\s*(?:[,;]|\band\b|\bas well as\b)\s*",re.I)

def select_span(q, report):
    direct=passage(q,report)
    if direct:
        return "single_sentence",direct,direct
    chosen=[]
    qlob,qseg=anatomy(q)
    if len(qlob)!=1 or not side(q):return None
    for sentence in re.split(r"(?<=[.!?])\s+|\n+",report.get("Findings_EN","")):
        if not (TYPE.search(sentence) and PATTERN.search(sentence)) or REPORT_BAD.search(sentence):
            continue
        spans=SPLIT.split(sentence.strip())
        sites=[(span,side(span),*anatomy(span)) for span in spans if span.strip()]
        present=[s for s in sites if s[1] and s[2]]
        if len(present)<2:continue
        for span,loc,lob,seg in present:
            if loc!=side(q) or lob!=qlob or not qseg.issubset(seg):continue
            if QUESTIONABLE.search(span) or FIBROSIS.search(span):continue
            if sum(x[1]==loc and x[2]==lob and qseg.issubset(x[3]) for x in present)!=1:continue
            different=any(x[1]!=loc or x[2]!=lob or x[3]!=seg for x in present)
            if not different:continue
            chosen.append((span.strip(),sentence.strip()))
    if len({(norm(a),norm(b)) for a,b in chosen})==1:
        return "target_span_in_multisite_sentence",*chosen[0]
    return None

def screen():
    os.umask(0o077);ROOT.mkdir(exist_ok=True)
    prior_phase=read(PREVIOUS/"screen.private.json")
    existing=read(PREVIOUS/"pilot_private.json")
    used={identity(r["name"])[1] for r in prior_phase}
    allowed=[]
    run(REX,REPORTS,[CHALLENGE],matched_names=allowed)
    allow=set(allowed)
    rex=read(REX);cases={r["name"]:r for r in rex["train"] if r["name"] in allow}
    raw=[(n,k,v) for n,r in cases.items()
         for k,v in r["findings"].items()
         if TYPE.search(v) and PATTERN.search(v) and r["categories"].get(k)=="2a"
         and r["entity_counts"].get(k)==1 and (r["pixels"].get(k) or 0)>0 and side(v)
         and not QUESTIONABLE.search(v) and not FIBROSIS.search(v)]
    reports=read_reports({n for n,_,_ in raw})
    defined=defaultdict(list)
    counters=Counter()
    for n,k,q in raw:
        p=identity(n)[1]
        if p in used:
            counters["removed_previous_48"]+=1;continue
        span=select_span(q,reports[n])
        if not span:continue
        form,query,context=span
        assert query in context and context in reports[n]["Findings_EN"]
        defined[p].append(dict(name=n,channel=int(k),query=query,context=context,matched_type=form,
            published=q,native_count=cases[n]["pixels"][k],side=side(q)))
        counters[form]+=1
    def rank(r):return sha256(("thesis-expand-1|"+r["name"]+":"+str(r["channel"])).encode()).hexdigest()
    chosen=[min(v,key=lambda r:(r["matched_type"]!="single_sentence",rank(r))) for v in defined.values()]
    chosen.sort(key=rank)
    historical=set(used)
    for p in Path("phase1/cohort/split_v1").rglob("T.csv"):
        with p.open() as f:historical.update(identity(r["volume"])[1] for r in csv.DictReader(f) if r.get("volume"))
    for p in Path("phase1/data").glob("*/manifest.private.json"):
        d=read(p)
        for r in d.get("rows",[]) if isinstance(d,dict) else d:
            if r.get("name"):historical.add(identity(r["name"])[1])
    for group in ("fallback_train","fallback_val","swap_dev","unlabelled_train"):
        file=Path("phase3_swap/fallback")/(group+".private.json")
        historical.update(identity(r["name"])[1] for r in read(file))
    for i,row in enumerate(chosen,1):
        row["code"]=f"N{i:03d}";row["prior_exposure"]=identity(row["name"])[1] in historical
    seal(SOURCES,chosen)
    overview=dict(release_train=len(cases),strict_published_findings=len(raw),additions=len(chosen),
        strata=dict(Counter(r["matched_type"] for r in chosen)),already_exposed=sum(r["prior_exposure"] for r in chosen),
        earlier_splits={k:len(v) for k,v in existing.items()},
        references={str(p):sha(p) for p in (REX,CHALLENGE,*REPORTS)},candidate_lock=sha(SOURCES),counts=dict(counters),
        extraction="Same target phenotype, one released entity, one patient. A marker identifies the exact contiguous target span of a report; source sentence retained. Coordinate prompts and report-generated masks are absent.")
    seal(ROOT/"screen.json",overview)
    print(json.dumps(overview,indent=2))
if __name__=="__main__":screen()

