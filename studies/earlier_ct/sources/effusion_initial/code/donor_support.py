"""Freeze report donor support in the U training stream; never open reference or audit reports."""
import csv
import hashlib
import math
import os
import re
from collections import Counter
from pathlib import Path
from fetch_four import sha256, write_json
from replay_controls import read_reports, word_count
from real_pair_smoke import text_ids
from report_strata import describe_report, VERSION


def describe(row):
    full=describe_report(row["Findings_EN"],row["Impressions_EN"])
    clipped=[]
    for field in ("Findings_EN","Impressions_EN"):
        pieces=re.findall(r"[A-Za-z]+|[0-9]+(?:\.[0-9]+)?|[^\w\s]",row[field].lower())
        clipped.append(" ".join(pieces[:31]))
    shortened=describe_report(*clipped)
    full_text=" ".join(row.values()).casefold()
    # A vocabulary-level sensitivity check only, as explicit negations also mention the word.
    context=tuple(bool(re.search(pattern,full_text)) for pattern in (r"\batelecta",r"\bcardiomegal"))
    return {**full,"context":context,"text_length":word_count(row),"encoded_prefix":tuple(text_ids(row)[0]),
            "short_state":shortened["state"]}


def main():
    os.umask(0o077)
    roster=Path("phase1/cohort/split_v1/b10/U_train.csv")
    with roster.open(newline="") as f:
        patient={r["volume"]:r["patient"] for r in csv.DictReader(f)}
    if len(patient)!=len(set(patient.values())):
        raise ValueError("Donor table requires one selected exam per patient")
    rows={n:describe(r) for n,r in read_reports(set(patient)).items()}
    states=Counter(v["state"] for v in rows.values())
    unchanged=Counter(v["state"] for v in rows.values() if v["state"]==v["short_state"])
    out=Path("phase1/donor_support")
    out.mkdir(parents=True,exist_ok=True)
    assignments=[]
    supported=Counter()
    for n,case in sorted(rows.items()):
        tiers={"coarse":[], "side_size":[], "side_size_context":[], "affirmative_both_descriptors_known":[]}
        for k,other in rows.items():
            if (patient[n]==patient[k] or other["state"]!=case["state"] or case["text_length"]<=0 or
                    other["text_length"]<=0 or abs(math.log(case["text_length"]/other["text_length"]))>0.25
                    or case["encoded_prefix"]==other["encoded_prefix"]):
                continue
            tiers["coarse"].append(k)
            if (other["side"],other["size"])!=(case["side"],case["size"]):
                continue
            tiers["side_size"].append(k)
            if case["context"]==other["context"]:
                tiers["side_size_context"].append(k)
            if (case["state"]=="affirmed" and case["side"] in ("left","right","bilateral")
                    and case["size"] in ("small","moderate","large")):
                tiers["affirmative_both_descriptors_known"].append(k)
        assignment={"volume":n,"state":case["state"],"tiers":{}}
        for tier,matches in tiers.items():
            ranked=sorted(matches,key=lambda other:hashlib.sha256(f"donor:891:{n}:{other}".encode()).digest())
            assignment["tiers"][tier]={"n_available":len(matches),"selected":ranked[:3]}
            for required in (1,2,3):
                if len(matches)>=required:
                    supported[tier,case["state"],required]+=1
        assignments.append(assignment)
    write_json(out/"training_donors.private.json",assignments)
    def rate(tier,condition=None):
        subset=states.keys() if condition is None else [condition]
        return {"denominator":sum(states[s] for s in subset),
                "at_least_1":sum(supported[tier,s,1] for s in subset),
                "at_least_3":sum(supported[tier,s,3] for s in subset)}
    write_json(out/"summary.json",{"training_U_patients":len(rows),"source_sha256":sha256(roster),
        "assignment_sha256":sha256(out/"training_donors.private.json"),"report_rule":VERSION,
        "case_state_denominators":dict(states),
        "full_report_state_preserved_by_pilot_31_token_section_prefix":{k:unchanged[k] for k in states},
        "all_U":{tier:rate(tier) for tier in ("coarse","side_size","side_size_context")},
        "affirmative_U":{tier:rate(tier,"affirmed") for tier in ("coarse","side_size","side_size_context","affirmative_both_descriptors_known")},
        "coarse_by_state":{s:rate("coarse",s) for s in states},
        "matching":"Distinct patient, same CT-RATE noncontrast chest protocol, identical report-derived state, absolute log word-count ratio <=0.25. Excludes pair with identical encoded pilot prefix.",
        "fine":"Exact side/size tokens; two unknown descriptors can match. Last affirmative tier requires informative, specified side and size for both. Context tier also matches mention indicators for atelectasis and cardiomegaly.",
        "scope":"Support and report-truncation diagnostics in U training only. No CT, annotations, outcomes or independent reference reports used."})
    print("matched training patients",len(rows))


if __name__=="__main__":
    main()
