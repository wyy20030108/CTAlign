"""A separate report-stratified U training cohort after the natural eight had no affirmative reports."""
import collections
import csv
import hashlib
import json
import os
import tempfile
from pathlib import Path
from fetch_four import write_json,sha256
from fetch_manifest import FILES
from prepare_unlabeled import obtain_file,release_path,convert_image


def main():
    os.umask(0o077)
    out=Path("phase1/data/U_report_strata32")
    out.mkdir(parents=True,exist_ok=True)
    natural={r["name"] for r in json.loads(Path("phase1/data/U/manifest.private.json").read_text())["rows"]}
    roster=Path("phase1/cohort/split_v1/b10/U_train.csv")
    with roster.open(newline="") as f: permitted={r["volume"]:r["patient"] for r in csv.DictReader(f)}
    frozen=Path("phase1/donor_support/training_donors.private.json")
    rows=json.loads(frozen.read_text())
    if {r["volume"] for r in rows}!=set(permitted):
        raise ValueError("Existing textual strata do not match the U-only roster")
    controls=collections.Counter(r["state"] for r in rows if r["volume"] in natural)
    if sum(controls.values())!=8: raise ValueError("Natural roster is incomplete")
    excludes={Path(s.strip()).name for p in Path("data/four_train/ct_rate/dataset/metadata").glob("no_chest_*.txt")
              for s in p.read_text().splitlines()}
    strata={"affirmed":16,"negated":8,"uncertain":4,"unmentioned":4}
    chosen=[]
    counts={}
    for state,n in strata.items():
        group=sorted((a["volume"] for a in rows if a["state"]==state and a["volume"] not in natural|excludes),
                     key=lambda k:hashlib.sha256(f"training-U-positive-support-v1:891:{k}".encode()).digest())
        if len(group)<n: raise ValueError("Not enough patients in predefined stratum")
        counts[state]={"eligible":len(group),"selected":n,"inclusion_probability":n/len(group)}
        chosen.extend((k,state) for k in group[:n])
    if len(set(permitted[k] for k,s in chosen))!=32: raise ValueError("Duplicate patients in U subset")
    protocol={"seed":891,"training_only":True,"natural8_report_states":dict(controls),"state_sampling":counts,
        "natural_U_reference":2143,"selection":"Hash rank within frozen report stratum after removal of natural eight and published non-chest IDs. Conflicting and historical wording not sampled in this enriched pilot. State is report wording.",
        "use":"Separate mechanism-focused training cohort, not an estimate of population prevalence. All comparisons use same image list. No reference/test/masks read.",
        "split_sha256":sha256(roster),"strata_sha256":sha256(frozen)}
    keep=out/"plan.json"
    if keep.exists() and json.loads(keep.read_text())!=protocol:raise ValueError("Frozen selection changed")
    write_json(keep,protocol)
    selection=[{"name":k,"stratum":s} for k,s in chosen]
    lock=out/"selection.private.json"
    if lock.exists() and json.loads(lock.read_text())!=selection: raise ValueError("Frozen names changed")
    write_json(lock,selection)
    repo,rev,_=FILES["ct_rate"]
    results=[]
    for i,(name,state) in enumerate(chosen,1):
        item=out/(name.removesuffix(".nii.gz")+".npz")
        receipt=item.with_suffix(".json")
        if receipt.exists():
            row=json.loads(receipt.read_text())
            if row["name"]!=name or row["npz_sha256"]!=sha256(item) or row["revision"]!=rev:
                raise ValueError("Invalid cached scan")
        else:
            with tempfile.TemporaryDirectory(dir=out,prefix="stream_") as td:
                ct,source_sha=obtain_file(repo,rev,release_path(name),td)
                compact=Path(td)/"compact.npz"
                shape=convert_image(ct,compact)
                compact.replace(item)
            row={"name":name,"file":str(item),"shape":shape,"source_sha256":source_sha,"npz_sha256":sha256(item),
                 "revision":rev,"stratum":state}
            write_json(receipt,row)
        results.append(row)
        print("U report-stratified",i,flush=True)
    write_json(out/"manifest.private.json",{"role":"U_train","rows":results,"split_sha256":sha256(roster)})
    write_json(out/"receipt.json",{"patients":len(results),"U_masks_accessed":0,
        "npz_bytes":sum(Path(r["file"]).stat().st_size for r in results),"source_plan_sha256":sha256(keep)})


if __name__=="__main__":
    main()
