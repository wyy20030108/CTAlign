"""Preselect a patient-random test sampling frame before reading any outcomes."""
import csv
import hashlib
import json
import os
from collections import defaultdict
from pathlib import Path
from audit_pairs import identity, split_names
from fetch_four import sha256, write_json


def rank(domain, identifier):
    return hashlib.sha256(f"test-frame-v1:{domain}:{identifier}".encode()).digest()


def main():
    os.umask(0o077)
    dest=Path("phase1/cohort/independent_frame")
    dest.mkdir(mode=0o700, parents=True, exist_ok=True)
    rex_files=[Path("metadata/rex/dataset.json"), Path("metadata/rex/MICCAI_challenge_dataset.json")]
    excluded=set()
    for manifest in rex_files:
        for partition in split_names(manifest).values():
            excluded.update(identity(n)[1] for n in partition)
    sources=Path("data/four_train/ct_rate/dataset/metadata")
    nonchest={Path(s.strip()).name for s in (sources/"no_chest_train.txt").read_text().splitlines()}
    volumes=defaultdict(set)
    with (sources/"train_metadata.csv").open(encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            name,pid,exam=identity(row["VolumeName"])
            if pid[0] != "train":
                raise ValueError("Unexpected official validation volume in source train metadata")
            if pid not in excluded and name not in nonchest:
                volumes[pid].add(name)
    ordered=sorted(volumes,key=lambda pid:rank("sampling frame","_".join(pid)))
    # This frame is permanently excluded from all training streams. Endpoint panel size
    # will be determined from development precision, before choosing its ordered prefix.
    withheld=ordered[:len(ordered)//4]
    if not withheld or set(withheld) & excluded:
        raise ValueError("Invalid independent sampling frame")
    records=[]
    for pid in withheld:
        # First select an exam, then a reconstruction, so multiple reconstructions
        # cannot give an examination more chances to be selected.
        exams=defaultdict(list)
        for n in volumes[pid]:
            exams[identity(n)[2]].append(n)
        selected=min(exams,key=lambda e:rank("exam","_".join(e)))
        n=min(exams[selected],key=lambda key:rank("reconstruction",key))
        records.append({"patient":"_".join(pid),"volume":n})
    private=dest/"sealed_frame.private.json"
    if private.exists():
        if json.loads(private.read_text()) != records:
            raise ValueError("Sampling frame was previously generated with different upstream data")
    else:
        write_json(private,records)
    write_json(dest/"receipt.json",{
        "design":"All original and challenge ReX patients excluded across every split, including training. CT-RATE source validation namespace also excluded. Known non-chest scans excluded before selecting one examination/reconstruction per patient.",
        "eligible_source_patients":len(ordered),"reserved_patients":len(withheld),
        "sample_fraction":0.25,"sampling":"Fixed hash order, first floor(25% of eligible patients); equal patient probability under random hash. No phenotypes or results used.",
        "future_use":"All reserved patients excluded from present/future training and report donor pools. Evaluation panel is an ordered prefix to be fixed separately using development-based precision calculations.",
        "final_primary_test_size":None,"reports_opened":0,"images_opened":0,"masks_opened":0,
        "blind_human_labels_collected":0,"frame_sha256":sha256(private),
        "source_sha256":{str(f):sha256(f) for f in [*rex_files,sources/"train_metadata.csv",sources/"no_chest_train.txt"]}})
    print("reserved independent sampling frame",len(withheld),"of",len(ordered),"eligible patients")


if __name__=="__main__":
    main()
