"""Audit ReXGroundingCT release manifests against CT-RATE reports (standard library).

The CT-RATE naming unit is split_patient_exam_reconstruction.nii.gz, e.g.
valid_53_a_1.nii.gz. ReX splits are independent of this source namespace.
Only names from ReX are used. No images, annotations or test masks are opened.
ReX docs: https://huggingface.co/datasets/rajpurkarlab/ReXGroundingCT
CT-RATE naming / fields: https://github.com/ibrahimethemhamamci/CT-CLIP
"""

import argparse
import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path


VOLUME = re.compile(r"(train|valid)_([0-9]+)_([A-Za-z]+)_([0-9]+)\.nii\.gz")
PLACEHOLDERS = {"", "not given", "not available", "n/a", "na", "nan", "null"}


def identity(value):
    if not isinstance(value, str) or not (match := VOLUME.fullmatch(value.strip())):
        raise ValueError("invalid CT-RATE volume name (expected train|valid_patient_exam_reconstruction.nii.gz)")
    source, number, exam, _ = match.groups()
    patient = (source, number)
    return value.strip(), patient, (*patient, exam)


def split_names(path):
    with path.open(encoding="utf-8-sig") as stream:
        document = json.load(stream)
    if not isinstance(document, dict) or not document:
        raise ValueError("expected the release JSON object keyed by train/val/test")
    result = {}
    for split, rows in document.items():
        if split not in {"train", "val", "test"} or not isinstance(rows, list):
            raise ValueError("unexpected release manifest layout; check the dataset version")
        volumes = Counter()
        for index, row in enumerate(rows, 1):
            try:
                name = row.get("name") if isinstance(row, dict) else row
                volumes[identity(name)[0]] += 1
            except ValueError as exc:
                raise ValueError(f"invalid name in {path.name}, {split} row {index}") from exc
        result[split] = volumes
    return result


def canonical_section(raw):
    clean = " ".join((raw or "").split())
    return "" if clean.casefold().strip(". ") in PLACEHOLDERS else clean


def run(rex_path, reports, extra_holdouts, *, matched_names=None):
    manifests = split_names(rex_path)
    if not manifests.get("train"):
        raise ValueError("the base ReX manifest must include a nonempty train split")
    for index, path in enumerate(extra_holdouts, 1):
        extra = split_names(path)
        if not any(extra.get(split) for split in ("val", "test")):
            raise ValueError("--extra-holdout has no validation or test names")
        for split, names in extra.items():
            if split == "train":
                # The official challenge JSON repeats the training split. Only its
                # held-out patients belong in the exclusions for the original run.
                continue
            manifests[f"holdout_{index}_{split}"] = names
    patients = {key: {identity(n)[1] for n in names} for key, names in manifests.items()}
    reserved = set().union(*(group for key, group in patients.items() if key != "train"))
    train = manifests["train"]
    eligible = patients["train"] - reserved
    row_count, volume_counts = 0, Counter()
    report_versions, exam_reports = defaultdict(set), defaultdict(set)
    blank_rows = 0
    for path in reports:
        with path.open(encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            required = {"VolumeName", "Findings_EN", "Impressions_EN"}
            if not reader.fieldnames or not required.issubset(reader.fieldnames):
                raise ValueError(f"{path.name} is missing one of the required headers: {sorted(required)}")
            if len(reader.fieldnames) != len(set(reader.fieldnames)):
                raise ValueError(f"{path.name} has duplicated CSV headers")
            for row in reader:
                if None in row or any(row[key] is None for key in required):
                    raise ValueError(f"malformed CSV record in {path.name} at line {reader.line_num}")
                row_count += 1
                try:
                    name, patient, exam = identity(row["VolumeName"])
                except ValueError as exc:
                    raise ValueError(f"invalid volume name in {path.name} at line {reader.line_num}") from exc
                volume_counts[name] += 1
                if patient not in eligible:
                    continue
                sections = [canonical_section(row[col]) for col in ("Findings_EN", "Impressions_EN")]
                signature = None
                if any(sections):
                    signature = hashlib.sha256(json.dumps(sections).encode()).digest()
                    exam_reports[exam].add(signature)
                else:
                    blank_rows += 1
                report_versions[name].add(signature)

    counts = Counter(dict.fromkeys(
        ("heldout_patient", "duplicated_rex_train_key", "missing_report_key",
         "conflicting_report_rows", "empty_report", "inconsistent_exam_report", "matched_nonempty"), 0))
    joined_patients, joined_exams = set(), Counter()
    for name, multiplicity in train.items():
        _, patient, exam = identity(name)
        if patient in reserved:
            reason = "heldout_patient"
        elif multiplicity > 1:
            reason = "duplicated_rex_train_key"
        elif name not in report_versions:
            reason = "missing_report_key"
        elif len(report_versions[name]) > 1:
            reason = "conflicting_report_rows"
        elif report_versions[name] == {None}:
            reason = "empty_report"
        elif len(exam_reports[exam]) > 1:
            reason = "inconsistent_exam_report"
        else:
            reason = "matched_nonempty"
            joined_patients.add(patient)
            joined_exams[exam] += 1
            if matched_names is not None:
                matched_names.append(name)
        counts[reason] += 1

    return {
        "scope": "Metadata join only. Scan files, geometry, mask completeness and effusion status unverified.",
        "rex_by_split": {
            key: {"rows": sum(v.values()), "unique_volumes": len(v), "unique_patients": len(patients[key]),
                  "duplicate_keys": sum(n > 1 for n in v.values())}
            for key, v in manifests.items()
        },
        "patient_overlap_before_exclusion": {
            f"{a}/{b}": len(patients[a] & patients[b]) for a, b in combinations(manifests, 2)
        },
        "report_csv": {
            "total_rows": row_count, "unique_keys": len(volume_counts),
            "duplicate_keys": sum(n > 1 for n in volume_counts.values()),
            "empty_rows_in_eligible_rex_train_patients": blank_rows,
            "conflicting_volume_keys_in_eligible_rex_train_patients": sum(len(v) > 1 for v in report_versions.values()),
            "conflicting_nonempty_reports_within_eligible_exams": sum(len(v) > 1 for v in exam_reports.values()),
        },
        "exclusive_disposition_of_unique_rex_train_volumes": dict(counts),
        "report_matched_rex_train": {
            "unique_patients": len(joined_patients), "unique_exams": len(joined_exams),
            "unique_reconstructions": sum(joined_exams.values()),
            "exams_with_multiple_matched_reconstructions": sum(n > 1 for n in joined_exams.values()),
        },
        "reference_masks_verified": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rex", required=True, type=Path, help="original ReX dataset.json")
    parser.add_argument("--reports", required=True, nargs="+", type=Path, help="CT-RATE English report CSV(s)")
    parser.add_argument("--extra-holdout", action="append", default=[], type=Path, help="additional val/test manifests")
    args = parser.parse_args()
    try:
        summary = run(args.rex, args.reports, args.extra_holdout)
    except (csv.Error, OSError, ValueError) as exc:
        parser.exit(2, f"Audit error: {exc}\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary["report_matched_rex_train"]["unique_patients"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
