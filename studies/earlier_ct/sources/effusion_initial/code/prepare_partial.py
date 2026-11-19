"""Compact published expert findings for already-assigned T patients only."""
import argparse
import csv
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path
import numpy as np
from fetch_four import write_json, sha256
from fetch_manifest import FILES
from prepare_unlabeled import obtain_file, release_path

sys.path.insert(0, str(Path("phase1/geometry").resolve()))
from geometry import preprocess_finding


def run(screen, count):
    os.umask(0o077)
    dest = Path("phase1/data/T_partial")
    dest.mkdir(parents=True, exist_ok=True)
    split = Path("phase1/cohort/split_v1/b10/T.csv")
    with split.open(newline="") as f:
        allowed = {r["volume"] for r in csv.DictReader(f)}
    rows = [json.loads(s) for s in screen.read_text().splitlines() if s.strip()]
    if len(rows) != len(allowed) or {r["volume"] for r in rows} != allowed:
        raise ValueError("Screen must exactly match the authorized T split")
    selections = {}
    for row in rows:
        if row["data_check"] != "listed_calibration_finite":
            continue
        usable = [item for item in row["2e_findings"] if item["wording"] == "isolated_affirmative_fluid_wording"
                  and item["plausible_index_and_counts"]]
        if usable:
            selections[row["volume"]] = usable
    names = sorted(selections, key=lambda n: hashlib.sha256(("T-partial:891:" + n).encode()).digest())[:count]
    if not names:
        raise ValueError("No authorized candidate for a partial-label baseline")
    results, native_review = [], []
    native_dir = dest / "native_review"
    native_dir.mkdir(exist_ok=True, mode=0o700)
    ct_repo, ct_rev, _ = FILES["ct_rate"]
    rex_repo, rex_rev, _ = FILES["rex"]
    for index, name in enumerate(names, 1):
        target = dest / (name.removesuffix(".nii.gz") + ".npz")
        record = target.with_suffix(".json")
        selected_channels = [int(r["channel"]) for r in selections[name]]
        native_record = native_dir / (name + ".json")
        if native_record.exists():
            old=json.loads(native_record.read_text())
            if sha256(Path(old["ct"]))!=old["ct_sha256"] or sha256(Path(old["mask"]))!=old["mask_sha256"]:
                raise ValueError("Native-resolution review source changed")
            native_review.append(old)
            continue
        if record.exists() and target.exists():
            item = json.loads(record.read_text())
            if item["name"] == name and item["channels"] == selected_channels and item["npz_sha256"] == sha256(target):
                results.append(item)
                continue
            raise ValueError("Existing T derivative differs from this selection")
        with tempfile.TemporaryDirectory(prefix="t_finding_", dir=dest) as tmp:
            raw_ct, ct_hash = obtain_file(ct_repo, ct_rev, release_path(name), tmp)
            raw_mask, mask_hash = obtain_file(rex_repo, rex_rev, "segmentations/" + name, tmp)
            positives = None
            channel_geometry = []
            for info, channel in zip(selections[name], selected_channels):
                try:
                    ct, mask, grid = preprocess_finding(raw_ct, raw_mask, channel, spacing_mm=2.0)
                except ValueError as exc:
                    if str(exc) != "Positive contour lost at this spacing":
                        raise
                    keep_ct, keep_mask = native_dir / ("ct_"+name), native_dir / ("rex_"+name)
                    raw_ct.replace(keep_ct)
                    raw_mask.replace(keep_mask)
                    small={"name":name, "channel_lost_at_2mm":channel, "channels":selected_channels,
                           "ct":str(keep_ct),"mask":str(keep_mask),"ct_sha256":ct_hash,"mask_sha256":mask_hash,
                           "disposition":"Preserved native labels. Excluded from this 2-mm training cache; retain for higher-resolution analysis."}
                    write_json(native_record,small)
                    native_review.append(small)
                    break
                if ct.shape != mask.shape or not mask.any():
                    raise ValueError("Finding lost on 2-mm grid; review at native resolution")
                support = np.argwhere(mask)
                centered = (support[:,0] - (ct.shape[0]-1)/2) * np.linalg.norm(grid[:3,0])
                side = info["side_words"]
                same_side = (float(np.mean(centered < 0)) if side == "left" else
                             float(np.mean(centered > 0)) if side == "right" else None)
                channel_geometry.append({"index":channel,"published_side_word":side,
                    "fraction_on_named_side_of_fov":same_side,"compact_voxels":int(mask.sum()),
                    "ct_HU_p10_p50_p90":np.percentile(ct[mask.astype(bool)], [10,50,90]).tolist()})
                positives = mask.astype(bool) if positives is None else positives | mask.astype(bool)
            if native_record.exists():
                print("preserved_native_resolution", index, "/", len(names), flush=True)
                continue
            part = Path(tmp) / "cache.npz"
            np.savez_compressed(part, ct_hu=ct, positive_mask=positives.astype("uint8"), affine=grid)
            part.replace(target)
        item = {"name":name,"channels":selected_channels,"geometry":channel_geometry,
                "ct_sha256":ct_hash,"mask_sha256":mask_hash,"npz_sha256":sha256(target),
                "positive_voxels":int(positives.sum()),"file":str(target),
                "meaning":"Candidate expert finding support; outside annotated foreground remains unknown."}
        write_json(record, item)
        results.append(item)
        print("prepared_partial", index, "/", len(names), flush=True)
    write_json(dest / "manifest.private.json", {"source_split_sha256":sha256(split), "screen_sha256":sha256(screen),
               "status":"training-only finding masks awaiting image overlay review", "rows":results})
    write_json(dest / "native_review.private.json",native_review)
    write_json(dest / "receipt.json", {"patient_count":len(results), "eligible_in_T":len(selections),
             "original_resolution_review":len(native_review),
             "finding_channels":sum(len(r["channels"]) for r in results),
             "clinical_negative_patients_certified":0,"whole_volume_reference_patients_certified":0})


if __name__ == "__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--screen",type=Path,required=True)
    p.add_argument("--count",type=int,default=8)
    a=p.parse_args()
    if a.count<=0: p.error("count must be positive")
    run(a.screen,a.count)
