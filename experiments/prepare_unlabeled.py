"""Stream a preselected subset of training-only CT-RATE volumes to a compact 2-mm RAS grid."""
import argparse
import csv
import hashlib
import json
import os
import tempfile
from pathlib import Path
import nibabel as nib
import numpy as np
from huggingface_hub import HfApi, hf_hub_download
from nibabel.processing import resample_to_output
from audit_pairs import identity
from fetch_manifest import FILES
from fetch_four import sha256, write_json


def release_path(name):
    _, (split, number), (_, _, exam) = identity(name)
    patient = f"{split}_{number}"
    return f"dataset/{split}_fixed/{patient}/{patient}_{exam}/{name}"


def obtain_file(repo, rev, name, temp):
    api = HfApi(endpoint="https://huggingface.co")
    info = api.get_paths_info(repo, [name], repo_type="dataset", revision=rev)
    if len(info) != 1 or not info[0].lfs:
        raise ValueError("Pinned file lacks a publisher hash")
    checksum = info[0].lfs.sha256
    path = Path(hf_hub_download(repo, filename=name, repo_type="dataset", revision=rev,
                                local_dir=temp, endpoint="https://huggingface.co"))
    if sha256(path) != checksum:
        raise ValueError("Pinned download checksum mismatch")
    return path, checksum


def convert_image(path, out):
    image = nib.load(path)
    spacing = np.asarray(image.header.get_zooms()[:3])
    if (len(image.shape) != 3 or np.any(~np.isfinite(spacing)) or np.any(spacing <= 0)
            or not np.isfinite(image.affine).all() or abs(np.linalg.det(image.affine[:3,:3])) < 1e-6):
        raise ValueError("Invalid physical CT geometry")
    result = resample_to_output(image, (2, 2, 2), order=1, cval=-1024)
    hu = np.asanyarray(result.dataobj)
    if not np.isfinite(hu).all() or hu.max() <= hu.min() or nib.aff2axcodes(result.affine) != ("R","A","S"):
        raise ValueError("Invalid HU volume or orientation")
    if hu.min() < -32768 or hu.max() > 32767:
        raise ValueError("HU values outside int16 support")
    with out.open("wb") as f:
        np.savez_compressed(f, ct_hu=np.rint(hu).astype(np.int16), affine=result.affine)
    return list(result.shape)


def prepare(count):
    os.umask(0o077)
    split = Path("phase1/cohort/split_v1/b10/U_train.csv")
    out = Path("phase1/data/U")
    out.mkdir(parents=True, exist_ok=True)
    with split.open(newline="") as f:
        eligible = [r["volume"] for r in csv.DictReader(f)]
    excluded = {Path(s.strip()).name for path in Path("data/four_train/ct_rate/dataset/metadata").glob("no_chest_*.txt")
                for s in path.read_text().splitlines()}
    if not excluded or count > len(eligible):
        raise ValueError("Missing non-chest exclusion metadata or requested too many patients")
    names = sorted((n for n in eligible if n not in excluded),
                   key=lambda n: hashlib.sha256(f"U-stage1:891:{n}".encode()).digest())[:count]
    if len(names) != count or len(set(identity(n)[1] for n in names)) != count:
        raise ValueError("Insufficient distinct U patients")
    repo, rev, _ = FILES["ct_rate"]
    from huggingface_hub import __version__ as hf_version
    results = []
    for i, n in enumerate(names, 1):
        item = out / (n.removesuffix(".nii.gz") + ".npz")
        record = item.with_suffix(".json")
        if record.exists() and item.exists():
            row = json.loads(record.read_text())
            if (row["name"] == n and row["revision"] == rev and row["grid_mm"] == 2
                    and sha256(item) == row["npz_sha256"]):
                results.append(row)
                continue
            raise ValueError("Cached U preprocessing record does not match the committed data")
        with tempfile.TemporaryDirectory(prefix="ct_stream_", dir=out) as tmp:
            source, digest = obtain_file(repo, rev, release_path(n), tmp)
            partial = Path(tmp) / "compact.npz"
            shape = convert_image(source, partial)
            partial.replace(item)
        row = {"name":n, "file":str(item), "revision":rev, "source_sha256":digest,
               "npz_sha256":sha256(item), "shape":shape, "grid_mm":2}
        write_json(record, row)
        results.append(row)
        print("compressed_U", i, "/", count, flush=True)
    write_json(out / "manifest.private.json", {"role":"U_train", "source_manifest_sha256":sha256(split), "rows":results})
    write_json(out / "receipt.json", {"stage":"U_train compact subset", "count":len(results), "seed":891,
               "image":"corrected CT, 2-mm RAS, linear intensity resampling", "HF":hf_version,
               "masks_opened":0, "persistent_npz_bytes":sum(Path(r["file"]).stat().st_size for r in results)})


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--count", type=int, default=8)
    args = p.parse_args()
    if args.count <= 0:
        p.error("count must be positive")
    prepare(args.count)
