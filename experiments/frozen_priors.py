"""Generate frozen, training-only spatial priors. Restore the nnU-Net crop to the CT grid."""
import argparse
import csv
import hashlib
import json
import os
import pickle
import tempfile
from pathlib import Path

import nibabel as nib
import numpy as np


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value, indent=2) + "\n")
    tmp.replace(path)


def roster(root):
    lock = read(root / "phase1/baselines/partial_v1/input_lock.private.json")
    us = read(root / "phase1/data/U_report_strata32/manifest.private.json")["rows"]
    entries = []
    for i, row in enumerate(lock["roles"]["T"], 1):
        name = row["volume"]
        path = root / "phase1/data/T_partial" / (name.removesuffix(".nii.gz") + ".npz")
        entries.append((f"T{i:02}", name, path, row["npz_sha256"]))
    for i, row in enumerate(us, 1):
        entries.append((f"U{i:02}", row["name"], root / row["file"], row["npz_sha256"]))
    for prefix, file in (("T","T"),("U","U_train")):
        with (root / f"phase1/cohort/split_v1/b10/{file}.csv").open() as stream:
            allowed = {r["volume"] for r in csv.DictReader(stream)}
        assert all(n in allowed for code,n,_,_ in entries if code.startswith(prefix))
    assert len(entries) == len({r[1] for r in entries}) == 38
    return entries


def margin_file(raw, code, shape, affine, dest):
    hard = nib.load(raw / (code + ".nii.gz"))
    assert hard.shape == shape and np.allclose(hard.affine, affine, atol=1e-4, rtol=0)
    with (raw / (code + ".pkl")).open("rb") as f:
        props = pickle.load(f)  # Generated locally by the verified nnU-Net installation.
    with np.load(raw / (code + ".npz"), allow_pickle=False) as f:
        p = f["softmax"].astype(np.float32)
    assert p.shape[0] == 3 and np.isfinite(p).all() and p.min() >= 0 and p.max() <= 1.001
    assert tuple(props["original_size_of_raw_data"]) == shape[::-1]
    bbox = props["crop_bbox"] or [(0,n) for n in shape[::-1]]
    sl = tuple(slice(a, min(b,n)) for (a,b),n in zip(bbox,shape[::-1],strict=True))
    margin = np.full(shape[::-1], -12, np.float16)
    # Margin > 0 denotes effusion beating both competing classes in the three-class model.
    rival = np.maximum(p[0],p[1])
    log_ratio = np.log(np.maximum(p[2], 1e-7)) - np.log(np.maximum(rival, 1e-7))
    assert margin[sl].shape == log_ratio.shape
    margin[sl] = np.clip(log_ratio, -12, 12)
    margin = np.ascontiguousarray(margin.T)
    binary = np.asarray(hard.dataobj) == 2
    # The official hard map is computed before its float16 sidecar is rounded.
    disagreements = (margin > 0) != binary
    if np.any(np.abs(margin[disagreements]) > .005):
        raise ValueError("Soft/hard alignment failed away from a score tie")
    margin[disagreements] = np.where(binary[disagreements], .001, -.001)
    assert np.array_equal(margin > 0, binary)
    tmp = dest.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, margin=margin, affine=affine)
    tmp.replace(dest)
    return {"file_sha256":sha(dest),"shape":list(shape),"hard_foreground":int(binary.sum()),
            "half_precision_tie_fixes":int(disagreements.sum()),"raw_soft_sha256":sha(raw/(code+".npz")),
            "raw_hard_sha256":sha(raw/(code+".nii.gz"))}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard", type=int, choices=(0,1), required=True)
    args = parser.parse_args()
    assert os.environ["CUDA_VISIBLE_DEVICES"] == str(args.shard)
    root = Path.cwd()
    src = root / "phase1/ai_contours"
    out = root / "phase2/prior_train"
    out.mkdir(parents=True, exist_ok=True)
    raw = out / f"raw{args.shard}"
    raw.mkdir(exist_ok=True)
    prov = read(src / "model_provenance.json")
    assert sha(src / "model.verified.zip") == prov["weights_archive_sha256"]
    for k in ("nnUNet_raw_data_base","nnUNet_preprocessed","RESULTS_FOLDER"):
        os.environ[k] = str(src / "nnunet_runtime" / k)
    os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"
    import torch
    from nnunet.inference.predict import predict_from_folder
    torch.set_num_threads(2)
    torch.backends.cudnn.benchmark = False
    model = src / "model_files/Task307_pleuralfinal/nnUNetTrainerV2__nnUNetPlansv2.1"
    rows = roster(root)[args.shard::2]
    with tempfile.TemporaryDirectory(prefix="ct_priors_") as batch:
        geometries = {}
        for code, name, path, signature in rows:
            assert sha(path) == signature
            with np.load(path,allow_pickle=False) as v:
                ct,affine = v["ct_hu"],v["affine"]
            assert ct.dtype == np.int16 and np.allclose(affine[:3,:3],2*np.eye(3),atol=1e-5)
            obj = nib.Nifti1Image(ct,affine)
            obj.header.set_xyzt_units("mm")
            obj.set_qform(affine,1)
            obj.set_sform(affine,1)
            nib.save(obj, Path(batch) / (code + "_0000.nii.gz"))
            geometries[code] = (ct.shape,affine)
        # Resume only complete sets. Incomplete output is regenerated in the same fixed shard.
        for code, *_ in rows:
            if not all((raw/(code+ext)).is_file() for ext in (".nii.gz",".npz",".pkl")):
                (raw/(code+".nii.gz")).unlink(missing_ok=True)
        predict_from_folder(str(model),batch,str(raw),[0,1,2,3,4],True,1,1,None,0,1,True,
            mixed_precision=True,overwrite_existing=False,mode="normal",overwrite_all_in_gpu=False,
            step_size=.5,checkpoint_name="model_final_checkpoint")
    receipt = {}
    for code,name,path,signature in rows:
        receipt[code] = margin_file(raw,code,*geometries[code],out/(code+".npz"))
        receipt[code]["source_sha256"] = signature
    write(out / f"shard{args.shard}.json", {"cases":receipt,"model_archive_sha256":prov["weights_archive_sha256"],
        "script_sha256":sha(Path(__file__)),"inputs":"CT HU and affine only","scope":"training T6 and U32"})
    print("Verified",len(rows),"spatial priors",flush=True)


if __name__ == "__main__":
    os.umask(0o077)
    main()
