"""Preserve native-resolution, physical-space crops of findings lost at the pilot spacing."""
import json
import os
from pathlib import Path
import nibabel as nib
import numpy as np
from nibabel.affines import apply_affine
from fetch_four import sha256, write_json


def main():
    os.umask(0o077)
    root=Path("phase1/data/T_partial")
    jobs=json.loads((root/"native_review.private.json").read_text())
    out=root/"native_review"/"crops"
    out.mkdir(parents=True,exist_ok=True)
    records=[]
    for job in jobs:
        ct_path, seg_path=Path(job["ct"]),Path(job["mask"])
        if sha256(ct_path)!=job["ct_sha256"] or sha256(seg_path)!=job["mask_sha256"]:
            raise ValueError("Source checksum mismatch")
        image,segment=nib.load(ct_path),nib.load(seg_path)
        if len(segment.shape)!=4 or segment.shape[1:]!=image.shape:
            raise ValueError("Paired spatial dimensions differ")
        spacing=nib.affines.voxel_sizes(image.affine)
        margin=np.ceil(20/spacing).astype(int)
        for channel in job["channels"]:
            raw=np.asarray(segment.dataobj[channel,...])>0
            points=np.argwhere(raw)
            if not len(points):
                raise ValueError("Empty native finding")
            lo=np.maximum(points.min(axis=0)-margin,0)
            hi=np.minimum(points.max(axis=0)+margin+1,image.shape)
            crop=tuple(slice(int(a),int(b)) for a,b in zip(lo,hi))
            positive=raw[crop]
            assert int(positive.sum())==len(points)
            affine=image.affine.copy()
            affine[:3,3]=apply_affine(image.affine,lo)
            path=out/(job["name"].removesuffix(".nii.gz")+f"_finding{channel}.npz")
            np.savez_compressed(path,ct_hu=np.asarray(image.dataobj[crop]),positive_mask=positive.astype(np.uint8),affine=affine)
            with np.load(path) as verify:
                assert np.array_equal(verify["positive_mask"],positive)
            records.append({"source":job["name"],"channel":channel,"crop":str(path),"crop_sha256":sha256(path),
                            "was_lost_at_2mm":channel==job["channel_lost_at_2mm"],"positive_native_voxels":len(points),
                            "native_finding_ml":len(points)*abs(np.linalg.det(image.affine[:3,:3]))/1000,
                            "native_spacing_mm":spacing.tolist()})
    write_json(out/"crops.private.json",records)
    write_json(out/"receipt.json",{"patients":len(jobs),"retained_native_finding_channels":len(records),
               "vanished_channel_volumes_ml":[r["native_finding_ml"] for r in records if r["was_lost_at_2mm"]],
               "purpose":"Native resolution supplemental training/QC. Unknown voxels are not clinical negatives."})
    print("native channels preserved",len(records))


if __name__=="__main__":
    main()
