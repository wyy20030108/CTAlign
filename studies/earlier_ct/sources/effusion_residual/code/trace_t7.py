"""Recount the same seven partial-positive development CTs on the identical voxel grid."""
import os
from pathlib import Path
import nibabel as nib
import numpy as np
from frozen_priors import margin_file, read, sha, write


def score(mask, positive, voxel_ml):
    hit = np.count_nonzero(mask & positive)
    return {"known_voxels":int(positive.sum()),"intersect":int(hit),"foreground":int(mask.sum()),
            "coverage":float(hit / positive.sum()),"fraction_fov":float(mask.mean()),
            "volume_ml":float(mask.sum()*voxel_ml),"whole_fov_voxels":int(mask.size)}


def aggregate(records):
    return {"mean_coverage":float(np.mean([r["coverage"] for r in records])),
            "mean_fraction_fov":float(np.mean([r["fraction_fov"] for r in records])),
            "median_volume_ml":float(np.median([r["volume_ml"] for r in records])),
            "cases_zero_known_hits":sum(r["intersect"] == 0 for r in records)}


def main():
    root = Path.cwd()
    src = root / "phase1/basel_T7_comparator"
    locked = read(src / "raw_prediction_lock.json")
    assert sha(src/"inference_design.json") == locked["design_sha256"]
    raw = src / "prediction_raw"
    old = root / "phase1/full_report_trainonly/U32_source_comparison/development_7"
    manifest = root / "phase1/cross_patient_T7/input_manifest.private.json"
    assert sha(manifest) == read(src/"inference_design.json")["case_manifest_sha256"]
    protocol = root / "phase2/residual_protocol.json"
    out = root / "phase2/development_7"
    out.mkdir(exist_ok=True)
    rows = []
    for i, row in enumerate(read(manifest)["cases"]):
        code = row["code"]
        aff = np.asarray(row["affine_xyz_to_RAS_mm"])
        shape = tuple(row["shape_xyz"])
        voxel = abs(np.linalg.det(aff[:3,:3]))/1000
        for ext, signature in locked["cases"][code]["raw"].items():
            assert sha(raw/(code+ext)) == signature
        contour = src/"ai_binary"/(code+"_AI_effusion.nii.gz")
        assert sha(contour) == locked["cases"][code]["hard_class2_sha256"]
        prior = np.asarray(nib.load(contour).dataobj).astype(bool)
        receipt = margin_file(raw,code,shape,aff,out/(code+".npz"))
        with np.load(out/(code+".npz"),allow_pickle=False) as v:
            margin = v["margin"]
        assert np.array_equal(margin>0,prior)
        ct_pack = root / "phase1/data/T_partial" / (row["volume"].removesuffix(".nii.gz")+".npz")
        assert sha(ct_pack) == row["file_sha256"]
        with np.load(ct_pack,allow_pickle=False) as v:
            assert np.allclose(v["affine"],aff)
            known = v["positive_mask"].astype(bool)
        data = {"code":code, "margin_sha256":receipt["file_sha256"], "basel":score(prior,known,voxel),
                "upper_bound_plus2":score(margin > -2,known,voxel),"arms":{}}
        for arm in ("empty","paired","matched"):
            ref = read(old/(arm+".json"))["cases"][i]
            path = old/"predictions_private"/arm/(code+".npz")
            assert ref["code"] == code and sha(path) == ref["prediction_file_sha256"]
            with np.load(path,allow_pickle=False) as v:
                assert tuple(v["shape_xyz"])==shape and np.allclose(v["affine_xyz_to_RAS_mm"],aff)
                pred = np.unpackbits(v["positive_bitpacked"],bitorder="little",count=known.size).reshape(shape)>0
            metrics = score(pred,known,voxel)
            assert metrics["intersect"]==ref["intersection_voxels"] and metrics["foreground"]==ref["predicted_positive_voxels"]
            extra = pred & ~prior
            groups = [prior & pred, prior & ~pred, extra, ~prior & ~pred]
            counts = [int(np.count_nonzero(g & known)) for g in groups]
            assert sum(counts)==int(known.sum())
            data["arms"][arm] = {"metrics":metrics,"known_counts_both_prior_only_student_only_neither":counts,
                                 "student_only_ml":float(extra.sum()*voxel),
                                 "student_fraction_outside_prior":float(extra.sum()/pred.sum()) if pred.any() else None,
                                 "known_added_in_student_only_ml":float(counts[2]*voxel)}
        rows.append(data)
    result = {"n":len(rows),"protocol_sha256":sha(protocol),"raw_prior_lock_sha256":sha(src/"raw_prediction_lock.json"),
              "basel":aggregate([r["basel"] for r in rows]),"best_possible_coverage_plus2":float(np.mean([r["upper_bound_plus2"]["coverage"] for r in rows])),
              "arms":{a:aggregate([r["arms"][a]["metrics"] for r in rows]) for a in ("empty","paired","matched")},
              "paired_median_ml_outside_basel":float(np.median([r["arms"]["paired"]["student_only_ml"] for r in rows])),
              "cases":rows}
    write(out/"comparator.json",result)
    print({k:v for k,v in result.items() if k!="cases"})


if __name__=="__main__":
    os.umask(0o077)
    main()
