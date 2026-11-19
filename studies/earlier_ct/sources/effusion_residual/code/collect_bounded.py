"""Recount source-grid saved predictions, aggregate without treating seeds as additional patients, package."""
import hashlib
import json
from pathlib import Path
import shutil
from zipfile import ZipFile, ZIP_DEFLATED
import numpy as np
from frozen_priors import read, sha, write
from trace_t7 import aggregate, score


def main():
    root=Path.cwd()
    p2=root/"phase2"
    protocol=p2/"residual_protocol.json"
    plan=read(protocol)
    arms=plan["arms"]
    cases=read(root/"phase1/cross_patient_T7/input_manifest.private.json")["cases"]
    comparator=read(p2/"development_7/comparator.json")
    items={}
    def add(path,name=None):
        name=name or path.relative_to(root).as_posix()
        if name in items:
            assert items[name]==path
        items[name]=path
    add(protocol)
    add(p2/"development_7/comparator.json")
    for f in ("basel_T7.py","frozen_priors.py","trace_t7.py","bounded_residual.py","eval_bounded.py",
              "probe_report_source.py","collect_bounded.py","run_bounded.sh"):
        add(root/f,"code/"+f)
    for f in ("raw_prediction_lock.json","inference_design.json","case_geometry_qc.json"):
        add(root/"phase1/basel_T7_comparator"/f)
    for i in (0,1):
        add(p2/f"prior_train/shard{i}.json")
    data={}
    for src,base in zip(cases,comparator["cases"],strict=True):
        code=src["code"]
        path=root/"phase1/data/T_partial"/(src["volume"].removesuffix(".nii.gz")+".npz")
        assert sha(path)==src["file_sha256"] and code==base["code"]
        with np.load(path,allow_pickle=False) as v:
            known,aff=v["positive_mask"].astype(bool),v["affine"]
        with np.load(p2/"development_7"/(code+".npz"),allow_pickle=False) as v:
            margin=v["margin"]
            assert np.array_equal(v["affine"],aff)
        data[code]=(known,aff,margin,float(abs(np.linalg.det(aff[:3,:3]))/1000))
        contour=root/"phase1/basel_T7_comparator/ai_binary"/(code+"_AI_effusion.nii.gz")
        assert sha(contour)==read(root/"phase1/basel_T7_comparator/raw_prediction_lock.json")["cases"][code]["hard_class2_sha256"]
        add(contour)
    summaries={}
    counts=0
    for seed in plan["seeds"]:
        run=p2/"bounded"/str(seed)
        add(run/"design.json")
        probe=read(run/"teacher_sensitivity.json")
        add(run/"teacher_sensitivity.json")
        assert probe["seed"]==seed and probe["n"]==32 and probe["identical_repeat_max_abs"]<=1e-6
        result={}
        case_rows={}
        for arm in arms:
            rep=run/"development_7"/(arm+".json")
            add(rep)
            add(run/(arm+".json"))
            report=read(rep)
            assert report["checkpoint_sha256"]==sha(run/(arm+".pt"))
            assert read(run/(arm+".json"))["steps"]==plan["updates"] and report["protocol_sha256"]==sha(protocol)
            assert [q["code"] for q in report["cases"]]==list(data)
            for row in report["cases"]:
                code=row["code"]
                known,aff,margin,vox=data[code]
                saved=run/"development_7"/arm/(code+".npz")
                assert sha(saved)==row["map_sha256"]
                add(saved)
                with np.load(saved,allow_pickle=False) as v:
                    assert np.array_equal(v["affine"],aff) and tuple(v["shape"])==known.shape
                    pred=np.unpackbits(v["bits"],count=known.size,bitorder="little").reshape(known.shape).astype(bool)
                actual=score(pred,known,vox)
                assert all(actual[k]==row[k] for k in actual)
                assert int((pred&known&(margin<=0)).sum())==row["gained_known"]
                assert int((~pred&known&(margin>0)).sum())==row["lost_known"]
                assert not np.any(pred & (margin<=-2)) and not np.any(~pred & (margin>2))
                counts+=1
            assert aggregate(report["cases"])==report["summary"]
            result[arm]=report["summary"]
            case_rows[arm]=report["cases"]
        delta=lambda a,b:100*(result[a]["mean_coverage"]-result[b]["mean_coverage"])
        summaries[str(seed)]={"arms":result,"paired_minus_matched_coverage_pp":delta("paired","matched"),
            "paired_minus_empty_coverage_pp":delta("paired","empty"),
            "paired_minus_matched_by_case_pp":[100*(a["coverage"]-b["coverage"]) for a,b in zip(case_rows["paired"],case_rows["matched"],strict=True)],
            "probe":{k:v for k,v in probe.items() if k.startswith("mean_") or k in ("eligible","identical_repeat_max_abs")}}
    assert counts==63
    usage=shutil.disk_usage(root)
    total={"protocol_sha256":sha(protocol),"n_T_train":6,"n_U_train":32,"n_T_dev":7,"independent_seeds":plan["seeds"],
           "updates_total":128*9,"verified_maps":counts,"scores_per_seed":summaries,
           "prior":comparator["basel"],"upper_bound_mean_coverage":comparator["best_possible_coverage_plus2"],
           "original_student_outside_prior_median_ml":comparator["paired_median_ml_outside_basel"],
           "free_data_disk_GiB_at_verification":usage.free/2**30,"research_plan_changed":False}
    output=p2/"diagnostic_summary.json"
    write(output,total)
    add(output)
    bundle=p2/"bounded_evidence.zip"
    with ZipFile(bundle,"w",ZIP_DEFLATED,compresslevel=5) as z:
        listing={}
        for name,path in sorted(items.items()):
            z.write(path,name)
            listing[name]=sha(path)
        z.writestr("file_hashes.json",json.dumps(listing,indent=2))
    with ZipFile(bundle) as z:
        assert z.testzip() is None and all(hashlib.sha256(z.read(k)).hexdigest()==v for k,v in listing.items())
    print("Verified",counts,"maps,",len(items),"evidence files;",sha(bundle),flush=True)


if __name__=="__main__":
    main()
