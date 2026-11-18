"""Verify the two experiments and export de-identified patient-level evidence."""
import json,os,shutil
from pathlib import Path
from zipfile import ZipFile,ZIP_DEFLATED
from collections import Counter
import numpy as np
from swap_data import out,read,sha256,write_json
from swap_clinical import SENTENCES,SPANS

def interval(values):
    x=np.array(values,float);rng=np.random.default_rng(891)
    z=x[rng.integers(len(x),size=(10000,len(x)))].mean(axis=1)
    return {"mean":float(x.mean()),"patient_bootstrap_95":np.quantile(z,[.025,.975]).tolist(),
            "positive_zero_negative":[int((x>0).sum()),int((x==0).sum()),int((x<0).sum())]}

def main():
    os.umask(0o077)
    idx=read(out/"index.private.json");rows=idx["pairs"];seeds=(891,1901,2707)
    report=[];details=[];clinical=[];separate=[];controls=[]
    label_qc={r["code"] for r in read(out/"fallback/review.private.json")}
    for r in rows:
        c=r["code"];rec=read(out/(c+".json"))
        if r["role"]=="fallback_train" or rec["status"]!="ok":continue
        assert sha256(out/(c+".npz"))==rec["sha256"]
        all_s=[];all_cl=[]
        for seed in seeds:
            s=read(out/"volume_final"/f"{c}_{seed}.json")
            assert s["code"]==c and s["role"]==r["role"] and s["seed"]==seed and s["script"]==sha256(Path("swap_volume.py"))
            assert s["k_voxels"]==round(s["volume_voxels"]*.01) and s["scores"]["empty"]["delta"]==0
            all_s.append(s);details.append(s)
            if c in (SENTENCES|SPANS):
                cl=read(out/"clinical"/f"{c}_{seed}.json")
                assert cl["k_voxels"]==s["k_voxels"] and cl["source_substrings_verified"] and cl["script"]==sha256(Path("swap_clinical.py"))
                clinical.append(cl);all_cl.append(cl)
        def mean(path):return float(np.mean([v["scores"][path[0]][path[1]] for v in all_s]))
        case=dict(code=c,role=r["role"],categories=r["categories"],review=c in label_qc,
                  true_cover=mean(("finding","correct_equal_volume")),swapped_cover=mean(("finding","swapped_equal_volume")),
                  delta_pp=100*mean(("finding","delta")),auc=mean(("finding","preference_auc")),template_auc=mean(("template","preference_auc")),
                  template_delta_pp=100*mean(("template","delta")),side_control=all_s[0].get("coordinate_control_auc"))
        if all_cl:case.update(clinical_kind=all_cl[0]["kind"],clinical_delta_pp=100*float(np.mean([v["delta"] for v in all_cl])),
                             clinical_auc=float(np.mean([v["auc"] for v in all_cl])))
        report.append(case)
    assert len(report)==len(read(out/"prior/swap_dev.json"))+len(read(out/"prior/fallback_val.json"))
    def agg(rs):
        return {"n":len(rs),**{k:interval([r[k] for r in rs]) for k in ("delta_pp","auc","template_auc","template_delta_pp","true_cover","swapped_cover")},
                "auc_minus_template":interval([r["auc"]-r["template_auc"] for r in rs]),
                "no_hits_either_condition":sum(r["true_cover"]==r["swapped_cover"]==0 for r in rs)}
    clinrows=[r for r in report if "clinical_kind" in r]
    groups={k:agg(rs) for k,rs in {"all":report,"swap_dev":[r for r in report if r["role"]=="swap_dev"],
        "fallback_val":[r for r in report if r["role"]=="fallback_val"],"pass_side_qc":[r for r in report if not r["review"]]}.items()}
    cs={}
    for kind in ("all","sentence","span"):
        rs=[r for r in clinrows if kind=="all" or kind==r["clinical_kind"]]
        cs[kind]=dict(n=len(rs),delta_pp=interval([r["clinical_delta_pp"] for r in rs]),auc=interval([r["clinical_auc"] for r in rs]))
    statuses=Counter(x["status"] for x in read(out/"prepared.json")["records"])
    orient=[r["side_control"] for r in report if r["side_control"] is not None]
    value=dict(screen={k:v for k,v in idx.items() if k!="pairs"},candidate_count=len(rows),first_pair_status=statuses,
        split_summary=groups,clinical=cs,side_control_n=len(orient),side_control_mean=float(np.mean(orient)),
        seed_summary={str(seed):{m:float(np.mean([s["scores"][m]["delta"]*100 for s in details if s["seed"]==seed])) for m in ("finding","template")} for seed in seeds},
        effusion_category_members=sum("2e" in r["categories"] for r in report),number_scores=len(details),number_clinical_scores=len(clinical),
        batch_schedule_auc_max_difference=max(abs(s.get("roi_batch_auc_difference",0)) for s in details),cases=report,
        fallback=read(out/"fallback/receipt.json"),bundle=read(out/"fallback/bundle_receipt.json"),disk_free_gib=shutil.disk_usage(out).free/2**30)
    write_json(out/"results.json",value)
    with ZipFile(out/"swap_results.zip","w",ZIP_DEFLATED) as z:
        z.write(out/"results.json","results.json")
        z.writestr("scores_per_patient_and_seed.json",json.dumps(details,indent=2))
        z.writestr("clinical_scores.json",json.dumps(clinical,indent=2))
        for f in ("swap_data","swap_eval","swap_volume","swap_clinical","swap_prior","swap_fallback","swap_dataset_bundle","swap_results"):
            z.write(f+".py","code/"+f+".py")
    print("verified",len(details),len(clinical),"evaluations",flush=True)
if __name__=="__main__":main()
