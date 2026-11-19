"""Seal the first fallback experiments, trained weights, anonymised patient results and code."""
import hashlib,json,os,shutil
from pathlib import Path
from zipfile import ZipFile,ZIP_DEFLATED
import numpy as np
from finding_train import OUT,read,write,sha
from finding_summary import difference

def main():
    base=read(OUT/"results_base.json");vol=read(OUT/"results_volume_term.json")
    assert base["verified_patient_maps"]==vol["verified_patient_maps"]==216
    native={}
    for arm in ("supervised","semi"):
        p=OUT/"native"/arm;rows=read(p/"summary.json")["rows"];tr=read(p/"training.json")
        assert len(rows)==18 and tr["steps"]==256 and tr["checkpoint"]==sha(p/"model.pt")
        by_case={}
        for r in rows:
            assert r["sha256"]==sha(p/(r["code"]+".npz"))
            by_case[r["code"]]={k:float(np.mean([s["dice"]*100 for s in v])) for k,v in r["scores"].items()}
        pairs=[s for r in rows for s in r["scores"]["paired"]]
        native[arm]=dict(patient=by_case,initial=tr["initial"],
             scores={k:float(np.mean([v[k] for v in by_case.values() if k in v])) for k in ("paired","paired_swapped","template","clinical","clinical_swapped")},
             predicted_median_ml=float(np.median([v["pred_voxels"]*.008 for v in pairs])),
             true_median_ml=float(np.median([v["true_voxels"]*.008 for v in pairs])),
             empty=sum(v["pred_voxels"]==0 for v in pairs),missed_target=sum(v["intersection"]==0 for v in pairs))
    assert native["supervised"]["initial"]==native["semi"]["initial"]
    codes=sorted(native["semi"]["patient"])
    native["semi_minus_supervised"]=difference([native["semi"]["patient"][c]["paired"]-native["supervised"]["patient"][c]["paired"] for c in codes])
    native["semi_correct_minus_swapped"]=difference([native["semi"]["patient"][c]["paired"]-native["semi"]["patient"][c]["paired_swapped"] for c in codes])
    native["clinical_effect"]=difference([v["clinical"]-v["clinical_swapped"] for v in native["semi"]["patient"].values() if "clinical" in v])
    for arm in ("supervised","semi"):
        cal=read(OUT/"native"/arm/"native_calibration.json");best=cal["best"]
        assert best==int(np.argmax(np.mean([v["correct"] for v in cal["rows"]["fallback_train"]],axis=0)))
        dev=[v for g in ("fallback_val","swap_dev") for v in cal["rows"][g]];assert len(dev)==18
        native[arm]["train_calibrated"]=dict(threshold=cal["thresholds"][best],
                training_dice=100*float(np.mean([v["correct"][best] for v in cal["rows"]["fallback_train"]])),
                development_dice=100*float(np.mean([v["correct"][best] for v in dev])),
                development_swapped_dice=100*float(np.mean([v["swapped"][best] for v in dev])))
    grid=read(OUT/"grid_qc.json")["cases"];grid_stats={}
    for n in (80,128,160):
        subset=[v for v in grid if v["side"]==n and v["group"]!="fallback_train"];assert len(subset)==36
        grid_stats[n]=dict(mean_dice=float(np.mean([v["dice"] for v in subset])),lost=sum(v["recovered_voxels"]==0 for v in subset))
    write(OUT/"overview.json",dict(n_fit=16,n_unlabelled=32,n_dev=18,n_queries=36,checkpoints=26,updates=6656,
            verified_patient_outputs=468,native=native,grid=grid_stats,overfit_control=read(OUT/"native/overfit_control.json"),
            free_GiB=shutil.disk_usage(OUT).free/2**30))
    sources=[Path(n+".py") for n in ("finding_train","finding_eval","finding_unpaired","finding_calibration","finding_summary","finding_volume_term",
        "finding_grid_qc","finding_native","finding_native_check","finding_release","frozen_priors","replay_controls","audit_pairs","swap_data","prepare_unlabeled",
        "fetch_four","fetch_manifest","real_pair_smoke","report_strata","volume_benchmark","diagnose_whole_T")]
    sources.extend([Path("phase1/geometry/geometry.py")])
    entries={f.relative_to(OUT).as_posix():f for f in OUT.rglob("*") if f.is_file() and f.suffix in (".json",".npz",".pt")}
    entries.update({"code/"+f.as_posix():f for f in sources});assert len([k for k in entries if k.endswith(".pt")])==26
    assert len([k for k in entries if k.endswith(".npz")])==468
    final=OUT/"finding_results.zip"
    with ZipFile(final.with_suffix(".part"),"w",ZIP_DEFLATED,compresslevel=4) as z:
        hashes={}
        for name,path in sorted(entries.items()):
            z.write(path,name);hashes[name]=sha(path)
        z.writestr("SHA256.json",json.dumps(hashes,indent=2))
    with ZipFile(final.with_suffix(".part")) as z:
        assert z.testzip() is None and all(hashlib.sha256(z.read(k)).hexdigest()==v for k,v in hashes.items())
    final.with_suffix(".part").replace(final)
    print(len(entries),final.stat().st_size,sha(final),flush=True)
if __name__=="__main__":
    os.umask(0o077);main()
