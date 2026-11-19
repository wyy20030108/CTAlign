"""Report the fixed T7 developmental tradeoff for both auxiliary treatments, without tuning."""
import os
from pathlib import Path
import numpy as np
import torch
from full_report_model import FullReportUNet
from diagnose_whole_T import make_inputs, run_volume, score, verify_model, read, digest
from fetch_four import write_json, sha256


def main():
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--enriched",action="store_true")
    args=parser.parse_args()
    os.umask(0o077)
    torch.set_num_threads(2)
    base=Path(__file__).resolve().parent/("U32_source_comparison" if args.enriched else "auxiliary_AI")
    out=base/"development_7"
    out.mkdir(exist_ok=True)
    training=read(base/"summary.json")
    cases,plan=make_inputs()
    if plan!=read(Path("phase1/cross_patient_T7/input_manifest.private.json")):
        raise ValueError("Prior diagnostic sample or grid changed")
    results={}
    modes=("empty","paired","matched") if args.enriched else ("no_aux","with_aux")
    for mode in modes:
        for number in ((128,) if args.enriched else (32,128)):
            label=mode if args.enriched else f"{mode}_{number}"
            path=base/(label+".pt")
            entry=training["arms"][mode] if args.enriched else training["arms"][mode][str(number)]
            signed=entry["sha256"]
            if signed!=sha256(path): raise ValueError("Checkpoint changed")
            ck=torch.load(path,map_location="cpu",weights_only=True)
            if (ck["design"]!=training["design"] or ck.get("mode",ck.get("arm"))!=mode
                    or ck.get("step",entry.get("steps"))!=number):
                raise ValueError("Unexpected treatment or evaluation checkpoint")
            net=FullReportUNet(ck["vocab_size"]).cuda().eval()
            net.load_state_dict(ck["student"])
            dest=out/"predictions_private"/label
            dest.mkdir(parents=True,exist_ok=True)
            rows=[]
            for case,rec in zip(cases,plan["cases"],strict=True):
                pred=run_volume(net,True,case,rec)
                f=dest/(rec["code"]+".npz")
                np.savez_compressed(f,positive_bitpacked=np.packbits(pred,bitorder="little"),shape_xyz=np.array(pred.shape),
                                    affine_xyz_to_RAS_mm=np.array(rec["affine_xyz_to_RAS_mm"]))
                row=score(pred,case[1],rec["voxel_volume_mm3"])
                row.update(code=rec["code"],prediction_file_sha256=sha256(f))
                rows.append(row)
            stats={"patients":len(rows),"mean_known_positive_coverage":float(np.mean([r["known_positive_coverage_fraction"] for r in rows])),
                   "all_known_foreground_missed":sum(r["all_known_positive_missed"] for r in rows),
                   "mean_predicted_fraction_full_FOV":float(np.mean([r["predicted_fraction_full_FOV"] for r in rows])),
                   "median_predicted_volume_ml":float(np.median([r["predicted_volume_mL"] for r in rows]))}
            write_json(out/(label+".json"),{"model":label,"summary":stats,"cases":rows,"checkpoint_sha256":signed,
                       "input_digest":digest(plan),"patient_weighting":"equal"})
            if stats!=verify_model(out,label,plan,cases,signed):raise ValueError("Failed independent recomputation")
            results[label]=stats
            print("7 volumes verified",label,flush=True)
    write_json(out/"summary.json",{"inputs_sha256":digest(plan),"models":results,"prediction_maps_verified":len(cases)*len(results),
        "scope":"Same seven partially annotated T development patients reused after observing earlier oversegmentation. Includes only known-foreground coverage and total predicted extent, no clinical efficacy scores."})


if __name__=="__main__":
    main()
