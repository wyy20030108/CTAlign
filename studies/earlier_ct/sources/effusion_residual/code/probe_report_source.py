"""Exploratory teacher sensitivity under fixed CT. Uses the same 32 training U cases."""
import argparse
import os
from pathlib import Path
import numpy as np
import torch
from bounded_residual import P2, PTH, DIGEST, FullReportUNet, collate, correction, get_cases, patch
from frozen_priors import read, sha, write


@torch.inference_mode()
def main(seed):
    torch.set_num_threads(2)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cuda.matmul.allow_tf32=False
    assert sha(PTH)==DIGEST and seed in read(PTH)["seeds"]
    out=P2/"bounded"/str(seed)
    ck=out/"paired.pt"
    assert sha(ck)==read(out/"paired.json")["checkpoint_sha256"]
    state=torch.load(ck,map_location="cpu",weights_only=True)
    cases,codec,docs=get_cases()
    net=FullReportUNet(codec.vocab_size).cuda().eval()
    net.load_state_dict(state["teacher"])
    rows=[]
    sources=read(Path("phase1/data/U_report_strata32/manifest.private.json"))["rows"]
    for i,(c,original) in enumerate(zip(cases["U"],sources,strict=True)):
        assert c["name"]==original["name"]
        (weak,_,margin,_,_),start=patch(c,51907+i,"boundary",64)
        region=margin.abs()<=2
        if not region.any() or not c["donors"]:
            rows.append({"code":c["code"],"stratum":original["stratum"],"band_voxels":int(region.sum()),
                         "donors":len(c["donors"]),"scored":False})
            continue
        def output(name):
            tokens=collate([docs[name]],"cuda") if name else None
            return (margin+correction(net,weak,tokens)).sigmoid()[region]
        paired=output(c["name"])
        empty=output(None)
        repeat=(paired-output(c["name"])).abs().max().item()
        if repeat>1e-6:
            raise ValueError("Identical text failed numerical replay")
        d=[output(n) for n in c["donors"]]
        delta=[(paired-x).abs().mean().item() for x in d]
        flips=[((paired>.5)!=(x>.5)).float().mean().item() for x in d]
        rows.append({"code":c["code"],"stratum":original["stratum"],"band_voxels":int(region.sum()),
                     "donors":len(d),"scored":True,"paired_vs_empty_abs_score":(paired-empty).abs().mean().item(),
                     "paired_vs_matched_abs_score":float(np.mean(delta)),"paired_vs_matched_flip_fraction":float(np.mean(flips)),
                     "identical_repeat_max_abs":repeat,"start_xyz":start})
    valid=[r for r in rows if r["scored"]]
    write(out/"teacher_sensitivity.json",{"seed":seed,"model":"paired-arm EMA","n":len(rows),"eligible":len(valid),
          "mean_paired_vs_empty_abs_score":float(np.mean([r["paired_vs_empty_abs_score"] for r in valid])) if valid else None,
          "mean_paired_vs_matched_abs_score":float(np.mean([r["paired_vs_matched_abs_score"] for r in valid])) if valid else None,
          "mean_paired_vs_matched_flip_fraction":float(np.mean([r["paired_vs_matched_flip_fraction"] for r in valid])) if valid else None,
          "identical_repeat_max_abs":max((r["identical_repeat_max_abs"] for r in valid),default=None),
          "scope":"One fixed boundary-centered CT crop per U training patient. No U annotations. Only cases with an uncertain-band voxel and an eligible donor scored. Patient mean after equal averaging over all available matched donors. Measures sensitivity, not benefit.",
          "sha256":sha(Path(__file__)),"cases":rows})
    print("verified report-source replay",seed,len(valid),flush=True)


if __name__=="__main__":
    os.umask(0o077)
    p=argparse.ArgumentParser()
    p.add_argument("seed",type=int)
    main(p.parse_args().seed)
