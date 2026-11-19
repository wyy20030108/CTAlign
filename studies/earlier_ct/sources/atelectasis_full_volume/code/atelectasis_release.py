"""Independently count full-CT predictions against their prepared reference volumes."""
from pathlib import Path
import json
import numpy as np
from atelectasis_b0 import ROOT, LOCK, load_row
from frozen_priors import read, sha, write
import zipfile

def release():
    lock=read(LOCK);manifest=read(ROOT/"pilot_private.json")
    assert all(len(group)==len({r["code"] for r in group}) for group in manifest.values())
    assert not any(r["prior"] for r in manifest["holdout"])
    expected_codes={r["code"] for r in manifest["development"]}
    assert not expected_codes&{r["code"] for r in manifest["selected"]}
    groups={}
    entries=[ROOT/"screen.json",LOCK]
    for key in ("b10","all_labeled"):
        model_folders=sorted((ROOT/key).glob("*/results.json"))
        if not model_folders:continue
        groups[key]=[]
        for path in model_folders:
            s=read(path);folder=path.parent
            assert s["model_sha"]==sha(folder/"B0.pt") and s["lock_sha"]==sha(LOCK)
            stored={r["code"]:r for r in s["scores"]}
            assert expected_codes==stored.keys()
            assert set(s["train_codes"])=={r["code"] for r in manifest["selected" if key=="b10" else "train"]}
            entries+=[path,folder/"B0.pt"]
            for row in manifest["development"]:
                code=row["code"];f=folder/(code+".npz")
                assert sha(f)==stored[code]["prediction_sha"]
                _,truth=load_row(row)
                with np.load(f) as z:
                    assert tuple(z["shape"])==truth.shape
                    pred=np.unpackbits(z["threshold_half"],count=truth.size).reshape(truth.shape).astype(bool)
                tp=int(np.count_nonzero(pred&truth)); n=int(pred.sum()); total=int(truth.sum())
                score=2*tp/(n+total)
                assert total==stored[code]["truth"] and n==stored[code]["pred"]["0.5"]
                assert np.isclose(score,stored[code]["dice"]["0.5"],rtol=0,atol=1e-14)
                # Save scores and model. Dense image data and source reports stay on the project server.
            d=np.array([u["dice"]["0.5"] for u in stored.values()])
            z=np.array([u["empty_text_dice"]["0.5"] for u in stored.values()])
            groups[key].append(dict(seed=s["seed"],n=len(d),mean=float(d.mean()),median=float(np.median(d)),empty_text_mean=float(z.mean()),
                no_prediction=sum(u["pred"]["0.5"]==0 for u in stored.values()),zero_overlap=int(np.count_nonzero(d==0))))
    assert "b10" in groups and len(groups["b10"])==3
    write(ROOT/"baseline_summary.json",{"groups":groups,"stats":lock["stats"],"lock_sha":sha(LOCK),"holdout_opened_by_baseline":False})
    entries+=[ROOT/"baseline_summary.json",*sorted((ROOT/"receipts").glob("L???.json"))]
    code=[Path(f"atelectasis_{tag}.py") for tag in
          ("prepare","screen","b0","transfer","resumable","run","release")]
    with zipfile.ZipFile(ROOT/"atelectasis_evidence.zip","w",zipfile.ZIP_DEFLATED) as archive:
        for p in entries:archive.write(p,str(p.relative_to(ROOT)))
        for p in code:archive.write(p,"code/"+p.name)
        archive.writestr("checksums.json",json.dumps({(str(p.relative_to(ROOT)) if p in entries else "code/"+p.name):sha(p) for p in entries+code}))
    print((ROOT/"baseline_summary.json").read_text())
if __name__=="__main__":release()

