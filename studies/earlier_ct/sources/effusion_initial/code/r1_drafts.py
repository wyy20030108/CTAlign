"""Stream CT-only frozen Basel draft contours into a separate R1 algorithmic ledger."""
import argparse
import csv
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


def atomic_json(path, value):
    tmp=path.with_name(path.name+".part")
    tmp.write_text(json.dumps(value,indent=2)+"\n",encoding="utf8")
    tmp.replace(path)


def child(work,project):
    from basel_four import convert, BASE
    import torch
    from nnunet.inference.predict import predict_from_folder
    os.chdir(work)
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark=False
    base=project/"phase1/ai_contours"
    model=base/"model_files"/BASE
    records=json.loads(Path("batch.private.json").read_text())
    if {p.name for p in Path("ct_only").iterdir()}!={r["case"]+"_0000.nii.gz" for r in records}:
        raise ValueError("Input directory contains additional information")
    predict_from_folder(str(model),str(Path("ct_only").absolute()),str(Path("prediction_raw").absolute()),
                        [0,1,2,3,4],False,1,1,None,0,1,True,mixed_precision=True,overwrite_existing=False,
                        mode="normal",overwrite_all_in_gpu=False,step_size=.5,checkpoint_name="model_final_checkpoint")
    convert(records)


def run(root,limit,batch_size):
    import fcntl
    import nibabel as nib
    from prepare_unlabeled import obtain_file,release_path
    from fetch_four import sha256
    from fetch_manifest import FILES
    os.umask(0o077)
    base=root/"phase1/ai_contours"
    out=root/"phase1/R1_AI_drafts"
    out.mkdir(exist_ok=True,parents=True)
    worker_lock=(out/"worker.lock").open("a")
    fcntl.flock(worker_lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
    split=root/"phase1/cohort/split_v1"
    def roster(f):
        with f.open(newline="") as h: return list(csv.DictReader(h))
    source=split/"private/b10_R1.csv"
    source_rows=roster(source)
    excluded={row["patient"] for p in [split/"private/b10_R2.csv",split/"b10/T.csv",split/"b10/U_train.csv"]
              for row in roster(p)}
    if len(source_rows)!=54 or len({r["patient"] for r in source_rows})!=54 or any(r["patient"] in excluded for r in source_rows):
        raise ValueError("R1 roster failed exclusion check")
    ordered=sorted(source_rows,key=lambda row:hashlib.sha256(("R1-drafts:891:"+row["patient"]).encode()).digest())
    pairs=[{"case":f"R1A{i:03d}","volume":r["volume"]} for i,r in enumerate(ordered,1)]
    subset=pairs[:limit]
    provenance=json.loads((base/"model_provenance.json").read_text())
    plan={"source_roster_sha256":sha256(source),"weights_sha256":provenance["weights_archive_sha256"],
          "target":"R1 algorithmic annotation sidecar; no blinded human contours, reports, reference losses, or test evaluation",
          "model":"Basel Task307 official final checkpoints, 5 folds, native corrected CT, mirroring TTA, label 2 only",
          "panel":len(pairs),"case_order":"fixed patient hash v1","role":"R1 only",
          "model_paper":"https://doi.org/10.1097/RLI.0000000000000869","weights":"https://doi.org/10.5281/zenodo.21918112"}
    plan_path=out/"frozen_protocol.json"
    if plan_path.exists() and json.loads(plan_path.read_text())!=plan:
        raise ValueError("Existing annotation protocol differs")
    atomic_json(plan_path,plan)
    atomic_json(out/"mapping.private.json",pairs)
    receipts=out/"receipts"
    masks=out/"masks"
    for path in [receipts,masks]: path.mkdir(exist_ok=True)
    pending=[]
    def completed():
        done=[]
        for entry in pairs:
            f=receipts/(entry["case"]+".json")
            if f.exists():
                data=json.loads(f.read_text())
                if sha256(masks/(entry["case"]+"_AI_effusion.nii.gz"))!=data["export_sha256"]:
                    raise ValueError("Corrupt saved R1 mask")
                done.append(data)
        atomic_json(out/"progress.json",{"scheduled":len(pairs),"completed":len(done),
            "verified_native_grid":sum(r["full_input_grid_and_affine_verified"] for r in done),
            "human_verified":0,"reference_losses_computed":0,"role":"algorithmic drafts; disease counts sealed"})
        return {row["case"] for row in done}
    done=completed()
    pending=[e for e in subset if e["case"] not in done]
    repo,rev,_=FILES["ct_rate"]
    env=os.environ.copy()
    env["PYTHONPATH"]=os.pathsep.join([str(root),str(base)])
    env["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"]="1"
    for k in ("nnUNet_raw_data_base","nnUNet_preprocessed","RESULTS_FOLDER"):
        env[k]=str(base/"nnunet_runtime"/k)
    env["nnUNet_n_proc_DA"]="2"
    for i in range(0,len(pending),batch_size):
        if shutil.disk_usage(out).free<3*1024**3:
            raise RuntimeError("Insufficient data-disk space for this streaming batch")
        with tempfile.TemporaryDirectory(dir=out,prefix="stream_") as td:
            work=Path(td)
            inp=work/"ct_only"
            inp.mkdir()
            records=[]
            for r in pending[i:i+batch_size]:
                n=r["volume"]
                path,digest=obtain_file(repo,rev,release_path(n),work/"download")
                image=nib.load(path)
                if len(image.shape)!=3 or image.header.get_xyzt_units()[0]!="mm" or min(image.header.get_zooms()[:3])<=0:
                    raise ValueError("Invalid physical CT")
                alias=inp/(r["case"]+"_0000.nii.gz")
                alias.symlink_to(path.resolve())
                records.append({"case":r["case"],"image_path":str(alias),"sha256":digest})
            atomic_json(work/"batch.private.json",records)
            subprocess.run([str(base/"env/bin/python"),str(Path(__file__).resolve()),"--child",str(work),
                            "--project",str(root)],cwd=root,env=env,check=True)
            checked=json.loads((work/"case_geometry_qc.json").read_text())
            if {r["case"] for r in checked}!={r["case"] for r in records}: raise ValueError("Incomplete batch")
            for rec in checked:
                path=work/"ai_binary"/(rec["case"]+"_AI_effusion.nii.gz")
                if not rec["full_input_grid_and_affine_verified"] or sha256(path)!=rec["export_sha256"]:
                    raise ValueError("Invalid R1 prediction")
                path.replace(masks/path.name)
                atomic_json(receipts/(rec["case"]+".json"),rec)
        print("R1 AI drafts written:",len(completed()),"/",len(pairs),flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--child",type=Path)
    parser.add_argument("--project",type=Path,default=Path("."))
    parser.add_argument("--limit",type=int,default=54)
    parser.add_argument("--batch-size",type=int,default=3)
    opts=parser.parse_args()
    if opts.child:
        marker=opts.project.resolve()/"phase1/R1_AI_drafts/handoff_request.json"
        if marker.is_file() and json.loads(marker.read_text())["legacy_pid"]==os.getppid():
            from r1_shards import transfer_legacy_batch
            transfer_legacy_batch(opts.project.resolve(),opts.child.resolve())
            sys.exit(75)
        child(opts.child.resolve(),opts.project.resolve())
    else:
        if not 0<opts.limit<=54 or not 1<=opts.batch_size<=4: parser.error("Invalid R1 batch size/limit")
        run(opts.project.resolve(),opts.limit,opts.batch_size)
