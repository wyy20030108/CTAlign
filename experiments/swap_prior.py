"""Recreate the frozen CT component of the deployed teacher on swap cases."""
import os,sys,time,tempfile
from pathlib import Path
import numpy as np
import nibabel as nib
from frozen_priors import margin_file,read,write,sha

def main(role):
    out=Path("phase3_swap"); dest=out/"prior"; dest.mkdir(exist_ok=True)
    base=Path("phase1/ai_contours").resolve()
    for k in ("nnUNet_raw_data_base","nnUNet_preprocessed","RESULTS_FOLDER"):os.environ[k]=str(base/"nnunet_runtime"/k)
    os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"]="1"
    import torch
    from nnunet.inference.predict import predict_from_folder
    torch.set_num_threads(2)
    geometry={}
    with tempfile.TemporaryDirectory(prefix="swap_basel_") as tmp:
        inp=Path(tmp)/"ct";inp.mkdir(); raw=Path(tmp)/"raw"
        for r in read(out/"index.private.json")["pairs"]:
            if r["role"]!=role: continue
            code=r["code"]; rec=out/(code+".json")
            while not rec.exists():time.sleep(2)
            entry=read(rec)
            if entry["status"]!="ok":continue
            f=out/(code+".npz");assert sha(f)==entry["sha256"]
            with np.load(f) as v: ct,aff=v["ct"],v["affine"]
            img=nib.Nifti1Image(ct,aff);img.header.set_xyzt_units("mm");nib.save(img,inp/(code+"_0000.nii.gz"))
            geometry[code]=(ct.shape,aff)
        model=base/"model_files/Task307_pleuralfinal/nnUNetTrainerV2__nnUNetPlansv2.1"
        assert sha(base/"model.verified.zip")==read(base/"model_provenance.json")["weights_archive_sha256"]
        predict_from_folder(str(model),str(inp),str(raw),[0,1,2,3,4],True,1,1,None,0,1,True,mixed_precision=True,
            overwrite_existing=False,mode="normal",overwrite_all_in_gpu=False,step_size=.5,checkpoint_name="model_final_checkpoint")
        items={c:margin_file(raw,c,*g,dest/(c+".npz")) for c,g in geometry.items()}
        write(dest/(role+".json"),items)

if __name__=="__main__":main(sys.argv[1])
