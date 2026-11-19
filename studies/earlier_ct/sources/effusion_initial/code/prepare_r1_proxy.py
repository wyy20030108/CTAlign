"""Build a four-patient, exclusively algorithm-referenced R1 engineering panel."""
import json
import os
import tempfile
from pathlib import Path
import nibabel as nib
import numpy as np
from nibabel.processing import resample_from_to
from fetch_four import sha256,write_json
from fetch_manifest import FILES
from prepare_unlabeled import obtain_file,release_path,convert_image


def main():
    os.umask(0o077)
    base=Path("phase1/R1_AI_drafts")
    out=Path("phase1/R1_proxy4")
    out.mkdir(parents=True,exist_ok=True)
    entries=json.loads((base/"mapping.private.json").read_text())[:4]
    if [e["case"] for e in entries]!=[f"R1A{i:03d}" for i in range(1,5)]:
        raise ValueError("Expected first four prespecified report-blind hash draws")
    repo,rev,_=FILES["ct_rate"]
    manifest=[]
    for entry in entries:
        code=entry["case"]
        name=entry["volume"]
        receipt=json.loads((base/"receipts"/(code+".json")).read_text())
        manual=receipt["annotation_type"]
        if not manual.startswith("AI_only") or not receipt["full_input_grid_and_affine_verified"]:
            raise ValueError("Proxy panel must use the frozen, voxel-checked AI product")
        mask_file=base/"masks"/(code+"_AI_effusion.nii.gz")
        if sha256(mask_file)!=receipt["export_sha256"]: raise ValueError("Mask checksum differs")
        with tempfile.TemporaryDirectory(dir=out) as tmp:
            path,digest=obtain_file(repo,rev,release_path(name),tmp)
            if digest!=receipt["ct_sha256"]: raise ValueError("Source differs from automatically contoured CT")
            mask=nib.load(mask_file)
            ct=nib.load(path)
            if not np.allclose(mask.affine,ct.affine,rtol=0,atol=1e-5) or mask.shape!=ct.shape:
                raise ValueError("Reference geometry differs")
            cache=Path(tmp)/"ct.npz"
            convert_image(path,cache)
            with np.load(cache,allow_pickle=False) as f:
                hu,grid=f["ct_hu"],f["affine"]
            ref=np.asarray(resample_from_to(mask,(hu.shape,grid),order=0,cval=0).dataobj,dtype=np.uint8)
            if not np.isin(ref,[0,1]).all(): raise ValueError("Invalid mask value")
            final=out/(code+".npz")
            np.savez_compressed(final,ct_hu=hu,AI_reference=ref,affine=grid)
        manifest.append({"code":code,"source_sha256":digest,"native_AI_mask_sha256":receipt["export_sha256"],
                         "cache_sha256":sha256(final),"file":str(final)})
    write_json(out/"manifest.private.json",manifest)
    print("fixed R1 AI-referenced panel prepared",len(manifest))


if __name__=="__main__":
    main()
