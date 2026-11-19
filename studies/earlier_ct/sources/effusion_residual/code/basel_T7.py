"""Prepare CT-only T7 inputs, run the frozen Basel ensemble once, and seal official hard/soft outputs."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from zipfile import ZipFile
import nibabel as nib
import numpy as np


def sha(p):
    with p.open('rb') as f:
        return hashlib.file_digest(f,'sha256').hexdigest()


def read(p):
    return json.loads(p.read_text())


def put(p,v):
    temp=p.with_suffix('.part');temp.write_text(json.dumps(v,indent=2)+'\n');temp.replace(p)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--project',type=Path,default=Path.cwd());arg=parser.parse_args()
    root=arg.project.resolve(); out=root/'phase1/basel_T7_comparator';src=root/'phase1/ai_contours'; sys.path.insert(0,str(src))
    from basel_four import BASE,WEIGHT_MD5,convert
    os.umask(0o077)
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='1':
        raise ValueError('Only physical GPU 1 is allocated')
    out.mkdir(exist_ok=True); inp=out/'ct_only';inp.mkdir(exist_ok=True);os.chdir(out)
    if (out/'raw_prediction_lock.json').exists():
        raise FileExistsError('Predictions are frozen; do not rerun inference')
    manifest=root/'phase1/cross_patient_T7/input_manifest.private.json'; rows=read(manifest)['cases']
    if len(rows)!=7 or [r['code'] for r in rows]!=[f'Tdiag{i:02}' for i in range(1,8)]:
        raise ValueError('Expected exactly the prior T7 cohort and case order')
    ledger=[]; inputs=[]
    for r in rows:
        code=r['code'];folder=root/'phase1/data/T_partial';pack=folder/(r['volume'].removesuffix('.nii.gz')+'.npz')
        if pack.resolve().parent!=folder.resolve() or sha(pack)!=r['file_sha256']:
            raise ValueError('Unexpected source derivative')
        with np.load(pack,allow_pickle=False) as z:
            ct=z['ct_hu'];aff=z['affine']  # Only these two independently compressed members are opened.
        if (ct.dtype!=np.int16 or ct.ndim!=3 or not np.isfinite(aff).all() or
            not np.allclose(aff[:3,:3],np.eye(3)*2,atol=1e-5,rtol=0) or not np.allclose(aff[3],[0,0,0,1])):
            raise ValueError('Expected 2-mm RAS CT in HU')
        image=nib.Nifti1Image(ct,aff);image.header.set_xyzt_units('mm');image.set_qform(aff,1);image.set_sform(aff,1)
        scan=inp/(code+'_0000.nii.gz');nib.save(image,scan);echo=nib.load(scan)
        if not np.array_equal(ct,np.asanyarray(echo.dataobj)) or not np.allclose(echo.affine,aff,atol=1e-5,rtol=0):
            raise ValueError('CT export lost HU values or native grid')
        h=sha(scan); ledger.append({'case':code,'input_sha256':h,'source_npz_sha256':r['file_sha256'],
                                    'shape_xyz':list(ct.shape),'affine_xyz_to_RAS_mm':aff.tolist()})
        inputs.append({'case':code,'image_path':str(scan),'sha256':h})
    if {f.name for f in inp.iterdir()}!={r['case']+'_0000.nii.gz' for r in inputs}:
        raise ValueError('Unexpected data in CT-only folder')
    weights=src/'model.verified.zip'; model=src/'model_files'/BASE;prov=read(src/'model_provenance.json')
    if sha(weights)!=prov['weights_archive_sha256'] or prov['weights_archive_md5']!=WEIGHT_MD5:
        raise ValueError('Publisher-verified weights archive changed')
    with ZipFile(weights) as z:
        names=['plans.pkl']+[f'fold_{k}/model_final_checkpoint.model{tail}' for k in range(5) for tail in ('','.pkl')]
        if BASE+'postprocessing.json' in z.namelist():names.append('postprocessing.json')
        for f in names:
            with z.open(BASE+f) as stream:
                if hashlib.file_digest(stream,'sha256').hexdigest()!=sha(model/f):raise ValueError('Extracted model has changed')
    put(out/'ct_input_ledger.json',ledger)
    design={'role':'seven previously used partially-labelled T development scans','case_manifest_sha256':sha(manifest),
            'ct_inputs_sha256':sha(out/'ct_input_ledger.json'),'input':'CT HU, 2-mm RAS+ XYZ only',
            'model_archive_sha256':sha(weights),'five_folds':[0,1,2,3,4],'step_size':.5,'test_time_mirroring':True,
            'mixed_precision':True,'checkpoint':'model_final_checkpoint','trained_class_to_extract':2,
            'save_npz':True,'prediction_mode':'normal official nnU-Net v1','GPU':1,'script_sha256':sha(Path(__file__)),
            'scoring_order':'seal official native hard maps and saved scores for all seven before opening source positive-mask members',
            'comparison':'macro known-positive coverage, full field foreground fraction, median predicted mL; paired U32 archived binary maps and shared/only/neither covered known foreground'}
    locked_raw=out/'inference_design.json'
    if locked_raw.exists() and read(locked_raw)!=design:raise ValueError('Different inference settings')
    put(locked_raw,design)
    for k in ('nnUNet_raw_data_base','nnUNet_preprocessed','RESULTS_FOLDER'):os.environ[k]=str(src/'nnunet_runtime'/k)
    os.environ['TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD']='1'
    import torch
    from nnunet.inference.predict import predict_from_folder
    torch.set_num_threads(4);torch.backends.cudnn.benchmark=False;start=time.monotonic()
    print('Frozen five-fold inference on seven CT-only volumes; saving official score files.',flush=True)
    predict_from_folder(str(model),str(inp),str(out/'prediction_raw'),[0,1,2,3,4],True,1,1,None,0,1,True,
                        mixed_precision=True,overwrite_existing=False,mode='normal',overwrite_all_in_gpu=False,step_size=.5,checkpoint_name='model_final_checkpoint')
    convert(inputs)
    result={}
    for row in ledger:
        code=row['case'];raw=out/'prediction_raw';binary=out/'ai_binary'/(code+'_AI_effusion.nii.gz')
        files={ext:raw/(code+ext) for ext in ('.nii.gz','.npz','.pkl')}
        if any(not f.is_file() for f in files.values()) or not binary.is_file():raise ValueError('Missing frozen Basel output')
        obj=nib.load(binary)
        if list(obj.shape)!=row['shape_xyz'] or not np.allclose(obj.affine,row['affine_xyz_to_RAS_mm'],atol=1e-5,rtol=0):
            raise ValueError('Returned class-2 hard map differs from CT grid')
        result[code]={'raw':{ext:sha(p) for ext,p in files.items()},'hard_class2_sha256':sha(binary)}
    put(out/'raw_prediction_lock.json',{'cases':result,'design_sha256':sha(locked_raw),'ct_input_ledger_sha256':sha(out/'ct_input_ledger.json'),
                                      'explicit_label_access_during_inference':0,'elapsed_seconds':time.monotonic()-start})
    print('7 official softmax/property and hard-map sets frozen',flush=True)


if __name__=='__main__':
    main()
