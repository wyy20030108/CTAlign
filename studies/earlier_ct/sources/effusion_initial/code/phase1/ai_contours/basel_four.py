"""Generate independent, CT-only, exploratory AI contours on four already-paid T cases."""
import csv
import gzip
import hashlib
import json
import os
import pickle
import re
import shutil
import sys
import time
import zipfile
from pathlib import Path

import nibabel as nib
import numpy as np


BASE='Task307_pleuralfinal/nnUNetTrainerV2__nnUNetPlansv2.1/'
WEIGHT_MD5='52f3dab3815733018367a698c77400b0'
LABEL_DOC='https://github.com/usb-radiology/pleuraleffusion/blob/97c6bcbd5d8753ac9015c654101390b91cf5600e/pleural_effusion/extract_lung_opacity.py#L50-L57'


def digest(path, kind='sha256'):
    with path.open('rb') as f: return hashlib.file_digest(f,kind).hexdigest()


def put(path,obj):
    tmp=path.with_suffix('.part')
    tmp.write_text(json.dumps(obj,indent=2,sort_keys=True)+'\n')
    tmp.replace(path)


def prepare(root):
    archive=Path('model.verified.zip')
    assert digest(archive,'md5')==WEIGHT_MD5
    model=Path('model_files')/BASE
    with zipfile.ZipFile(archive) as z:
        names=['plans.pkl']+[f'fold_{n}/model_final_checkpoint.model{end}' for n in range(5) for end in ('','.pkl')]
        if BASE+'postprocessing.json' in z.namelist():names.append('postprocessing.json')
        for name in names:
            path=model/name
            path.parent.mkdir(exist_ok=True,parents=True)
            with z.open(BASE+name) as src, path.with_suffix(path.suffix+'.part').open('wb') as dst:
                shutil.copyfileobj(src,dst,8<<20)
            path.with_suffix(path.suffix+'.part').replace(path)
    plans=pickle.loads((model/'plans.pkl').read_bytes())
    if sorted([int(v) for v in plans['all_classes']])!=[1,2] or int(plans['num_modalities'])!=1:
        raise ValueError('Unexpected model class/channel definitions')
    records=json.loads((root/'phase1/geometry/corrected.private.json').read_bytes())
    reservation=json.loads((root/'data/four_train/pilot_T_reservation.json').read_bytes())
    assert len(records)==4 and {r['name'] for r in records}==set(reservation['volumes'])
    with (root/'phase1/cohort/split_v1/b05/T.csv').open() as f:
        assert set(reservation['volumes'])<={r['volume'] for r in csv.DictReader(f)}
    inputs=Path('blinded_inputs');inputs.mkdir(exist_ok=True)
    covered=[]
    for i,r in enumerate(sorted(records,key=lambda r:r['name']),1):
        scan=(root/'phase1/geometry/corrected'/r['path']).resolve()
        assert scan.name==r['name'] and scan.is_relative_to((root/'phase1/geometry/corrected').resolve()) and digest(scan)==r['sha256']
        name=f'B{i:03d}'
        dest=inputs/(name+'_0000.nii.gz')
        if dest.is_symlink():
            assert dest.resolve()==scan
        else:
            dest.symlink_to(scan)
        covered.append(dict(case=name,volume=r['name'],sha256=r['sha256'],image_path=str(scan)))
    assert {p.name for p in inputs.iterdir()}=={r['case']+'_0000.nii.gz' for r in covered}
    put(Path('input_ledger.private.json'),covered)
    stage=max(plans['plans_per_stage'])
    band=plans['plans_per_stage'][stage]
    put(Path('model_provenance.json'),{
        'weights':'https://doi.org/10.5281/zenodo.21918112','study':'https://doi.org/10.1097/RLI.0000000000000869',
        'class_mapping_source':LABEL_DOC,'classes':{'0':'background','1':'lung','2':'pleural_effusion'},
        'weights_archive_sha256':digest(archive),'weights_archive_md5':WEIGHT_MD5,'folds':[0,1,2,3,4],
        'checkpoint':'model_final_checkpoint','combined_postprocessing_available':(model/'postprocessing.json').exists(),
        'training_patch_size':[int(k) for k in band['patch_size']], 'training_spacing':[float(k) for k in band['current_spacing']],
        'architecture':'nnUNetTrainerV2__nnUNetPlansv2.1','mirroring_TTA':True,'sliding_window_step':0.5,
        'scope':'CT only, Basel pretrained model. No paired reports or ReX label data supplied. Exploratory algorithmic segmentations.'})
    return model,inputs,covered


def convert(covered):
    binary=Path('ai_binary');binary.mkdir(exist_ok=True)
    audits=[]
    for case in covered:
        src=nib.load(case['image_path'])
        rawpath=Path('prediction_raw')/(case['case']+'.nii.gz')
        prediction=nib.load(rawpath)
        if src.shape!=prediction.shape or not np.allclose(src.affine,prediction.affine,rtol=0,atol=1e-3):
            raise ValueError(f'Prediction grid mismatch {case["case"]}')
        if src.header.get_xyzt_units()[0]!='mm':
            raise ValueError('Input units are not millimetres')
        data=np.asanyarray(prediction.dataobj)
        labs,nums=np.unique(data,return_counts=True)
        if not np.isin(labs,[0,1,2]).all():
            raise ValueError('Invalid class ID')
        counts={str(i):int(nums[labs==i].sum()) for i in range(3)}
        out=(data==2).astype('uint8')
        amount=int(out.sum(dtype=np.uint64))
        assert amount==counts['2']
        header=src.header.copy()
        header.set_data_dtype(np.uint8)
        output=nib.Nifti1Image(out,src.affine,header)
        for mode in ['sform','qform']:
            matrix,code=getattr(src,'get_'+mode)(coded=True)
            if code:
                getattr(output,'set_'+mode)(matrix,int(code))
        file=binary/(case['case']+'_AI_effusion.nii.gz')
        tmp=binary/(case['case']+'_AI_effusion.tmp.nii.gz')
        nib.save(output,tmp)
        tmp.replace(file)
        check=nib.load(file)
        if check.shape!=src.shape or not np.allclose(check.affine,src.affine,atol=1e-5,rtol=0):
            raise ValueError('Exported binary grid mismatch')
        written=np.asanyarray(check.dataobj)
        if not np.array_equal(written,out):raise ValueError('Exported segmentation is not identical to class 2')
        voxel_mm3=float(abs(np.linalg.det(src.affine[:3,:3])))
        audits.append({'case':case['case'],'ct_shape':src.shape,'affine_axis_codes':nib.aff2axcodes(src.affine),
                       'ct_sha256':case['sha256'],'raw_sha256':digest(rawpath),'export_sha256':digest(file),
                       'stored_class_counts':counts,'effusion_voxels':amount,'voxel_volume_mm3':voxel_mm3,
                       'ai_effusion_ml':amount*voxel_mm3/1000,'foreground_axial_slices':int(np.any(out,axis=(0,1)).sum()),
                       'lung_axial_slices':int(np.any(data==1,axis=(0,1)).sum()),'full_input_grid_and_affine_verified':True,
                       'empty_prediction':amount==0,'annotation_type':'AI_only_not_expert_or_complete_ground_truth'})
        del written,out,data
    put(Path('case_geometry_qc.json'),audits)
    put(Path('summary.json'),{'status':'complete','subject_to_clinical_review':True,'n_cases':len(audits),
                              'empty_predictions':sum(a['empty_prediction'] for a in audits),'cases':audits})


if __name__=='__main__':
    root=Path(sys.argv[1]).resolve()
    run=root/'phase1/ai_contours'
    os.chdir(run);os.umask(0o077)
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='1': raise RuntimeError('Reserve GPU 1 for this run')
    for k in ['nnUNet_raw_data_base','nnUNet_preprocessed','RESULTS_FOLDER']:
        os.environ[k]=str(run/'nnunet_runtime'/k)
    os.environ['TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD']='1'
    os.environ['nnUNet_n_proc_DA']='2'
    import torch
    from nnunet.inference.predict import predict_from_folder
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark=False
    model,inp,records=prepare(root)
    print('Validated publisher digest, model classes, input roles and CT SHA-256.',flush=True)
    started=time.monotonic()
    predict_from_folder(str(model),str(inp),'prediction_raw',[0,1,2,3,4],False,1,1,None,0,1,True,
                        mixed_precision=True,overwrite_existing=False,mode='normal',overwrite_all_in_gpu=False,
                        step_size=.5,checkpoint_name='model_final_checkpoint')
    convert(records)
    print('AI contours completed',len(records),'seconds',time.monotonic()-started,flush=True)
