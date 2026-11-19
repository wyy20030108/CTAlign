"""Independent image-only inference on the first six prespecified training cases."""
import csv
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import nibabel as nib
import numpy as np

from basel_four import BASE, WEIGHT_MD5, convert, digest, put


def prepare(root, run):
    from zipfile import ZipFile
    partial = root / 'phase1/data/T_partial'
    manifest = run / 'selection_qc_snapshot.private.json'
    source = json.loads(manifest.read_bytes())
    split = root / 'phase1/cohort/split_v1/b10/T.csv'
    assert digest(split) == source['split_sha256']
    with split.open() as f:
        allowed = {r['volume']:r['patient'] for r in csv.DictReader(f)}
    # The custodian signed six positives before model output. Keep all six, with no replacements.
    # Only the signed case names and digests go into derivative extraction.
    assert len(source['usable']) == 6 and len(source['excluded']) == 3 and len(source['cases']) == 9
    names = sorted(source['usable'], key=lambda n:hashlib.sha256(('T-partial:891:'+n).encode()).digest())
    selected = [(n, partial/(n.removesuffix('.nii.gz')+'.npz'),source['cases'][n]['npz_sha256']) for n in names]
    assert len(selected) == 6 and len({allowed[n] for n,_,_ in selected}) == 6
    inp = run / 'ct_only_inputs'; inp.mkdir(exist_ok=True)
    series = []
    for i,(name,infile,checksum) in enumerate(selected, 1):
        npz = (root / infile).resolve()
        assert npz.parent == partial.resolve() and npz.name == name.removesuffix('.nii.gz')+'.npz' and digest(npz)==checksum
        code = f'P{i:03d}'
        itk = inp / (code+'_0000.nii.gz')
        # NPZ members are inflated independently. No label member is opened by this program.
        with np.load(npz) as arr:
            ct = arr['ct_hu']; affine = arr['affine']
        assert ct.ndim==3 and ct.dtype==np.int16 and np.allclose(affine[:3,:3], np.eye(3)*2)
        obj = nib.Nifti1Image(ct,affine)
        obj.header.set_xyzt_units('mm'); obj.set_sform(affine,1); obj.set_qform(affine,1)
        nib.save(obj,itk)
        recovered = nib.load(itk)
        assert np.allclose(recovered.affine,affine,atol=1e-5,rtol=0) and np.array_equal(ct,np.asanyarray(recovered.dataobj))
        series.append({'case':code,'volume':name,'image_path':str(itk),'sha256':digest(itk),
                       'source_npz':str(npz),'source_npz_sha256':checksum})
    assert {x.name for x in inp.iterdir()}=={r['case']+'_0000.nii.gz' for r in series}
    put(run/'input_ledger.private.json',series)
    put(run/'design.json',{'n':6,'selection':'all six custodian-approved training positives of first nine; ordered by deterministic prespecified hash',
        'manifest_sha256':digest(manifest),'input':'only ct_hu and affine; released 2 mm RAS+ grid',
        'locked_before_outcomes':True,'qc':'6 of first 9 passed independent geometry and overlay QC before seeing any Basel outcomes; 3 preexcluded',
        'coverage':'intersection divided by expert annotated positive voxels only. Outside annotated foreground is unknown.',
        'device':'physical GPU 1', 'resampled_ct_grid':True})
    prev = root/'phase1/ai_contours'
    archive = prev/'model.verified.zip'
    provenance = json.loads((prev/'model_provenance.json').read_bytes())
    assert provenance['weights_archive_md5']==WEIGHT_MD5 and digest(archive,'md5')==WEIGHT_MD5
    model = prev/'model_files'/BASE
    with ZipFile(archive) as z:
        for folder in range(5):
            for ext in ('','.pkl'):
                member = f'fold_{folder}/model_final_checkpoint.model{ext}'
                with z.open(BASE+member) as src:
                    from hashlib import file_digest
                    assert file_digest(src,'sha256').hexdigest()==digest(model/member)
    return inp,model,series


def infer(root):
    run = root/'phase1/ai_contours/partial_six'; run.mkdir(exist_ok=True); os.chdir(run)
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='1' and not (run/'prediction_lock.json').exists()
    for k in ['nnUNet_raw_data_base','nnUNet_preprocessed','RESULTS_FOLDER']:
        os.environ[k]=str(root/'phase1/ai_contours/nnunet_runtime'/k)
    os.environ['TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD']='1'
    import torch
    from nnunet.inference.predict import predict_from_folder
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark=False
    inputs,model,cases=prepare(root,run)
    start=time.monotonic()
    print('Six image-only inputs verified; running locked Basel v1 five-fold ensemble.',flush=True)
    predict_from_folder(str(model),str(inputs),'prediction_raw',[0,1,2,3,4],False,1,1,None,0,1,True,
                        mixed_precision=True,overwrite_existing=True,mode='normal',overwrite_all_in_gpu=False,
                        step_size=.5,checkpoint_name='model_final_checkpoint')
    convert(cases)
    put(run/'prediction_lock.json',{'files':{p.name:digest(p) for p in sorted((run/'ai_binary').glob('*.nii.gz'))},
          'cohort_identity':digest(run/'input_ledger.private.json'),'class_mapping':{'2':'pleural effusion'},
          'data_input_only':'CT, no report text or lesion voxels','inference_QC_seconds':round(time.monotonic()-start,2)})
    print('All six predictions sealed.',flush=True)


if __name__=='__main__':
    os.umask(0o077)
    infer(Path(sys.argv[1]).resolve())
