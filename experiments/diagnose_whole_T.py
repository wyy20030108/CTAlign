"""Frozen CT-only full-field diagnostics in the remainder of the engineering-QC T group."""
import argparse
import csv
import hashlib
import itertools
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch
from audit_pairs import identity
from fetch_four import sha256, write_json
from volume_benchmark import ReportUNet


PARTIAL=Path('phase1/baselines/partial_v1')
MATCH=Path('phase1/baselines/matched_two_crop_v1')
FULL=Path('phase1/full_report_trainonly')
DATA=Path('phase1/data/T_partial')
MODELS=('weak_off','weak_on','full_empty','full_paired','full_matched')
PROTO={'window_xyz':[64]*3,'stride_xyz':[48]*3,'probability_threshold':.5,'CT_grid':'2 mm isotropic native compact RAS+',
       'normalization':'clip HU to [-1000,200] then scale to [0,1]; train-time augmentation disabled',
       'tile_rule':'start at 0; regular stride 48, append n-64 if last window would leave uncovered field; pad only short axes with air and crop back',
       'merge':'equal tile weights for arithmetic mean of sigmoid probabilities, classify at >=0.5, entire CT field incl air; no postprocessing',
       'coverage':'intersection of prediction and available local positive annotation divided by annotated positive count; unmarked voxels unknown',
       'role':'exploration entirely in T; no optimization here, no R1/R2/E or external data, incomplete reference'}


def read(p):
    return json.loads(p.read_text())


def digest(x):
    return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def windows(n):
    if n<=0:
        raise ValueError('Empty axis')
    n=max(n,64); ends=list(range(0,n-63,48))
    if ends[-1]!=n-64:
        ends.append(n-64)
    return ends


def make_inputs():
    final_path=DATA/'qc_release.private.json'; final=read(final_path)
    lock=read(PARTIAL/'input_lock.private.json'); base=read(PARTIAL/'run_config.json')
    matched=read(MATCH/'run_config.json'); t_csv=Path('phase1/cohort/split_v1/b10/T.csv')
    with t_csv.open(newline='') as f:
        allowed={r['volume'] for r in csv.DictReader(f)}
    finalnames=set(final['usable']); tused={r['volume'] for r in lock['roles']['T']}; allused=[r['volume'] for rows in lock['roles'].values() for r in rows]
    used_patients={identity(n)[1] for n in allused}
    if (final['status']!='final' or final['split_sha256']!=sha256(t_csv) or len(finalnames)!=13 or len(tused)!=6 or
        len(used_patients)!=14 or not tused<=finalnames or not finalnames<=allowed or
        finalnames&set(final['excluded']) or digest(lock)!=base['inputs_sha256'] or
        matched['inputs_sha256']!=digest(lock) or read(MATCH/'verification.json')['status']!='passed'):
        raise ValueError('Unexpected training roles or incompatible frozen manifests')
    names=sorted(finalnames-tused,key=lambda n:digest(['T7',n]))
    if len({identity(n)[1] for n in names}-used_patients)!=7:
        raise ValueError('Cross-patient exclusion failed')
    cases=[]; records=[]
    for index,n in enumerate(names,1):
        info=final['cases'][n]; target=DATA/(n.removesuffix('.nii.gz')+'.npz'); note=read(target.with_suffix('.json'))
        if (info['status']!='usable' or sha256(target)!=info['npz_sha256'] or note['npz_sha256']!=info['npz_sha256'] or
            note['name']!=n or note['channels']!=info['channels']):
            raise ValueError('QC or mask channel evidence does not match derivative')
        with np.load(target,allow_pickle=False) as d:
            ct,mask,aff=d['ct_hu'],d['positive_mask'],d['affine']
        voxel_mm3=float(abs(np.linalg.det(aff[:3,:3])))
        if (ct.ndim!=3 or ct.dtype!=np.int16 or ct.shape!=mask.shape or mask.max()!=1 or mask.min()!=0 or
            int(mask.sum())!=note['positive_voxels'] or not np.isfinite(aff).all() or
            not np.allclose(aff[-1],[0,0,0,1]) or not np.allclose(aff[:3,:3],np.eye(3)*2,atol=1e-5) or
            not np.isclose(voxel_mm3,8,atol=1e-5)):
            raise ValueError('Unverified spatial geometry, calibration, or foreground count')
        orig=ct.shape; dims=tuple(max(v,64) for v in orig)
        starts=[windows(v) for v in orig]; hits=np.ones(dims,dtype=np.uint16)
        for ax,(dimension,s) in enumerate(zip(dims,starts,strict=True)):
            counts=np.zeros(dimension,dtype=np.uint16)
            for lower in s:
                counts[lower:lower+64]+=1
            shape=[1,1,1];shape[ax]=dimension; hits*=counts.reshape(shape)
        if hits.min()<1:
            raise ValueError('Window schedule leaves some voxels uncovered')
        code=f'Tdiag{index:02}'
        records.append({'code':code,'volume':n,'file_sha256':info['npz_sha256'],'overlay_sha256':info['overlay_sha256'],
                       'CT_original_sha256':note['ct_sha256'],'ReX_original_sha256':note['mask_sha256'],
                       'channels':info['channels'],'shape_xyz':list(orig),'affine_xyz_to_RAS_mm':aff.tolist(),
                       'voxel_volume_mm3':voxel_mm3,'known_positive_voxels':int(mask.sum()),'axis_window_starts':starts,
                       'number_of_windows':int(np.prod([len(v) for v in starts])),'coverage_min':int(hits.min()),'coverage_max':int(hits.max())})
        cases.append((ct,mask.astype(bool),hits,starts))
    return cases,{'cases':records,'T_split_sha256':sha256(t_csv),'final_QC_sha256':sha256(final_path),
                  'excluded_roles_lock_sha256':digest(lock),'excluded_patient_count_T':6,'excluded_patient_count_U':8,
                  'matched_batches_digest':matched['actual_batches_digest'],'protocol':PROTO}


def get_model(label,plan):
    if label in MODELS[:2]:
        path=MATCH/(label+'.pt'); expected=read(MATCH/(label+'_metrics.json'))['checkpoint_sha256']
    else:
        arm=label.removeprefix('full_'); folder=FULL/'three_arms'; path=folder/(arm+'.pt')
        expected=read(folder/'summary.json')['arms'][arm]['saved_sha256']
    if sha256(path)!=expected:
        raise ValueError('Frozen checkpoint integrity failed')
    ck=torch.load(path,map_location='cpu',weights_only=True)
    exposure={'inference_model':'student','reports_input_at_inference':0}
    if label in MODELS[:2]:
        if (ck['arm']!=label or ck['common']['inputs_sha256']!=plan['excluded_roles_lock_sha256'] or
            ck['common']['actual_batches_digest']!=plan['matched_batches_digest']):
            raise ValueError('Wrong original-model training patients/batches')
        net=ReportUNet(); full=False
    else:
        design=ck['design']; arm=label.removeprefix('full_'); mod=FULL/'full_report_model.py'
        if (ck['arm']!=arm or design['cohort_sha256']!=plan['excluded_roles_lock_sha256'] or
            design['actual_tensor_ledger_sha256']!=plan['matched_batches_digest'] or design['model_code_sha256']!=sha256(mod) or
            design['vocab_size']!=ck['vocab_size']):
            raise ValueError('Wrong full-report model, patients or batches')
        # Read existing donor identifiers only, without opening any case report.
        drawn=read(FULL/'three_arms/U_donors.private.json')['draw']
        if digest(drawn)!=design['frozen_donor_sequence_sha256']:
            raise ValueError('Wrong frozen donor exposure ledger')
        diag={identity(r['volume'])[1] for r in plan['cases']}
        train_lock=read(PARTIAL/'input_lock.private.json')
        with Path('phase1/cohort/split_v1/b10/U_train.csv').open(newline='') as f:
            fit_names={r['volume'] for r in csv.DictReader(f)} | {r['volume'] for r in train_lock['roles']['T']}
        fit_sha=hashlib.sha256('\n'.join(sorted(fit_names)).encode()).hexdigest()
        fit_audit=read(FULL/'fulltext_validation.json')
        if (len(fit_names)!=2149 or fit_sha!=fit_audit['training_vocabulary_roster_sha256'] or
            {identity(n)[1] for n in fit_names}&diag or sha256(FULL/'vocabulary.private.json')!=design['vocab_sha256']):
            raise ValueError('Clean full-report roster or vocabulary could not be verified')
        donorids={identity(n)[1] for n in drawn if n is not None}
        exposure.update(diagnostic_reports_in_training_pool_vocabulary_fit=0,vocabulary_roster_sha256=fit_sha,
                        diagnostic_reports_actually_used_as_U_donor=len(diag&donorids) if arm=='matched' else 0)
        sys.path.insert(0,str(FULL.resolve()))
        from full_report_model import FullReportUNet
        net=FullReportUNet(ck['vocab_size']); full=True
    net.load_state_dict(ck['student'],strict=True)
    return net.cuda().eval().requires_grad_(False),full,expected,exposure


def score(pred,known,dvoxel):
    k=int(known.sum()); n=int(pred.sum()); h=int(np.count_nonzero(pred & known)); volume=k*dvoxel/1000
    if not k:
        raise ValueError('The denominator must have published positive voxels')
    return {'known_positive_voxels':k,'predicted_positive_voxels':n,'intersection_voxels':h,
            'known_positive_coverage_fraction':h/k,'all_known_positive_missed':h==0,
            'predicted_volume_mL':n*dvoxel/1000,'known_volume_mL':volume,'predicted_fraction_full_FOV':n/pred.size,
            'whole_FOV_voxels':pred.size,'FOV_volume_mL':pred.size*dvoxel/1000}


@torch.inference_mode()
def run_volume(net,full,case,rec,batch_size=4):
    ct,_,expected,starts=case; original=ct.shape
    x=np.pad(ct,[(0,max(0,64-n)) for n in original],constant_values=-1024) if min(original)<64 else ct
    bounds=tuple(slice(0,n) for n in original); coords=list(itertools.product(*starts))
    if len(coords)!=rec['number_of_windows']:
        raise ValueError('Inconsistent window count')
    prob_sum=np.zeros(x.shape,dtype=np.float32); seen=np.zeros(x.shape,dtype=np.uint16)
    for j in range(0,len(coords),batch_size):
        crops=[tuple(slice(int(i),int(i)+64) for i in xyz) for xyz in coords[j:j+batch_size]]
        array=np.stack([x[sl] for sl in crops]).astype(np.float32)
        array=np.clip((array+1000)/1200,0,1)
        xb=torch.from_numpy(array).cuda()[:,None]
        ctx=None if full else torch.zeros((len(crops),64),dtype=torch.long,device='cuda')
        values=net(xb,ctx).sigmoid().squeeze(1).cpu().numpy()
        if values.shape!=array.shape or not np.isfinite(values).all():
            raise ValueError('Unexpected model prediction')
        for sl,prob in zip(crops,values,strict=True):
            prob_sum[sl]+=prob; seen[sl]+=1
    if not np.array_equal(seen,expected):
        raise ValueError('Accumulated full-grid coverage differs from independent axis product')
    merged=prob_sum[bounds]/seen[bounds]
    return np.ascontiguousarray(merged>=PROTO['probability_threshold'])


def freeze(path,obj):
    if path.exists():
        if read(path)!=obj:
            raise ValueError('Frozen manifest changed')
    else:
        write_json(path,obj)


def verify_model(output,label,plan,cases,ck_sha):
    report=read(output/(label+'.json'))
    if report['checkpoint_sha256']!=ck_sha or report['input_digest']!=digest(plan) or len(report['cases'])!=len(cases):
        raise ValueError('Invalid diagnostic report')
    for (ct,known,_,_),rec,row in zip(cases,plan['cases'],report['cases'],strict=True):
        file=output/'predictions_private'/label/(rec['code']+'.npz')
        if row['code']!=rec['code'] or sha256(file)!=row['prediction_file_sha256']:
            raise ValueError('Saved prediction identity or hash mismatch')
        with np.load(file,allow_pickle=False) as v:
            if tuple(v['shape_xyz'])!=ct.shape or not np.array_equal(v['affine_xyz_to_RAS_mm'],np.array(rec['affine_xyz_to_RAS_mm'])):
                raise ValueError('Saved prediction grid differs from source grid')
            pred=np.unpackbits(v['positive_bitpacked'],bitorder='little',count=ct.size).reshape(ct.shape).astype(bool)
        d=float(abs(np.linalg.det(np.asarray(rec['affine_xyz_to_RAS_mm'])[:3,:3])))
        results=score(pred,known,d)
        if any(row[k]!=value for k,value in results.items()):
            raise ValueError('Published counts, units, or fractions fail recomputation')
    return report['summary']


def run_model(output,label,plan,cases,verify_only):
    net,full,ck_hash,exposure=get_model(label,plan)
    target=output/(label+'.json')
    if verify_only or target.exists():
        return verify_model(output,label,plan,cases,ck_hash)
    start=time.perf_counter(); dest=output/'predictions_private'/label; dest.mkdir(parents=True,exist_ok=True)
    rows=[]
    for case,rec in zip(cases,plan['cases'],strict=True):
        pred=run_volume(net,full,case,rec); path=dest/(rec['code']+'.npz')
        np.savez_compressed(path,positive_bitpacked=np.packbits(pred,bitorder='little'),shape_xyz=np.array(pred.shape),affine_xyz_to_RAS_mm=np.array(rec['affine_xyz_to_RAS_mm']))
        r=score(pred,case[1],rec['voxel_volume_mm3']); r.update(code=rec['code'],prediction_file_sha256=sha256(path),window_count=rec['number_of_windows'],
                                                                 grid_coverage_min=rec['coverage_min'],grid_coverage_max=rec['coverage_max'])
        rows.append(r)
    summary={'patients':len(rows),'patient_mean_known_positive_coverage':float(np.mean([r['known_positive_coverage_fraction'] for r in rows])),
             'pooled_known_positive_coverage':sum(r['intersection_voxels'] for r in rows)/sum(r['known_positive_voxels'] for r in rows),
             'patients_all_known_positive_missed':sum(r['all_known_positive_missed'] for r in rows),
             'patient_mean_predicted_fraction_full_FOV':float(np.mean([r['predicted_fraction_full_FOV'] for r in rows])),
             'median_predicted_volume_mL':float(np.median([r['predicted_volume_mL'] for r in rows])),
             'range_predicted_volume_mL':[min(r['predicted_volume_mL'] for r in rows),max(r['predicted_volume_mL'] for r in rows)]}
    write_json(target,{'model':label,'input_digest':digest(plan),'checkpoint_sha256':ck_hash,'exposure':exposure,
                      'patient_weighting':'equal patient weight unless explicitly pooled','cases':rows,'summary':summary,
                      'elapsed_seconds':time.perf_counter()-start})
    return verify_model(output,label,plan,cases,ck_hash)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--models',nargs='+',choices=MODELS,default=list(MODELS));p.add_argument('--verify-only',action='store_true')
    p.add_argument('--output',type=Path,default=Path('phase1/cross_patient_T7'))
    a=p.parse_args(); os.umask(0o077)
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='0':
        raise ValueError('Reserve GPU 1')
    a.output.mkdir(parents=True,exist_ok=True);torch.set_num_threads(2)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
    cases,plan=make_inputs();freeze(a.output/'input_manifest.private.json',plan)
    freeze(a.output/'inference_protocol.json',{'rules':PROTO,'code_sha256':sha256(Path(__file__)),'torch':str(torch.__version__),
           'cuda':torch.version.cuda,'GPU':torch.cuda.get_device_name(),'batch_windows':4,'inputs_digest':digest(plan),
           'physical_voxel_mL':.008,'diagnostic_cases':len(cases),'windows_per_patient':[r['number_of_windows'] for r in plan['cases']]})
    summaries={}
    for label in a.models:
        summaries[label]=run_model(a.output,label,plan,cases,a.verify_only)
        print('verified' if a.verify_only else 'completed',label,flush=True)
    if all((a.output/(m+'.json')).is_file() for m in MODELS):
        all_reports={m:read(a.output/(m+'.json')) for m in MODELS}
        result={'role':PROTO['role'],'inputs_digest':digest(plan),'protocol_sha256':sha256(a.output/'inference_protocol.json'),
                'models':{m:r['summary'] for m,r in all_reports.items()},'source_coverage':{m:r['exposure'] for m,r in all_reports.items()},
                'case_results':{m:r['cases'] for m,r in all_reports.items()},'limitations':'All seven are selected T cases with incomplete positive findings. Full FOV includes tissue and air. Predicted foreground volume and its fraction cannot identify true negatives or boundary errors.'}
        freeze(a.output/'summary.json',result)
    if a.verify_only:
        freeze(a.output/'verification.json',{'models_recounted':list(a.models),'status':'passed','role_inputs_digest':digest(plan),
               'independent_saved_map_to_source_grid_positive_counts_recounted':len(a.models)*len(cases)})


if __name__=='__main__':
    main()
