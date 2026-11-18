"""Training-only ReX finding-grid audit for the four locked engineering cases."""
import argparse
import ast
from collections import Counter
import csv
import hashlib
import json
import os
from pathlib import Path
import re

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import nibabel as nib
from nibabel.processing import resample_to_output, resample_from_to
import numpy as np
from scipy.ndimage import binary_erosion


SUBMISSION_CODE = 'https://github.com/rajpurkarlab/ReXGroundingCT/blob/28998f58d833b3cd378277d2a49213154be72d06/ct-rate-grounding-main/src/format_data_for_upload.py#L8-L42'
REX = '662d836108a807d47a40421ac5ed4d1c7e133b16'
CT = '546640db5d4674f6e6b36d40d2cfcd57c8b39e21'


def file_hash(path):
    with open(path, 'rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def check_ct(original, fixed, meta):
    # These tests apply to the four reserved reconstructions, not all CT-RATE contracts.
    if len(original.shape) != 3 or original.shape != fixed.shape:
        raise ValueError('The two reconstructions do not share matrix dimensions')
    slope, intercept = (float(meta[k]) for k in ['RescaleSlope', 'RescaleIntercept'])
    if original.dataobj.slope != 1 or original.dataobj.inter != 0:
        raise ValueError('Unexpected v1 header scaling')
    if fixed.header.get_xyzt_units()[0] not in ('unknown','mm'):
        raise ValueError('Unexpected spatial unit')
    origin = np.asarray(ast.literal_eval(meta['ImagePositionPatient']), dtype=float)
    rowcol = np.asarray(ast.literal_eval(meta['ImageOrientationPatient']), dtype=float).reshape(2, 3)
    drow, dcol = ast.literal_eval(meta['XYSpacing'])
    aff = np.eye(4)
    aff[:3,:3] = np.column_stack((rowcol[0]*dcol, rowcol[1]*drow,
                                  np.cross(*rowcol)*float(meta['ZSpacing'])))
    aff[:3,3] = origin
    aff[:2,:] *= -1
    affine_error = float(np.max(np.abs(aff - fixed.affine)))
    if not np.isfinite(affine_error) or affine_error > 0.01:
        raise ValueError('Header and source CSV DICOM geometry disagree; review reconstruction')
    native = np.asanyarray(original.dataobj)
    corrected = np.asanyarray(fixed.dataobj)
    maxdiff = 0.
    for z in range(0, native.shape[2], 8):
        raw = native[...,z:z+8] * slope + intercept
        diff = np.max(np.abs(raw - corrected[...,z:z+8]))
        if not np.isfinite(diff):
            raise ValueError('Non-finite calibration check')
        maxdiff = max(maxdiff,float(diff))
    if maxdiff > 0.51:
        raise ValueError('v1 calibration and fixed data disagree at matched voxel indices')
    return corrected, dict(v1_hu_vs_fixed_max_abs=maxdiff, all_native_voxels_compared=int(native.size),
                           fixed_affine_vs_csv_max_abs=affine_error,
                           fixed_native_axes=''.join(nib.aff2axcodes(fixed.affine)),
                           fixed_spacing_mm=nib.affines.voxel_sizes(fixed.affine).tolist(),
                           fixed_header_declared_units=fixed.header.get_xyzt_units()[0])


def reduce_ct(img, spacing):
    # Use float input during interpolation; pad outside source FOV with exterior air.
    wor = nib.Nifti1Image(np.asarray(img.dataobj,dtype=np.float32),img.affine)
    out = resample_to_output(wor,(spacing,)*3, order=1, mode='constant', cval=-1024.)
    assert nib.aff2axcodes(out.affine) == ('R','A','S')
    ct = np.rint(np.asarray(out.dataobj)).clip(-32768,32767).astype('int16')
    return ct, out.affine


def reduce_mask(data, source_affine, output_grid):
    im = nib.Nifti1Image(np.asarray(data>0,dtype='uint8'),source_affine)
    return np.asarray(resample_from_to(im, output_grid, order=0, cval=0).dataobj,dtype='uint8')


def preprocess_finding(corrected_ct, rex_mask, finding_index, spacing_mm=2.0):
    """Return CT HU/int16, finding-positive/uint8 and affine, XYZ on one RAS+ grid.

    Input must be a CT-RATE *_fixed reconstruction and its paired released ReX
    F,H,W,D file. The implementation uses the provisional native-index mapping
    evaluated on the reserved T cases. Output 0 means outside this finding's
    contour; completeness and all target-negative voxels remain unadjudicated.
    Callers should ensure that the patient/series IDs match and that the use is
    permitted by their data split before reading voxels.
    """
    ct_img, seg = nib.load(corrected_ct), nib.load(rex_mask)
    if len(ct_img.shape) != 3 or len(seg.shape) != 4 or seg.shape[1:] != ct_img.shape:
        raise ValueError('ReX FHWD and CT HWD do not match')
    if Path(corrected_ct).name != Path(rex_mask).name:
        raise ValueError('The two input volume names must agree')
    if nib.aff2axcodes(ct_img.affine) != ('L','P','S') or np.allclose(ct_img.affine, np.eye(4)):
        raise ValueError('Only validated native LPS corrected geometry is supported')
    if not isinstance(finding_index, (int,np.integer)) or not 0 <= finding_index < seg.shape[0]:
        raise ValueError('Invalid integer ReX finding index')
    if not .5 <= spacing_mm <= 5:
        raise ValueError('Spacing must be in the supported 0.5 to 5 mm range')
    ct_hu, affine = reduce_ct(ct_img, float(spacing_mm))
    source_mask = np.asanyarray(seg.dataobj[finding_index, :, :, :])
    if not np.any(source_mask>0):
        raise ValueError('Empty source finding; positive contour unavailable')
    positive = reduce_mask(source_mask, ct_img.affine, (ct_hu.shape, affine))
    if not np.any(positive):
        raise ValueError('Positive contour lost at this spacing')
    return ct_hu, positive, affine


def checked_paths(audit, fixed_index, roots):
    paths = [roots[0]/'ct_rate'/audit['ct_release_path'],roots[0]/'rex'/audit['mask_release_path'],
             roots[1]/'corrected'/fixed_index['path']]
    for path, expected in zip(paths,[audit['ct_sha256'],audit['mask_sha256'],fixed_index['sha256']]):
        if file_hash(path) != expected:
            raise ValueError('Source file SHA-256 mismatch')
    return paths


def overlay(ct, masks, alternate, labels, output):
    fig,axes=plt.subplots(len(masks),3,figsize=(9.6,2.8*len(masks)),squeeze=False)
    for i,(m,alt) in enumerate(zip(masks,alternate)):
        for ax,seg,title in zip(axes[i],(None,m,alt),('CT','Native index','Alternate XY flip')):
            # Both transforms leave superior/inferior index unchanged.
            z=int(np.argmax(m.sum(axis=(0,1))))
            ax.imshow(ct[:,:,z].T,origin='lower',cmap='gray',vmin=-1000,vmax=250)
            if seg is not None:
                rat=seg[:,:,z].astype(bool)
                edge=np.ma.masked_where(~(rat ^ binary_erosion(rat)), np.ones_like(rat))
                ax.imshow(edge.T,origin='lower',cmap='autumn',vmin=0,vmax=1,interpolation='nearest')
            ax.set_title(f'F{i} ({labels[i]}) | {title}',fontsize=9)
            ax.set_xlabel('patient L  ->  patient R',fontsize=8)
            ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(output,dpi=125)
    plt.close(fig)


def candidate_diagnostic(mask, img, affine, text):
    indices=np.argwhere(mask)
    hu = img[tuple(indices.T)]
    # A scan-centre heuristic for lateral, explicitly unilateral annotations only.
    # Patient right in this array is negative i; all four official fixed volumes are LPS.
    d=(indices[:,0] - (img.shape[0]-1)/2)*affine[0,0]
    left = bool(re.search(r'\bleft\b',text,flags=re.I)); right=bool(re.search(r'\bright\b',text,flags=re.I))
    unilateral = left != right
    if unilateral:
        fraction = float(np.mean(d < 0 if left else d > 0))
    return dict(mentions_one_side=unilateral, fraction_on_mentioned_side=fraction if unilateral else None,
                median_signed_x_from_fov_center_mm=float(np.median(d)),
                median_CT_HU_under_mask=float(np.median(hu)),fraction_outside_scan_padding=float(np.mean(hu<=-2048)))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--project',type=Path,required=True)
    p.add_argument('--spacing',type=float,default=2.)
    args=p.parse_args()
    if not 0.5<=args.spacing<=5: p.error('Use a reasonable low-resolution spacing in mm')
    os.umask(0o077)
    raw=args.project/'data/four_train';folder=args.project/'phase1/geometry'
    private=folder/'cache_private';private.mkdir(parents=True,exist_ok=True)
    lock=json.loads((raw/'pilot_T_reservation.json').read_text())
    names=lock['volumes']
    if len(set(names))!=4 or len(names)!=4 or lock['role']!='T_paid_engineering_pilot':
        raise ValueError('Only the immutable four-patient T engineering reserve is supported')
    downloaded=json.loads((raw/'four_header_audit.json').read_text())
    assert downloaded['repository_revisions']=={'ibrahimhamamci/CT-RATE':CT,'rajpurkarlab/ReXGroundingCT':REX}
    cases={i['name']:i for i in downloaded['cases']}
    fixed={i['name']:i for i in json.loads((folder/'corrected.private.json').read_text())}
    if set(names)!=set(cases) or set(names)!=set(fixed):
        raise ValueError('Release records do not exactly match the locked cases')
    rex=json.loads((args.project/'metadata/rex/dataset.json').read_text())
    findings={row['name']:row for row in rex['train'] if row['name'] in names}
    del rex
    table={}
    for f in ['train_metadata.csv','validation_metadata.csv']:
        with open(raw/'ct_rate/dataset/metadata'/f,encoding='utf-8-sig',newline='') as stream:
            for row in csv.DictReader(stream):
                if row['VolumeName'] in names:
                    if row['VolumeName'] in table: raise ValueError('Duplicate metadata')
                    table[row['VolumeName']]=row
    output=[]
    for ci,name in enumerate(names,1):
        code=f'T{ci:02}'
        record=cases[name];ff=fixed[name]; rr=findings[name]
        v1,label,v2=checked_paths(record,ff,(raw,folder))
        v1,mask,fixedimg=(nib.load(p) for p in (v1,label,v2))
        data,ck=check_ct(v1,fixedimg,table[name])
        if nib.aff2axcodes(fixedimg.affine)!=('L','P','S'):
            raise ValueError('Revisit laterality diagnostic for this series')
        if mask.shape != (len(rr['findings']),*fixedimg.shape):
            raise ValueError('Invalid FHWD layout')
        mdata=np.asanyarray(mask.dataobj)
        labels=[]; masks=[]; alternates=[];checks=[]
        ct,aff=reduce_ct(fixedimg,args.spacing); grid=(ct.shape,aff)
        for ci in range(mdata.shape[0]):
            idx=str(ci);m=mdata[ci]; alt=m[::-1,::-1,:]
            pixels=int(np.count_nonzero(m))
            if pixels<=0 or pixels!=int(rr['pixels'][idx]):
                raise ValueError('Finding voxel count does not match the official ReX channel index')
            labels.append(rr['categories'][idx])
            entry={'finding':ci,'code':labels[-1],'voxels_original':pixels}
            for data_mask, tag, dest in [(m,'native_index',masks),(alt,'alternative_double_flip',alternates)]:
                entry[tag]=candidate_diagnostic(data_mask>0,data,fixedimg.affine,rr['findings'][idx])
                dest.append(reduce_mask(data_mask,fixedimg.affine,grid))
            entry['positive_resampled_voxels']=int(masks[-1].sum());checks.append(entry)
        assert len(masks)==len(labels)
        np.savez_compressed(private/(code+'.npz'),ct_hu=ct,positive_masks=np.stack(masks),
                            category=np.array(labels),finding_index=np.arange(len(labels)),
                            affine_ras=aff,spacing_mm=np.array([args.spacing]*3))
        overlay(ct,masks,alternates,labels,private/(code+'_review.png'))
        output.append(dict(case=code,native_shape=list(fixedimg.shape),output_shape=list(ct.shape),**ck,channels=checks))
        print(code,'ok',len(masks),flush=True)
    scores=[m for r in output for m in r['channels'] if m['native_index']['mentions_one_side']]
    summ=dict(patients=len(names),findings=sum(len(x['channels']) for x in output),
              category_counts=dict(Counter(m['code'] for x in output for m in x['channels'])),
              native_CT_voxels_compared=sum(x['all_native_voxels_compared'] for x in output),
              max_element_error_fixed_affine_vs_DICOM_csv=max(x['fixed_affine_vs_csv_max_abs'] for x in output),
              native_spacings_mm=[x['fixed_spacing_mm'] for x in output],
              positive_voxels_2mm_min=min(m['positive_resampled_voxels'] for x in output for m in x['channels']) if args.spacing==2.0 else None,
              positive_voxels_2mm_max=max(m['positive_resampled_voxels'] for x in output for m in x['channels']) if args.spacing==2.0 else None,
              unilateral_findings=len(scores),
              unilateral_native_agree=sum(m['native_index']['fraction_on_mentioned_side']>=.8 for m in scores),
              unilateral_double_flip_agree=sum(m['alternative_double_flip']['fraction_on_mentioned_side']>=.8 for m in scores),
              max_absolute_v1_calibrated_vs_fixed_HU=max(x['v1_hu_vs_fixed_max_abs'] for x in output),
              spacing_mm=args.spacing, coordinate_reliability='CT verified against source CSV geometry and full-grid intensity. ReX finding masks use provisional native CT indices. X evidence covers explicit unilateral cases only; no external physical transform for labels or independent clinical adjudication of AP/SI, boundary or effusion completeness.',
              derived_grid='Axes XYZ in RAS+, millimetres; linear CT interpolation and nearest-neighbour binary finding positives. Outside positive masks is unknown. No dense negatives or complete effusion labels produced.',
              qa='Candidate comparison of native and double XY flip on axial CT; charts and NPZ stay private on instance. Never select transform by fitting a held-out mask.',
              rex_upload_source=SUBMISSION_CODE, ct_revision=CT,rex_revision=REX)
    (folder/'geometry_cases.private.json').write_text(json.dumps(output,indent=2)+'\n')
    (folder/'geometry_summary.json').write_text(json.dumps(summ,indent=2)+'\n')
    (private/'case_map.private.json').write_text(json.dumps({f'T{i:02}':n for i,n in enumerate(names,1)},indent=2)+'\n')
    print(json.dumps(summ))

if __name__=='__main__': main()
