"""Score only the independently annotated positive support after prediction lock."""
import csv
import json
import os
import sys
from pathlib import Path

import nibabel as nib
import numpy as np

from basel_four import digest, put


def score(run):
    locked=json.loads((run/'prediction_lock.json').read_text())
    source=run/'input_ledger.private.json'
    assert locked['cohort_identity']==digest(source)
    ledger=json.loads(source.read_text())
    binroot=run/'ai_binary'
    assert len(ledger)==len(locked['files'])==6
    assert {p.name:digest(p) for p in binroot.glob('*.nii.gz')}==locked['files']
    design=json.loads((run/'design.json').read_text())
    selection=run/'selection_qc_snapshot.private.json'
    assert design['manifest_sha256']==digest(selection)
    cohort=json.loads(selection.read_text())
    assert set(cohort['usable'])=={r['volume'] for r in ledger}
    records=[]
    for item in ledger:
        pred_file=binroot/(item['case']+'_AI_effusion.nii.gz')
        prediction=nib.load(pred_file)
        allowed=cohort['cases'][item['volume']]
        npz=Path(item['source_npz'])
        assert allowed['status']=='usable' and digest(npz)==allowed['npz_sha256']==item['source_npz_sha256']
        with np.load(npz) as pack:
            true_data,aff=pack['positive_mask'],pack['affine']
        out=np.asanyarray(prediction.dataobj)
        assert true_data.shape==out.shape and np.allclose(aff,prediction.affine,atol=1e-5,rtol=0)
        assert np.isin(true_data,[0,1]).all() and np.isin(out,[0,1]).all()
        reference=int(np.count_nonzero(true_data)); assert reference>0
        model=int(np.count_nonzero(out))
        overlap=int(np.count_nonzero(np.logical_and(true_data,out)))
        mm3=abs(np.linalg.det(aff[:3,:3])); assert np.isclose(mm3,8.)
        records.append({'case':item['case'],'known_positive_voxels':reference,'ai_foreground_voxels':model,
            'known_positive_covered_voxels':overlap,'known_positive_coverage':overlap/reference,
            'known_positive_volume_ml':reference*mm3/1000,'ai_volume_ml':model*mm3/1000})
    ratios=[r['known_positive_coverage'] for r in records]
    assessment={'cases':records,'eligibility':{'candidates_reviewed':9,'qc_pass_before_model':6,'qc_excluded_before_model':3},
        'summary':{'patient_mean_coverage':float(np.mean(ratios)),'patient_median_coverage':float(np.median(ratios)),
        'voxel_weighted_coverage':sum(r['known_positive_covered_voxels'] for r in records)/sum(r['known_positive_voxels'] for r in records),
        'patients_with_any_overlap':sum(x>0 for x in ratios),'patients_with_nonempty_ai':sum(r['ai_foreground_voxels']>0 for r in records)},
        'definition':'|AI foreground intersect known expert positive| divided by |known expert positive| for each patient',
        'scope':'QC-selected training positives on a 2 mm RAS+ derivative; unannotated regions are unknown. No population, whole-volume or external-test estimate.',
        'prediction_lock_sha256':digest(run/'prediction_lock.json'),'qc_snapshot_sha256':digest(selection)}
    put(run/'known_positive_coverage.json',assessment)
    with (run/'known_positive_coverage.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=records[0].keys()); w.writeheader(); w.writerows(records)
    print(json.dumps(assessment,indent=2),flush=True)


if __name__=='__main__':
    os.umask(0o077)
    score(Path(sys.argv[1]).resolve())
