"""Audit cohort independence, lock timing, and per-case arithmetic before release."""

from itertools import combinations
import math
import os
from pathlib import Path

from ctalign_eval import CALIBRATION, FRESH, FROZEN, OUT, PLAN, PRIOR, SEEDS, reference_rows
from ctalign_fresh import CANDIDATE, MANIFEST, PREVIOUS
from ctalign_size import ROOT as SIZE_ROOT
from ctalign_summary import read_scores
from ctalign_structured import LOCK as STRUCTURED_LOCK, OUT as STRUCTURED
from frozen_priors import read, sha, write
from joint_nodule_run import COHORT
from thesis_model import GRID


ARMS = ('AS', 'AN', 'AU', 'AP', 'BS', 'BN', 'BU', 'BP', 'BR', 'off_U', 'off_P', 'constant_U', 'constant_P')


def patients(rows):
    keys = {tuple(r['patient'].split(':')) for r in rows}
    assert len(keys) == len(rows)
    return keys


def checks(data, total):
    n, hit = data['pred'], data['intersection']
    assert isinstance(n, int) and isinstance(hit, int) and 0 <= hit <= min(total, n)
    assert total > 0 and data['miss'] == (hit == 0)
    expected = dict(recall=hit/total, dice=2*hit/(n+total), fp_ml=(n-hit)*GRID**3/1000)
    assert all(math.isclose(data[field], value, rel_tol=0, abs_tol=1e-12) for field, value in expected.items())


def audit():
    dev = read(COHORT)
    published = read(FRESH / 'design.json')
    new = read(CANDIDATE)
    old = read(PRIOR / 'candidate.private.json')
    splits = {key: patients(dev[key]) for key in ('train', 'dev', 'test', 'unlabelled')}
    splits.update(previous_candidates=patients(old), new_candidates=patients(new))
    assert all(a.isdisjoint(b) for a,b in combinations(splits.values(),2))
    assert len(new) == published['participants'] and published['candidate_sha'] == sha(CANDIDATE)
    seen = PREVIOUS() | patients(old)
    assert len(seen) == published['past_patient_count'] and seen.isdisjoint(patients(new))
    frozen = read(CALIBRATION)
    assert frozen['references'] == sha(MANIFEST) and frozen['plan'] == sha(PLAN)
    data = read(OUT / 'followup_aggregate.json')
    structure = read(STRUCTURED / 'aggregate.json')
    assert data['calibration'] == sha(CALIBRATION) and data['design'] == sha(PLAN)
    assert structure['design'] == sha(STRUCTURED_LOCK)
    expect_design_time = max(p.stat().st_mtime_ns for p in (CALIBRATION, STRUCTURED_LOCK, PLAN, CANDIDATE, MANIFEST))
    for role in ('dev', 'official', 'replication', 'fresh'):
        observed, refs = read_scores(role)
        assert data['public_cohorts'][role]['patients'] == len(refs) == len(observed)
        for code, row in observed.items():
            total = row['candidate']['targets_voxels']
            expected = ARMS if role == 'fresh' else ('BN','BP')
            assert set(row['results']) == {f'{m}_{seed}' for m in expected for seed in SEEDS}
            if role == 'fresh':
                pred = OUT / role / f'{code}.json'
                structured = STRUCTURED / role / f'{code}.json'
                assert pred.stat().st_mtime_ns >= expect_design_time
                assert structured.stat().st_mtime_ns >= expect_design_time
                shapes = read(structured)['results']
                assert set(shapes) == {f'{m}_{seed}' for m in ('BN','BP') for seed in SEEDS}
                for r in shapes.values():
                    checks(r['frozen'], total)
            for r in row['results'].values():
                checks(r['frozen'], total)
                for metrics in (*r['thresholds'].values(), *r.get('text', {}).values()):
                    checks(metrics,total)
    receipt = read(FRESH / 'receipt.json')
    matched = reference_rows('fresh')
    size = read(SIZE_ROOT / 'design.json')
    assert len(matched) == len(frozen['scans'])
    result = dict(fresh_candidates=len(new), screened_pool=published['eligible_metadata'],
                  excluded_prior=published['past_patient_count'], passed_reference_geometry=receipt['references'],
                  geometry_exclusions=receipt['excluded'], retained_after_anatomy=len(matched),
                  unlabelled_training_with_size=size['eligible_unlabelled'], constant_voxels=size['constant_training_voxels'],
                  cohort_design=sha(FRESH/'design.json'), frozen_primary=sha(FROZEN),
                  analysis=sha(Path(__file__)), followup=sha(OUT/'followup_aggregate.json'),
                  structured=sha(STRUCTURED/'aggregate.json'))
    write(OUT/'public_provenance.json', result)
    print('Independence, lock timing, and per-case arithmetic verified for all cohorts.')


if __name__ == '__main__':
    os.umask(0o077)
    audit()
