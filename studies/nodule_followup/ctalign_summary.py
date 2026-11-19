"""Verify matched controls and release only patient-aggregated follow-up statistics."""

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np

from ctalign_eval import CALIBRATION, FIELDS, FROZEN, GRID, OUT, PLAN, PRIOR, SEEDS, TEXT, reference_rows
from ctalign_fresh import MANIFEST
from frozen_priors import read, sha, write
from joint_metrics import compare, summary
from joint_nodule_gate import report_volume
from joint_nodule_run import COHORT


def averages(rows):
    assert rows
    return {key: float(np.mean([row[key] for row in rows])) for key in FIELDS}


def paired(treatment, baseline):
    common = sorted(treatment.keys() & baseline.keys())
    if not common:
        return None
    return {key: summary([treatment[code][key]-baseline[code][key] for code in common]) for key in FIELDS}


def patient_scores(answer, group, *, threshold=None, text=None):
    records = {}
    for case, row in answer.items():
        values = [row['results'][f'{group}_{s}'] for s in SEEDS]
        if threshold is not None:
            picked = [v['thresholds'][str(threshold)] for v in values]
        elif text is not None:
            picked = [v['text'].get(text) for v in values]
            if any(x is None for x in picked):
                continue
        else:
            picked = [v['frozen'] for v in values]
        records[case] = averages(picked)
    return records


def archived_check(role, scored):
    source = PRIOR / 'confirmed.json' if role == 'replication' else FROZEN.with_name('confirmed.json')
    confirmed = read(source)['patients']
    for group in ('BN', 'BP'):
        expected = confirmed[group]
        got = scored[group]
        assert expected.keys() == got.keys()
        for field in FIELDS:
            target = [expected[case][field] for case in sorted(got)]
            present = [got[case][field] for case in sorted(got)]
            np.testing.assert_allclose(present, target, rtol=0, atol=1e-12,
                                       err_msg=f'Arms did not reconstruct frozen scores for {role}/{group}/{field}')
    return True


def read_scores(role):
    refs = reference_rows(role)
    result = {}
    for row in refs:
        path = OUT / role / f"{row['code']}.json"
        record = read(path)
        assert record['proof']['source'] == row['sha'] and record['proof']['plan'] == sha(PLAN)
        if role == 'fresh':
            assert record['proof']['calibration'] == sha(CALIBRATION)
        result[row['code']] = record
    return result, refs


def subgroup_data(answer, scores):
    groups = defaultdict(list)
    for code, row in answer.items():
        groups['candidate_within_6mm' if row['candidate']['within_6mm'] else 'candidate_beyond_6mm'].append(code)
        groups['reference_at_most_0.1ml' if row['candidate']['targets_voxels']*1.5**3/1000 <= .1
               else 'reference_above_0.1ml'].append(code)
        groups[row['site']].append(code)
    data = {}
    for name, cases in groups.items():
        if len(cases) < 5:
            data[name] = {'patients': len(cases)}
        else:
            left = {c: scores['BP'][c] for c in cases}
            right = {c: scores['BN'][c] for c in cases}
            data[name] = dict(patients=len(cases), bn=averages(list(right.values())),
                              bp=averages(list(left.values())), difference=paired(left, right))
    return data


def collect(role, lock):
    answer, refs = read_scores(role)
    groups = sorted({name.rsplit('_', 1)[0] for row in answer.values() for name in row['results']})
    scored = {group: patient_scores(answer, group) for group in groups}
    verified = archived_check(role, scored) if role in ('official', 'replication') else None
    original = {name: averages(list(records.values())) for name, records in scored.items()}
    output = dict(patients=len(answer), archival_anchors_exact=verified, frozen=original,
                  curves={group: {str(t): averages(list(patient_scores(answer, group, threshold=t).values()))
                                  for t in GRID} for group in ('BN', 'BP')},
                  contrast_frozen_bp_bn=paired(scored['BP'], scored['BN']),
                  candidate=dict(within_6mm=sum(row['candidate']['within_6mm'] for row in answer.values()),
                                 mean_reference_coverage=float(np.mean([r['candidate']['target_covered'] for r in answer.values()]))),
                  subgroups=subgroup_data(answer, scored))
    if lock:
        matched = patient_scores(answer, 'BP', threshold=lock['bp_threshold'])
        delta = paired(matched, scored['BN'])
        output['recall_matching'] = dict(bp=averages(list(matched.values())), delta=delta,
                                         recall_gap_within_2pp=abs(delta['recall']['delta']) <= .02,
                                         interval_within_2pp=(delta['recall']['ci95'][0] >= -.02
                                                             and delta['recall']['ci95'][1] <= .02))
    if role != 'dev':
        output['text'] = {}
        for group in ('BN', 'BP'):
            outputs = {variant: patient_scores(answer, group, text=variant) for variant in TEXT}
            output['text'][group] = {}
            for control in TEXT[1:]:
                matched = outputs[control]
                if not matched:
                    continue
                matched_original = {c: outputs['original'][c] for c in matched}
                output['text'][group][control] = dict(patients=len(matched),
                    original=averages(list(matched_original.values())), control=averages(list(matched.values())),
                    original_minus_control=paired(matched_original, matched))
    if role == 'fresh':
        output['frozen_factorial'] = compare(scored)
        output['frozen_factorial']['transport_given_prototypes'] = paired(scored['BP'], scored['BR'])
        measured = {r['code'] for r in refs if report_volume(r['query']) is not None}
        output['size_holds'] = {}
        for arm in ('U', 'P'):
            method = scored['B'+arm]
            for mode in ('off', 'constant'):
                control = scored[f'{mode}_{arm}']
                key = f'B{arm}_text_minus_{mode}'
                output['size_holds'][key] = dict(overall=paired(method, control), measured_subset=paired(
                    {c: method[c] for c in measured}, {c: control[c] for c in measured}) if len(measured)>=5 else None)
        output['size_query_measured'] = len(measured)
        output['per_seed'] = {
            str(seed): {group: averages([r['results'][f'{group}_{seed}']['frozen'] for r in answer.values()])
                        for group in groups} for seed in SEEDS}
        # Prospective internal case selection for reviewing change and failure.
        delta = {c: scored['BP'][c]['dice']-scored['BN'][c]['dice'] for c in answer}
        extra = {c: scored['BP'][c]['fp_ml']-scored['BN'][c]['fp_ml'] for c in answer}
        ranking = dict(best_dice=sorted(delta, key=lambda c: (-delta[c], c))[:3],
                       worst_dice=sorted(delta, key=lambda c: (delta[c], c))[:3],
                       largest_reference_exterior_increase=sorted(extra, key=lambda c: (-extra[c], c))[:3],
                       no_overlap_both=sorted(c for c in delta if scored['BP'][c]['miss']==scored['BN'][c]['miss']==1)[:3])
        write(OUT / 'case_review.private.json', ranking)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--roles', nargs='+', choices=('dev', 'official', 'replication', 'fresh'), required=True)
    parser.add_argument('--destination', type=Path, required=True)
    args = parser.parse_args()
    lock = read(CALIBRATION) if CALIBRATION.is_file() else None
    if 'fresh' in args.roles:
        assert lock and lock['plan'] == sha(PLAN) and lock['references'] == sha(MANIFEST)
    result = dict(design=sha(PLAN), calibration=sha(CALIBRATION) if lock else None,
                  bp_recall_matched_threshold=lock['bp_threshold'] if lock else None,
                  public_cohorts={role: collect(role, lock) for role in args.roles})
    write(args.destination, result)
    for role, data in result['public_cohorts'].items():
        print(role, data['patients'], 'BP−BN Dice pp', 100*data['contrast_frozen_bp_bn']['dice']['delta'])


if __name__ == '__main__':
    main()
