"""Compare a complete structured-report route to the original query route."""

import argparse
import os
from pathlib import Path

from ctalign_eval import CALIBRATION, FIELDS, FRESH, FROZEN, LOBE, PLAN, ROOT
from ctalign_eval import base, gate, load_models, lobes, reference_rows, score_field
from ctalign_summary import averages, paired, patient_scores, read_scores
from diag_localize import seal
from frozen_priors import read, sha, write
from thesis_cohort import site


OUT = ROOT / 'structured'
LOCK = OUT / 'design.json'
ARMS = ('BN', 'BP')


def template(region):
    name = 'left lingula' if region == 'left_lingular' else LOBE[region]
    query = f'Pulmonary nodule in the {name}.'
    assert site({'query': query}) == region
    return query


def design():
    return dict(program=sha(Path(__file__)), main_protocol=sha(PLAN), model=sha(FROZEN),
                templates={key: template(key) for key in sorted(LOBE)}, arms=list(ARMS),
                question='Use only location extracted from the report to build the control query. '
                         'Apply that query to both spatial routing and the network. Keep the frozen '
                         'detector, weights, thresholds and reference definitions. The 22 and 99 groups '
                         'are retrospective. Fresh results assess this specific inference variant.')


def score(role, shard, shards):
    assert read(LOCK) == design()
    if role == 'fresh':
        frozen = read(CALIBRATION)
        assert frozen['plan'] == sha(PLAN)
        base.BOXES = FRESH / 'detections'
        gate.LOCATIONS = FRESH / 'shape'
        lobes.CACHE = FRESH / 'lobes'
    where = OUT / role
    where.mkdir(parents=True, exist_ok=True)
    rows = reference_rows(role)
    if role == 'fresh':
        assert frozen['scans'] == {r['code']: r['sha'] for r in rows}
    models = load_models('dev')
    data = lobes.LobeStore(limit=1)
    for row in rows[shard::shards]:
        code = row['code']
        path = where / f'{code}.json'
        proof = dict(design=sha(LOCK), source=row['sha'])
        if path.exists():
            assert read(path)['proof'] == proof
            continue
        case = data.get(row, True)
        query = template(row['site'])
        result = {}
        for arm, seed, model, cap, threshold in models:
            volume, _ = base.volume(model, case, query)
            result[f'{arm}_{seed}'] = dict(frozen=score_field(volume, case.truth, len(case.positive),
                                                           cap, (threshold,))[str(threshold)])
        write(path, dict(proof=proof, results=result))
        print(role, code, 'structured', flush=True)


def aggregate(role):
    original, rows = read_scores(role)
    changed = {}
    for row in rows:
        code = row['code']
        path = OUT / role / f'{code}.json'
        r = read(path)
        assert r['proof'] == dict(design=sha(LOCK), source=row['sha'])
        changed[code] = r
    structured = {arm: patient_scores(changed, arm) for arm in ARMS}
    output = dict(patients=len(rows), full_route={})
    for arm in ARMS:
        full = patient_scores(original, arm)
        fixed_template = patient_scores(original, arm, text='template')
        output['full_route'][arm] = dict(structured=averages(list(structured[arm].values())),
            original=averages(list(full.values())), structured_minus_original=paired(structured[arm], full),
            structured_minus_fixed_spatial=paired(structured[arm], fixed_template))
    output['structured_bp_minus_bn'] = paired(structured['BP'], structured['BN'])
    return output


if __name__ == '__main__':
    os.umask(0o077)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('register', 'score', 'release'))
    parser.add_argument('--cohort', choices=('official', 'replication', 'fresh'), default='fresh')
    parser.add_argument('--shard', type=int, default=0)
    parser.add_argument('--shards', type=int, default=1)
    parser.add_argument('--roles', nargs='+', choices=('official', 'replication', 'fresh'))
    args = parser.parse_args()
    if args.stage == 'register':
        OUT.mkdir(parents=True, exist_ok=True)
        seal(LOCK, design())
    elif args.stage == 'score':
        score(args.cohort, args.shard, args.shards)
    else:
        assert read(LOCK) == design()
        report = {role: aggregate(role) for role in (args.roles or ['fresh'])}
        write(OUT / 'aggregate.json', dict(design=sha(LOCK), cohorts=report))
        for role, value in report.items():
            print(role, value['patients'], 'structured BP−BN pp', 100*value['structured_bp_minus_bn']['dice']['delta'])
