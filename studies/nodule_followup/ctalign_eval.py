"""Frozen inference, fixed-spatial text controls, and development-only threshold matching."""

import argparse
from hashlib import sha256
import os
from pathlib import Path

import numpy as np
import torch

from ctalign_fresh import CANDIDATE, DEST as FRESH, FROZEN, MANIFEST, ROOT, STORAGE
from ctalign_size import MODES, ROOT as SIZE_ROOT
from diag_localize import seal
from finding_train import configure, tokens
from frozen_priors import read, sha, write
import joint_external_nodule as detector
import joint_external_refine as base
from joint_metrics import ARM
import joint_nodule_gate as gate
from joint_nodule_run import COHORT
import joint_replication_eval as replicate
from swap_data import norm
from thesis_component_fast import segment
import thesis_lobe_model as lobes
import thesis_lobes as prepare_lobes
import thesis_model as image
from thesis_ssl import SEEDS


OUT = ROOT / 'evaluation'
PLAN = OUT / 'design.json'
CALIBRATION = OUT / 'development_lock.json'
BASE_PROTOTYPE = replicate.PROTO_AUDIT / 'evaluation_lock.json'
PRIOR = Path('phase9_joint/nodule_replication')
GRID = (.10, .15, .20, .25, .30, .35, .40, .45, .50, .55, .60, .65, .70, .75, .80, .85, .90, .95)
TEXT = ('original', 'template', 'same_site_swap', 'empty')
FIELDS = ('dice', 'recall', 'fp_ml', 'miss')
LOBE = {'left_upper': 'left upper lobe', 'left_lingular': 'lingula of the left upper lobe',
        'left_lower': 'left lower lobe', 'right_upper': 'right upper lobe',
        'right_middle': 'right middle lobe', 'right_lower': 'right lower lobe'}


def design():
    return dict(source=sha(Path(__file__)), primary_model=sha(FROZEN),
                prototype_model=sha(BASE_PROTOTYPE), size_design=sha(SIZE_ROOT / 'design.json'),
                cohort=sha(COHORT), old_replication=sha(PRIOR / 'model_lock.json'),
                fresh_design=sha(FRESH / 'design.json'), seed=list(SEEDS), threshold_grid=list(GRID),
                size_modes=list(MODES), text_conditions=list(TEXT),
                thresholds='BN uses its original operating point; choose one BP threshold closest in mean '
                           'recall on the development set only, breaking ties by reduced reference-exterior '
                           'volume, then by the higher threshold. Report the attained held-out recall gap. '
                           'Compare fixed points with a 2 percentage point absolute recall-gap tolerance.',
                text='Freeze the original query for all spatial features, windows and domain. Replace '
                     'only the network query. Compare the original report against a same-lobe short '
                     'template, lexically distinct same-lobe patient text, and empty tokens.',
                reference_groups='Official 22 and prior same-source 99 are retrospective. Fresh reserved '
                                 'cases provide same-source confirmation. Never pool the three groups.')


def cache_fresh(shard, shards):
    detector.OUT = FRESH / 'detections'
    detector.OUT.mkdir(parents=True, exist_ok=True)
    gate.LOCATIONS = FRESH / 'shape'
    prepare_lobes.CACHE = FRESH / 'lobes'
    replicate.DEST, replicate.STORAGE, replicate.CANDIDATE = FRESH, STORAGE, CANDIDATE
    replicate.PREP = FRESH / 'cache_receipts'
    alive = replicate.alive
    replicate.alive = lambda pid, _: alive(pid, 'ctalign_fresh.py')
    replicate.anatomical_cache(shard, shards)


def reference_rows(role):
    if role == 'fresh':
        manifest = read(MANIFEST)
        assert sha(MANIFEST) == read(FRESH / 'receipt.json')['manifest']
        good = []
        for row in manifest:
            code = row['code']
            receipt = read(FRESH / 'cache_receipts' / f'{code}.json')
            assert receipt['scan'] == row['sha']
            if receipt['status'] == 'ready':
                assert receipt['design'] == sha(FRESH / 'design.json')
                assert receipt['detector'] == sha(FRESH / 'detections' / f'{code}.json')
                good.append(row)
            else:
                assert receipt['status'] == 'anatomy_preparation'
        assert good
        return good
    if role == 'replication':
        locked = read(PRIOR / 'model_lock.json')['scans']
        rows = [r for r in read(PRIOR / 'replication_manifest.private.json') if r['code'] in locked]
        assert {r['code']: r['sha'] for r in rows} == locked
        return rows
    return read(COHORT)['test' if role == 'official' else 'dev']


def load_models(role):
    principal = read(FROZEN)
    auxiliary = read(BASE_PROTOTYPE)
    which = ('BN', 'BP') if role == 'dev' else ((*ARM, 'BR') if role == 'fresh' else ('BN', 'BP'))
    models = []
    for arm in which:
        for seed in SEEDS:
            if arm == 'BR':
                item = auxiliary['weights'][str(seed)]
                weight = replicate.PROTO / 'B' / str(seed) / 'R' / 'model.pt'
                cap, threshold, digest = item['cap'], auxiliary['threshold'], item['model']
            else:
                item = next(v for v in principal['weights'][arm] if v['seed'] == seed)
                weight = base.ROOT / arm[0] / str(seed) / arm[1] / 'model.pt'
                cap, threshold, digest = item['cap_ml'], principal['thresholds'][arm]['threshold'], item['weight']
            models.append((arm, seed, weight, digest, cap, threshold))
    if role == 'fresh':
        locked = read(CALIBRATION)['size']
        for group, evidence in locked.items():
            mode, arm = group.split('_')
            for seed in SEEDS:
                value = evidence['seeds'][str(seed)]
                path = SIZE_ROOT / mode / 'B' / str(seed) / arm / 'model.pt'
                assert sha(path.with_name('development.json')) == value['receipt']
                models.append((group, seed, path, value['model'], value['cap'], evidence['threshold']))
    result = []
    for arm, seed, weight, digest, cap, threshold in models:
        assert sha(weight) == digest
        configure(seed)
        net = image.FindingNet().to(image.DEVICE).eval()
        net.load_state_dict(torch.load(weight, weights_only=True, map_location=image.DEVICE))
        result.append((arm, seed, net, cap, threshold))
    return result


def variants(row, bank):
    original = row['query']
    location = LOBE[row['site']]
    pool = [r for r in bank if r['site'] == row['site'] and r['patient'] != row['patient']
            and norm(r['query']) != norm(original) and tokens(r['query']) != tokens(original)]
    donor = min(pool, key=lambda r: sha256(f"{row['code']}:{r['code']}".encode()).digest()) if pool else None
    result = {'original': original, 'template': f'Pulmonary nodule in the {location}.', 'empty': ''}
    if donor is not None:
        result['same_site_swap'] = donor['query']
    return result


class TextOnly:
    def __init__(self, net, query):
        self.net, self.query = net, query

    def __call__(self, features, _):
        return self.net(features, [self.query])


def footprint(field):
    support = field > -39.99
    extents = [np.flatnonzero(support.any(axis=tuple(j for j in range(3) if j != axis))) for axis in range(3)]
    if not all(len(axis) for axis in extents):
        return None
    return tuple(slice(int(axis[0]), int(axis[-1])+1) for axis in extents)


def metrics(prediction, truth, total):
    n = int(prediction.sum())
    intersection = int(np.count_nonzero(prediction & truth))
    return dict(dice=2*intersection/(total+n), recall=intersection/total,
                fp_ml=(n-intersection)*image.GRID**3/1000, miss=intersection == 0,
                pred=n, intersection=intersection)


def score_field(field, truth, total, cap, thresholds):
    region = footprint(field)
    if region is None:
        return {str(t): dict(dice=0., recall=0., fp_ml=0., miss=True, pred=0, intersection=0)
                for t in thresholds}
    local, reference = np.ascontiguousarray(field[region]), truth[region]
    return {str(t): metrics(segment(local, t, cap), reference, total) for t in thresholds}


def localize(case, query):
    points = np.asarray([point for point, _ in base.locations(case, query)])
    target = case.positive
    dist = np.linalg.norm(target[:, None] - points[None, :], axis=-1)
    visible = (dist.min(axis=1) <= 10) & case.domain(query)[tuple(target.T)]
    return dict(minimum_mm=float(dist.min()*image.GRID), within_6mm=bool(dist.min()*image.GRID <= 6),
                target_covered=float(visible.mean()), targets_voxels=len(target))


def score(role, shard, shards, count):
    assert read(PLAN) == design()
    if role == 'fresh':
        locked = read(CALIBRATION)
        assert locked['plan'] == sha(PLAN) and locked['references'] == sha(MANIFEST)
        base.BOXES = FRESH / 'detections'
        gate.LOCATIONS = FRESH / 'shape'
        lobes.CACHE = FRESH / 'lobes'
    root = OUT / role
    root.mkdir(parents=True, exist_ok=True)
    rows = reference_rows(role)
    if role == 'fresh':
        assert {r['code']: r['sha'] for r in rows} == locked['scans']
    dev = read(COHORT)
    bank = dev['train'] + dev['dev']
    models = load_models(role)
    case_store = lobes.LobeStore(limit=1)
    for row in rows[shard::shards][:count]:
        code = row['code']
        path = root / f'{code}.json'
        proof = dict(source=row['sha'], plan=sha(PLAN),
                     calibration=sha(CALIBRATION) if role == 'fresh' else None)
        if path.exists():
            assert read(path)['proof'] == proof
            continue
        case = case_store.get(row, True)
        query = row['query']
        text = variants(row, bank) if role != 'dev' else {}
        score_rows = {}
        for group, seed, net, cap, threshold in models:
            name = f'{group}_{seed}'
            side_query = '' if group.startswith('A') else query
            volume, _ = base.volume(net, case, side_query)
            total = len(case.positive)
            grid = GRID if group in ('BN', 'BP') else ()
            sweep = score_field(volume, case.truth, total, cap, sorted(set((*grid, threshold))))
            record = dict(frozen=sweep[str(threshold)], thresholds=sweep if grid else {})
            del volume
            if text and group in ('BN', 'BP'):
                control = {'original': record['frozen']}
                for variant, replaced in text.items():
                    if variant == 'original':
                        continue
                    changed, _ = base.volume(TextOnly(net, replaced), case, query)
                    control[variant] = score_field(changed, case.truth, total, cap, (threshold,))[str(threshold)]
                    del changed
                record['text'] = control
            score_rows[name] = record
        write(path, dict(proof=proof, results=score_rows, candidate=localize(case, query), site=row['site']))
        print(role, code, len(score_rows), flush=True)


def lock_development():
    assert read(PLAN) == design()
    cases = reference_rows('fresh')
    case_lock = {r['code']: r['sha'] for r in cases}
    dev_refs = reference_rows('dev')
    files = [OUT / 'dev' / (r['code'] + '.json') for r in dev_refs]
    curves = [read(p)['results'] for p in files]
    end = [r[f'BN_{seed}']['frozen']['recall'] for r in curves for seed in SEEDS]
    target = float(np.mean(end))
    candidates = {}
    for t in GRID:
        scores = [r[f'BP_{seed}']['thresholds'][str(t)] for r in curves for seed in SEEDS]
        candidates[t] = dict(recall=float(np.mean([s['recall'] for s in scores])),
                             fp_ml=float(np.mean([s['fp_ml'] for s in scores])))
    point = min(GRID, key=lambda t: (abs(candidates[t]['recall']-target), candidates[t]['fp_ml'], -t))
    size = {}
    for mode in MODES:
        for arm in ('U', 'P'):
            base_root = SIZE_ROOT / mode
            items = {}
            for seed in SEEDS:
                path = base_root / 'B' / str(seed) / arm / 'development.json'
                r = read(path)
                assert r['protocol'] == sha(base_root / 'protocol.json')
                assert r['model'] == sha(path.with_name('model.pt'))
                items[str(seed)] = dict(model=r['model'], cap=r['cap_ml'], receipt=sha(path))
            # Preserve the exact frozen original threshold, avoiding a threshold change as a size effect.
            t = read(FROZEN)['thresholds']['B'+arm]['threshold']
            size[f'{mode}_{arm}'] = dict(threshold=t, seeds=items)
    seal(CALIBRATION, dict(plan=sha(PLAN), references=sha(MANIFEST), scans=case_lock,
                           dev_receipts={p.stem: sha(p) for p in files}, bn_target_recall=target,
                           bp_threshold=point, bp_candidates={str(k): v for k, v in candidates.items()}, size=size))
    print('Locked development BP threshold', point, 'BN target recall', target, 'fresh patients', len(cases), flush=True)


def main():
    os.umask(0o077)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('register', 'cache', 'score', 'lock'))
    parser.add_argument('--cohort', choices=('dev', 'official', 'replication', 'fresh'), default='dev')
    parser.add_argument('--shard', type=int, default=0)
    parser.add_argument('--shards', type=int, default=1)
    parser.add_argument('--limit', type=int)
    args = parser.parse_args()
    assert 0 <= args.shard < args.shards
    if args.stage == 'register':
        OUT.mkdir(parents=True, exist_ok=True)
        seal(PLAN, design())
    elif args.stage == 'cache':
        assert read(PLAN) == design()
        cache_fresh(args.shard, args.shards)
    elif args.stage == 'score':
        score(args.cohort, args.shard, args.shards, args.limit)
    else:
        lock_development()


if __name__ == '__main__':
    main()
