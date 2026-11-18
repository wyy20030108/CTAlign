"""Recompute the release tables from locally held per-patient evidence on CPU."""

import argparse
import hashlib
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
METRICS = ('dice', 'recall', 'fp_ml', 'miss')
CONTRASTS = {
    'report_only_effect': ('BN', 'AN'),
    'transport_only_effect': ('AU', 'AN'),
    'add_transport_to_report': ('BU', 'BN'),
    'add_report_to_transport': ('BU', 'AU'),
    'joint_over_main_baseline': ('BU', 'AN'),
    'transport_report_interaction': ('BU', 'BN', 'AU', 'AN'),
    'unlabelled_report_over_supervised': ('BN', 'BS'),
    'joint_transport_over_supervised': ('BU', 'BS'),
    'prototype_increment': ('BP', 'BU'),
    'full_framework_without_report': ('AP', 'AN'),
    'full_framework_with_report': ('BP', 'BN'),
    'full_report_beyond_framework': ('BP', 'AP'),
    'joint_full_over_supervised': ('BP', 'BS'),
    'full_framework_report_interaction': ('BP', 'BN', 'AP', 'AN'),
    'transport_given_prototypes': ('BP', 'BR'),
}


def sha256(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def verify_sources():
    sources = json.loads((ROOT / 'results/source-provenance.json').read_text(encoding='utf-8'))
    actual = {path.relative_to(ROOT).as_posix() for path in (ROOT / 'experiments').glob('*.py')}
    if set(sources) != actual:
        raise ValueError('Source inventory does not match the repository')
    for name, source in sources.items():
        path = ROOT / name
        if sha256(path) != source['sha256']:
            raise ValueError(f'Source hash mismatch: {name}')
        compile(path.read_bytes(), str(path), 'exec')
    print(f'Checked hashes and Python syntax for {len(sources)} experiment files.')


def check_close(got, expected, label):
    try:
        np.testing.assert_allclose(got, expected, rtol=0, atol=1e-12)
    except AssertionError as e:
        raise ValueError(f'Result mismatch: {label}') from e


def verify_cohort(name, scores, expected):
    if set(scores) != set(expected['arm_means']):
        raise ValueError(f'Arm mismatch: {name}')
    patients = sorted(scores['AN'])
    size = len(patients)
    if size != expected['patients'] or any(set(arm) != set(patients) for arm in scores.values()):
        raise ValueError(f'Unmatched patient scores: {name}')
    indices = np.random.default_rng(20261119).integers(size, size=(20_000, size))
    values = {arm: {metric: np.array([rows[p][metric] for p in patients]) for metric in METRICS}
              for arm, rows in scores.items()}
    for arm, metrics in values.items():
        for metric, vector in metrics.items():
            check_close(vector.mean(), expected['arm_means'][arm][metric], f'{name}/{arm}/{metric}')
    for contrast, metrics in expected['paired_contrasts'].items():
        arms = CONTRASTS[contrast]
        signs = (1, -1, -1, 1) if len(arms) == 4 else (1, -1)
        for metric, stats in metrics.items():
            # Match the original Python scalar summation and its treatment of ties.
            delta = np.array([sum(sign * scores[arm][patient][metric]
                                  for arm, sign in zip(arms, signs, strict=True))
                              for patient in patients])
            bounds = np.quantile(delta[indices].mean(axis=1), (0.025, 0.975))
            check_close([delta.mean(), *bounds, (delta > 0).sum(), size],
                        [stats['delta'], *stats['ci95'], stats['improvement'], stats['patients']],
                        f'{name}/{contrast}/{metric}')
    main = expected['paired_contrasts']['full_framework_with_report']['dice']
    low, high = main['ci95']
    print(f'{name}: {size} patients, {len(scores)} arms, {len(expected["paired_contrasts"])} contrasts '
          f'verified. BP − BN: {100 * main["delta"]:+.2f} pp [{100 * low:+.2f}, {100 * high:+.2f}].')


def verify_archive(path, expected):
    if sha256(path) != expected['evidence_archive_sha256']:
        raise ValueError('The local evidence archive differs from the release SHA-256')
    with ZipFile(path) as archive:
        def read(name):
            return json.loads(archive.read(name))

        official = read('nodule_v2/external_matched/confirmed.json')['patients']
        official['BR'] = read('nodule_v2/external_proto_raw/confirmed.json')['patients']
        replication = read('nodule_replication/confirmed.json')['patients']
    verify_cohort('official_validation', official, expected['cohorts']['official_validation'])
    verify_cohort('replication', replication, expected['cohorts']['replication'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, help='Path to the original local evidence ZIP')
    args = parser.parse_args()
    verify_sources()
    if args.archive:
        release = json.loads((ROOT / 'results/nodule.json').read_text(encoding='utf-8'))
        verify_archive(args.archive, release)


if __name__ == '__main__':
    main()
