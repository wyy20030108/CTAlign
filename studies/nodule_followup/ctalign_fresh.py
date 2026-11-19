"""Reserve a disjoint same-source pulmonary nodule cohort before scoring."""

import argparse
from collections import Counter, defaultdict
import os
from pathlib import Path

from audit_pairs import identity
from diag_localize import seal
from frozen_priors import read, sha
import joint_replication_cohort as replicate
from replay_controls import read_reports
from swap_data import side
from thesis_cohort import choose_stratified, site


ROOT = Path('phase10_thesis')
DEST = ROOT / 'fresh'
STORAGE = Path('/tmp/ctalign_thesis_fresh')
CANDIDATE = DEST / 'candidate.private.json'
MANIFEST = DEST / 'replication_manifest.private.json'
PRIOR = Path('phase9_joint/nodule_replication')
FROZEN = Path('phase9_joint/nodule_v2/external_matched/external_lock.json')
PREVIOUS = replicate.previous_sources


def screen(target):
    DEST.mkdir(parents=True, exist_ok=True)
    STORAGE.mkdir(parents=True, exist_ok=True)
    seen = PREVIOUS()
    for name in ('candidate.private.json', 'replication_manifest.private.json'):
        seen.update(tuple(row['patient'].split(':')) for row in read(PRIOR / name))
    pending = []
    for row in read(replicate.REX)['train']:
        name, patient, _ = identity(row['name'])
        if patient in seen:
            continue
        sites = Counter(site({'query': text}) for text in row['findings'].values() if replicate.focal(text))
        for channel, text in row['findings'].items():
            if (replicate.focal(text) and row['entity_counts'][channel] == 1
                    and row['pixels'][channel] > 0 and sites[site({'query': text})] == 1):
                pending.append((name, patient, channel, text, row['pixels'][channel]))
    reports = read_reports({row[0] for row in pending})
    matched = defaultdict(list)
    for name, patient, channel, finding, pixels in pending:
        report = reports[name]['Findings_EN']
        result = replicate.query(finding, report)
        if not result:
            continue
        exact, context = result
        assert exact in context and context in report
        matched[patient].append(dict(name=name, patient=':'.join(patient), channel=int(channel),
                                     native_count=pixels, side=side(exact), query=exact, context=context,
                                     published=finding, site=site({'query': exact}), code=name+'-'+channel))
    pool = [min(rows, key=lambda row: row['code']) for rows in matched.values()]
    chosen = choose_stratified(pool, min(len(pool), target), 'ctalign-thesis-heldout-20261120')
    assert len(chosen) >= 40 and len({r['patient'] for r in chosen}) == len(chosen)
    assert not ({tuple(r['patient'].split(':')) for r in chosen} & seen)
    for number, row in enumerate(chosen, 1):
        row['code'] = f'Z{number:03d}'
    seal(CANDIDATE, chosen)
    seal(DEST / 'design.json', dict(participants=len(chosen), eligible_metadata=len(pool),
                                   past_patient_count=len(seen), candidate_sha=sha(CANDIDATE),
                                   frozen_model_and_thresholds=sha(FROZEN), program=sha(Path(__file__)),
                                   analysis='Fresh same-source holdout. Candidate set sealed before scoring. '
                                            'Published single-instance reference matched to the same examination report.'))
    print('Fresh candidates', len(chosen), 'eligible metadata', len(pool), 'prior patients excluded', len(seen), flush=True)


def main():
    os.umask(0o077)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('screen', 'fetch', 'seal'))
    parser.add_argument('--target', type=int, default=72)
    parser.add_argument('--shard', type=int, default=0)
    parser.add_argument('--shards', type=int, default=8)
    args = parser.parse_args()
    replicate.DEST, replicate.STORAGE = DEST, STORAGE
    replicate.CANDIDATE, replicate.LOCK = CANDIDATE, MANIFEST
    if args.stage == 'screen':
        screen(args.target)
        return
    assert sha(CANDIDATE) == read(DEST / 'design.json')['candidate_sha']
    if args.stage == 'fetch':
        replicate.fetch(args.shard, args.shards)
    else:
        replicate.seal_references()


if __name__ == '__main__':
    main()
