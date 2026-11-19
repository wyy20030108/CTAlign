"""Matched re-fits isolating the training-side size prior inside local Adaptive UOT."""

import argparse
import os
from pathlib import Path
from statistics import median

from diag_localize import seal
from frozen_priors import read, sha
import joint_external_refine as base
import joint_factorial as experiment
import joint_nodule_gate as gate
import joint_nodule_growth_refine as growth
from joint_nodule_run import COHORT
from joint_nodule_size import SIZE_CALIBRATION, get_calibrator
from thesis_ssl import SEEDS


ROOT = Path('phase10_thesis/size')
MODES = ('off', 'constant')
ARMS = ('U', 'P')


def settings():
    cohort = read(COHORT)
    visible = set(cohort['labels']['20'])
    training = [row for row in cohort['train'] if row['code'] in visible]
    assert len(training) == 7
    calibrated = get_calibrator(COHORT)
    constant = float(median(row['coords'] for row in training))
    regions = sorted(base.ROI.glob('B_*.json'))
    assert len(regions) == len(cohort['unlabelled']) == 36
    expected = dict(program=sha(Path(__file__)), model=sha(Path(base.__file__)),
                    training=sha(Path(experiment.__file__)), teacher=sha(Path(growth.__file__)),
                    size_calibration=sha(SIZE_CALIBRATION), cohort=sha(COHORT),
                    original_protocol=sha(base.ROOT / 'protocol.json'),
                    matched_pseudo_regions={path.name: sha(path) for path in regions},
                    constant_training_voxels=constant, seeds=list(SEEDS), arms=list(ARMS),
                    eligible_unlabelled=sum(calibrated(row['query']) is not None for row in cohort['unlabelled']))
    return expected, constant, calibrated


def run(stage, mode, seed, arm):
    expected, constant, calibrated = settings()
    experiment.MANIFEST, experiment.PREVIOUS, experiment.ROI = COHORT, base.WARM, base.ROI
    if stage == 'lock':
        ROOT.mkdir(parents=True, exist_ok=True)
        seal(ROOT / 'design.json', expected)
        for variant in MODES:
            experiment.OUT = ROOT / variant
            experiment.lock('agreement', 'agreement', 'conditioned')
        print('Size controls locked; eligible unlabelled:', expected['eligible_unlabelled'], flush=True)
        return
    assert read(ROOT / 'design.json') == expected
    experiment.OUT = ROOT / mode
    base.install()
    growth.ROI = base.ROI
    if mode == 'off':
        gate.report_volume = lambda _: None
    else:
        # Retain the same textual availability trigger. This control tests individual size values.
        gate.report_volume = lambda query: constant if calibrated(query) is not None else None
    experiment.teacher_target = growth.target
    experiment.fit('B', arm, seed)


if __name__ == '__main__':
    os.umask(0o077)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('lock', 'fit'))
    parser.add_argument('--mode', choices=MODES, default='off')
    parser.add_argument('--seed', type=int, choices=SEEDS, default=SEEDS[0])
    parser.add_argument('--arm', choices=ARMS, default='U')
    args = parser.parse_args()
    run(args.stage, args.mode, args.seed, args.arm)
