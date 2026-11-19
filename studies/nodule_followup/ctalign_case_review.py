"""Render internal case panels from released references and freshly replayed predictions."""

import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from ctalign_eval import CALIBRATION, FRESH, OUT, PLAN, base, gate, load_models, lobes, reference_rows
from ctalign_summary import read_scores
from frozen_priors import read, sha
from thesis_component_fast import segment
from thesis_model import score


LABELS = {'best_dice': 'Largest Dice increase', 'worst_dice': 'Largest Dice decrease',
          'largest_reference_exterior_increase': 'Largest exterior-volume increase',
          'no_overlap_both': 'Zero overlap in both groups'}


def selected_cases():
    ranked = read(OUT / 'case_review.private.json')
    chosen = {}
    for group, candidates in ranked.items():
        unused = [code for code in candidates if code not in chosen.values()]
        if unused:
            chosen[group] = unused[0]
    return chosen


def render():
    base.BOXES = FRESH / 'detections'
    gate.LOCATIONS = FRESH / 'shape'
    lobes.CACHE = FRESH / 'lobes'
    archive, _ = read_scores('fresh')
    refs = {r['code']: r for r in reference_rows('fresh')}
    lock = read(CALIBRATION)
    assert lock['plan'] == sha(PLAN) and lock['scans'] == {c: row['sha'] for c, row in refs.items()}
    models = [m for m in load_models('dev') if m[1] == 891]
    selected = selected_cases()
    store = lobes.LobeStore(limit=1)
    fig, axes = plt.subplots(len(selected), 3, figsize=(9, len(selected)*3.05),
                             squeeze=False, layout='constrained')
    for row, (kind, code) in zip(axes, selected.items(), strict=True):
        card = refs[code]
        case = store.get(card, True)
        predictions = {}
        values = {}
        for arm, seed, net, cap, threshold in models:
            field, _ = base.volume(net, case, card['query'])
            predictions[arm] = segment(field, threshold, cap)
            values[arm] = score(predictions[arm], case.truth)
            expected = archive[code]['results'][f'{arm}_{seed}']['frozen']
            assert all(values[arm][m] == value for m, value in expected.items())
        target = case.truth
        if kind == 'largest_reference_exterior_increase':
            added = predictions['BP'] & ~target & ~predictions['BN']
            selection = added if added.any() else target
        else:
            selection = target
        z = int(np.argmax(selection.sum(axis=(0, 1))))
        x, y = np.argwhere(selection[:, :, z]).mean(axis=0).round().astype(int)
        roi = (slice(max(0, x-26), x+27), slice(max(0, y-26), y+27), z)
        for ax, group in zip(row, ('reference', 'BN', 'BP'), strict=True):
            ax.imshow(case.ct[roi].T, cmap='gray', vmin=-1000, vmax=200, origin='lower')
            truth_slice = target[roi].T
            if truth_slice.any():
                ax.contour(truth_slice, levels=[.5], colors=['#43D9EF'], linewidths=1.2, linestyles='--')
            if group == 'reference':
                text = 'Reference' if truth_slice.any() else 'Reference on another slice'
                ax.set_title(f'{LABELS[kind]}\n{text}', fontsize=9)
            else:
                bet = predictions[group][roi].T
                if bet.any():
                    ax.contour(bet, levels=[.5], colors=['#FFB154'], linewidths=1.2)
                report = values[group]
                ax.set_title(f"{group} · Dice {100*report['dice']:.1f}%\nExterior {report['fp_ml']:.3f} mL", fontsize=9)
            ax.axis('off')
    fig.suptitle('Internal review · seed 891 · cyan: published reference · orange: prediction', fontsize=10)
    fig.savefig(OUT / 'ctalign_case_review.private.png', dpi=250)
    plt.close(fig)


if __name__ == '__main__':
    os.umask(0o077)
    render()
