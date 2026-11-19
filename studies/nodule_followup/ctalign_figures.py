"""Create print-ready figures using only released patient-aggregate statistics."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


COHORT_NAMES = {'official': 'Official validation', 'replication': 'Previous replication',
                'fresh': 'New same-source holdout'}
COMPARISONS = (
    ('BU_text_minus_off', 'UOT: size estimate vs. off'),
    ('BU_text_minus_constant', 'UOT: size estimate vs. constant'),
    ('BP_text_minus_off', 'Full: size estimate vs. off'),
    ('BP_text_minus_constant', 'Full: size estimate vs. constant'),
)
COLORS = {'BN': '#276082', 'BP': '#9E421C'}


def operating_curves(cohorts, threshold, directory):
    fig, axes = plt.subplots(1, 3, figsize=(11.8, 3.65), sharey=True, layout='constrained')
    for ax, (role, name) in zip(axes, COHORT_NAMES.items(), strict=True):
        data = cohorts[role]
        for group in ('BN', 'BP'):
            points = [value for _, value in sorted(data['curves'][group].items(), key=lambda item: float(item[0]))]
            ax.plot([100*p['recall'] for p in points], [p['fp_ml'] for p in points],
                    color=COLORS[group], marker='.', linewidth=1.5, markersize=4, label=group)
        mark = {'BN': data['frozen']['BN'], 'BP': data['recall_matching']['bp']}
        for group, p in mark.items():
            ax.scatter(100*p['recall'], p['fp_ml'], color=COLORS[group], edgecolors='white', zorder=5, s=59)
        ax.set_title(f"{name}\n$n$ = {data['patients']}", fontsize=10)
        ax.set_xlabel('Mean voxel recall (%)')
        ax.grid(True, alpha=.22)
    axes[0].set_ylabel('Mean volume outside reference (mL)')
    axes[0].legend(frameon=False, loc='upper left')
    fig.suptitle(f'Frozen models across thresholds. Emphasized points use BN original and BP {threshold:.2f}.', fontsize=11)
    fig.savefig(directory / 'ctalign_recall_volume.png', dpi=300, facecolor='white')
    plt.close(fig)


def contrast_plot(ax, labels, effects, title):
    for y, (label, stats) in enumerate(zip(labels, effects, strict=True)):
        est = stats['delta']*100
        low, high = [100*x for x in stats['ci95']]
        color = '#9E421C' if low > 0 or high < 0 else '#4E687A'
        ax.errorbar(est, y, xerr=[[est-low], [high-est]], color=color, fmt='o', capsize=3, markersize=4)
    ax.axvline(0, color='#6E7780', linestyle='--', linewidth=.9)
    ax.set_yticks(range(len(labels)), labels)
    ax.invert_yaxis()
    ax.set_title(title, fontsize=11)
    ax.set_xlabel('Paired Dice difference (percentage points)')
    ax.grid(axis='x', alpha=.22)


def ablation_effects(cohorts, directory):
    fig, axes = plt.subplots(2, 1, figsize=(8.6, 7.5), layout='constrained', gridspec_kw={'height_ratios': [3, 2]})
    labels = []
    effects = []
    for role, name in COHORT_NAMES.items():
        for condition, label in (('template', 'template'), ('same_site_swap', 'same-lobe swap')):
            report = cohorts[role]['text']['BP'][condition]
            labels.append(f"{name} · {label} (n={report['patients']})")
            effects.append(report['original_minus_control']['dice'])
    contrast_plot(axes[0], labels, effects, 'Original report vs. control with identical spatial inputs')
    fresh = cohorts['fresh']
    labels = [label for _, label in COMPARISONS]
    effects = [fresh['size_holds'][key]['overall']['dice'] for key, _ in COMPARISONS]
    contrast_plot(axes[1], labels, effects, f"Training-side report size prior on the new holdout (n={fresh['patients']})")
    fig.savefig(directory / 'ctalign_text_size.png', dpi=300, facecolor='white')
    plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('aggregate', type=Path)
    parser.add_argument('output_directory', type=Path)
    args = parser.parse_args()
    data = json.loads(args.aggregate.read_text(encoding='utf-8'))
    args.output_directory.mkdir(parents=True, exist_ok=True)
    operating_curves(data['public_cohorts'], data['bp_recall_matched_threshold'], args.output_directory)
    ablation_effects(data['public_cohorts'], args.output_directory)
