"""Develop candidate pseudo-labels on previously seen development patients only."""
from argparse import ArgumentParser
from pathlib import Path
import os

import numpy as np
from scipy import ndimage as ndi
import torch

import thesis_model as image
from finding_train import configure
from frozen_priors import read, sha, write
from thesis_cohort import MANIFEST
from thesis_component_fast import segment
from thesis_lobe_model import CODES, LobeStore
from thesis_ssl import OUT as PREVIOUS, SEEDS


ROOT = Path("phase9_joint")
ROI = ROOT / "candidate_rois"
AUDIT = ROOT / "candidate_audit"
CHOICES = ("single", "mean", "agreement", "local")


def warm_models(kind):
    result = []
    for seed in SEEDS:
        source = PREVIOUS / "20" / str(seed) / "initial" / (kind + ".pt")
        assert sha(source) == read(source.with_suffix(".json"))["sha"]
        configure(seed)
        net = image.FindingNet().to(image.DEVICE).eval().requires_grad_(False)
        net.load_state_dict(torch.load(source, map_location=image.DEVICE, weights_only=True))
        result.append(net)
    return result


def candidate_fields(models, case, row, text):
    data = np.stack([image.volume_logits(net, case, text)[0] for net in models])
    mean = np.mean(data, axis=0)
    agreement = mean - .65 * np.std(data, axis=0)
    lobe = CODES.get(row["site"]) if text else None
    local = agreement.copy()
    if lobe is not None:
        nearby = case.allowed.get(lobe)
        if nearby is None:
            nearby = ndi.binary_dilation(case.lobes == lobe, iterations=4)
        local[nearby] += .60
    # High local contrast suppresses broad tissue without closing the same-side search space.
    local += .45 * np.clip(case.detail.astype(np.float32) / 300.0, 0, 1)
    return [data[0], mean, agreement, local]


def evidence_region(field, domain, count):
    active = domain & np.isfinite(field)
    available = np.flatnonzero(active)
    scores = field.ravel()[available]
    assert len(scores) >= 24
    strongest = available[np.argmax(scores)]
    center = np.asarray(np.unravel_index(strongest, field.shape))
    begin = np.clip(center - image.SIDE // 2, 0, np.array(field.shape) - image.SIDE)
    region = image.slices(begin)
    eligible = np.flatnonzero(active[region])
    values = field[region].ravel()[eligible]
    maximum = min(count, len(eligible))
    indices = eligible[np.argpartition(values, -maximum)[-maximum:]]
    selected = np.zeros((image.SIDE,) * 3, bool)
    selected.ravel()[indices] = True
    return list(map(int, begin)), selected


def run(role, kind, shard, shards):
    out = AUDIT if role == "dev" else ROI
    out.mkdir(exist_ok=True, parents=True)
    candidates = read(MANIFEST)[role][shard::shards]
    models = warm_models(kind)
    store = LobeStore(limit=2)
    for row in candidates:
        code = row["code"]
        path = out / (kind + "_" + code + ".json")
        if path.exists():
            assert read(path)["source"] == row["sha"]
            continue
        case = store.get(row, role == "dev")
        text = row["query"] if kind == "B" else ""
        domain = case.domain(text)
        fields = candidate_fields(models, case, row, text)
        results = {}
        for label, field in zip(CHOICES, fields, strict=True):
            origin, foreground = evidence_region(field, domain, 1024)
            one = dict(origin=origin, proposal_foreground=int(foreground.sum()))
            if role == "dev":
                region = image.slices(origin)
                reference = case.truth[region]
                one.update(window_hits=int(reference.sum()), target=int(case.truth.sum()),
                           foreground_hits=int((reference & foreground).sum()))
                one["mask"] = image.score(segment(field, .45, 7.), case.truth)
            results[label] = one
        write(path, dict(code=code, source=row["sha"], modality=kind, role=role,
                         protocol=sha(Path(__file__)), choices=results))
        print(role, kind, code, {k:v.get("foreground_hits") for k,v in results.items()}, flush=True)


if __name__ == "__main__":
    os.umask(0o077)
    p = ArgumentParser()
    p.add_argument("role", choices=("dev", "unlabelled"))
    p.add_argument("kind", choices=("A", "B"))
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--shards", type=int, default=1)
    a = p.parse_args()
    run(a.role, a.kind, a.shard, a.shards)
