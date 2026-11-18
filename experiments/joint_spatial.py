"""Check whether labeled feature prototypes improve the teacher's spatial ranking."""
from argparse import ArgumentParser
from itertools import product
from pathlib import Path
import os

import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage as ndi

import thesis_model as image
from frozen_priors import read, sha, write
from joint_factorial import manifold
from joint_roi import ROOT, evidence_region, warm_models
from thesis_cohort import MANIFEST
from thesis_component_fast import segment
from thesis_lobe_model import LobeStore


OUT = ROOT / "spatial_prototype_audit"
WEIGHTS = (0., 1., 2., 4.)


@torch.no_grad()
def gather(model, rows, store, use_text):
    fg, bg = [], []
    for i, row in enumerate(rows):
        case = store.get(row, True)
        query = row["query"] if use_text else ""
        region = image.positive_region(case, np.random.default_rng(9801 + i))
        logits, features = manifold(model, case.features(region, query), query)
        prepared = F.normalize(features.reshape(-1, features.shape[-1]), dim=1)
        truth = case.truth[region]
        assert truth.sum() > 0
        y = torch.from_numpy(truth.ravel().copy()).to(image.DEVICE)
        fg.append(F.normalize(prepared[y].mean(0), dim=0))
        confusing = case.domain(query)[region] & ~ndi.binary_dilation(truth, iterations=2)
        indices = torch.from_numpy(np.flatnonzero(confusing)).to(image.DEVICE)
        uncertain = indices[logits.flatten()[indices].topk(min(256, len(indices))).indices]
        bg.append(F.normalize(prepared[uncertain].mean(0), dim=0))
    return torch.stack(fg), torch.stack(bg)


@torch.inference_mode()
def scores(model, case, query, anchors):
    shape = case.ct.shape
    focus = case.domain(query)
    result = torch.zeros(shape,device=image.DEVICE)
    match = torch.zeros_like(result)
    denom = torch.zeros_like(result)
    ax = torch.tensor(.2 + .8 * np.hanning(image.SIDE),device=image.DEVICE,dtype=torch.float32)
    weight = ax[:,None,None] * ax[None,:,None] * ax[None,None,:]
    for begin in product(*(image.windows(j) for j in shape)):
        region = image.slices(begin)
        if not focus[region].any():
            continue
        feat = case.features(region, query)
        z, h = manifold(model, feat, query)
        unit = F.normalize(h.reshape(-1, h.shape[-1]), dim=1)
        foreground, background = (torch.max(unit @ refs.T,1).values for refs in anchors)
        contrast = (foreground - background).reshape(z.shape[1:])
        result[region] += z[0] * weight
        match[region] += contrast * weight
        denom[region] += weight
    assert np.all(denom.cpu().numpy()[focus] > 0)
    norm = denom.clamp_min(1e-6)
    return (result / norm).cpu().numpy(), (match / norm).cpu().numpy(), focus


def run(modality):
    OUT.mkdir(parents=True,exist_ok=True)
    study = read(MANIFEST)
    visible = [r for r in study["train"] if r["code"] in study["labels"]["20"]]
    model = warm_models(modality)[0]
    data = LobeStore(limit=10)
    use_text = modality == "B"
    prototypes = gather(model, visible, data, use_text)
    for row in study["dev"]:
        code = row["code"]
        dest = OUT / (modality + "_" + code + ".json")
        if dest.exists():
            continue
        query = row["query"] if use_text else ""
        case = data.get(row, True)
        original, margin, valid = scores(model, case, query, prototypes)
        measures = {}
        for strength in WEIGHTS:
            field = original + strength * margin
            field[~valid] = -40.
            coords, certain = evidence_region(field, valid, 1024)
            subtruth = case.truth[image.slices(coords)]
            measures[str(strength)] = dict(hits=int((subtruth & certain).sum()), voxels=int(certain.sum()),
                                          coverage=float(subtruth.sum()/case.truth.sum()),
                                          dice=image.score(segment(field,.45,7.),case.truth)["dice"])
        write(dest,dict(code=code,source=row["sha"],program=sha(Path(__file__)),strengths=measures))
        print(modality,code,{k:v["hits"] for k,v in measures.items()},flush=True)


if __name__ == "__main__":
    os.umask(0o077)
    cli=ArgumentParser()
    cli.add_argument("modality",choices=("A","B"))
    run(cli.parse_args().modality)
