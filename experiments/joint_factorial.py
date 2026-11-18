"""Factorial development experiment with text, adaptive transport and prototype controls."""
from argparse import ArgumentParser
from copy import deepcopy
from pathlib import Path
import os

import numpy as np
import torch
import torch.nn.functional as F

import thesis_model as image
from diag_localize import seal
from finding_train import configure
from frozen_priors import read, sha, write
from joint_roi import ROI, ROOT
from thesis_cohort import MANIFEST
from thesis_component_fast import segment
from thesis_lobe_model import LobeStore
from thesis_ssl import EMA, OUT as PREVIOUS, SEEDS, checkpoint, make_schedule, update


image.segment = segment
OUT = ROOT / "factorial"
OPTIONS = {"S": (False, False, False), "N": (True, False, False),
           "U": (True, True, False), "P": (True, True, True)}
STEPS = 384


def transport(p, prior, mode):
    """PRCV's fixed voxel marginal and adaptive KL target relaxation."""
    source = p.clamp(1e-5, 1 - 1e-5)
    mass = float(source.mean())
    expected = .95 if mode == "direct" else max(.006, 2.0 * prior)
    penalty = .1 + 9.9 * min(1.0, mass / expected)
    kernel = torch.stack((1 - source, source), dim=1).square().clamp_min(1e-10)
    column = kernel.new_tensor((1 - prior, prior))
    filt = torch.ones_like(column)
    for _ in range(5):
        row = 1 / (len(source) * (kernel @ filt).clamp_min(1e-10))
        filt = (column / (kernel.T @ row).clamp_min(1e-10)).pow(penalty / (penalty + .5))
    joint = kernel * filt
    result = joint[:, 1] / joint.sum(1)
    return result, penalty, mass, float(result.mean())


def sharpen(p):
    return p.square() / (p.square() + (1 - p).square()).clamp_min(1e-10)


def manifold(model, x, q):
    captured = []
    hook = model.e.register_forward_hook(lambda module, inputs, features: captured.append(features))
    try:
        logits = model(x, [q])
    finally:
        hook.remove()
    return logits, captured[0][0].permute(1, 2, 3, 0)


class Prototype:
    def __init__(self):
        self.centres = [None, None]

    def embeddings(self, features, masks):
        for mask in masks:
            indices = mask.flatten().nonzero(as_tuple=True)[0]
            if len(indices) > 256:
                indices = indices[torch.randperm(len(indices), device=indices.device)[:256]]
            yield F.normalize(features.reshape(-1, features.shape[-1])[indices], dim=1)

    def loss(self, features, masks, learn):
        samples = list(self.embeddings(features, masks))
        if learn:
            with torch.no_grad():
                for i, points in enumerate(samples):
                    if len(points):
                        centre = F.normalize(points.mean(0), dim=0)
                        known = self.centres[i]
                        self.centres[i] = centre if known is None else F.normalize(.99 * known + .01 * centre, dim=0)
        loss = features.new_zeros(())
        if all(v is not None for v in self.centres):
            bank = torch.stack(self.centres).detach()
            each = [F.cross_entropy(p @ bank.T / .5, torch.full((len(p),), i, device=p.device))
                    for i, p in enumerate(samples) if len(p)]
            if each:
                loss = torch.stack(each).mean()
        return loss


def labeled(model, case, q, rng, bank):
    region = image.positive_region(case, rng)
    neg_region = image.negative_region(case, q, rng)
    x = image.augment(case.features(region, q), rng)
    n = image.augment(case.features(neg_region, q), rng)
    label = torch.from_numpy(case.truth[region][None].copy()).to(image.DEVICE)
    valid = torch.from_numpy(case.domain(q)[region].copy()).to(image.DEVICE)
    ng = torch.from_numpy(case.truth[neg_region][None].copy()).to(image.DEVICE)
    if bank is not None:
        score, features = manifold(model, x, q)
        proto_loss = bank.loss(features, (valid & ~label[0], label[0]), learn=True)
    else:
        score = model(x, [q])
        proto_loss = score.new_zeros(())
    total = image.supervised(score, label) + .5 * image.supervised(model(n, [q]), ng)
    fraction = (label[0] & valid).sum().item() / max(1, valid.sum().item())
    return total + .06 * proto_loss, float(np.clip(fraction, .0001, .08))


@torch.no_grad()
def teacher_target(teacher, case, q, origin, rng, prior, use_uot, mode, cap):
    region = image.slices(origin)
    x = case.features(region, q)
    valid = torch.from_numpy(case.domain(q)[region].copy()).to(image.DEVICE)
    a = teacher(image.augment(x, rng), [q])[0].sigmoid()
    b = teacher(image.augment(x, rng), [q])[0].sigmoid()
    p = .5 * (a + b)
    stable = valid & ((a - b).abs() <= .15)
    if not bool(stable.any()):
        return None, (0, 0, None, None, None)
    if use_uot:
        value, strength, before, after = transport(p[stable], prior, mode)
        qprob = p.clone()
        qprob[stable] = value
    else:
        qprob = p
        strength = before = after = None
    soft = sharpen(qprob)
    pos = stable & (soft >= .72) & (p >= .40)
    n = int(pos.sum())
    if n < 12:
        return None, (0, n, strength, before, after)
    if n > cap:
        idx = pos.flatten().nonzero(as_tuple=True)[0]
        keep = idx[soft.flatten()[idx].topk(cap).indices]
        pos.fill_(False)
        pos.flatten()[keep] = True
    neg = stable & (soft <= .10) & (p <= .20)
    return (x, pos, neg, soft), (1, int(pos.sum()), strength, before, after)


def unsupervised(net, example, query, rng, bank):
    x, pos, neg, soft = example
    strong = image.augment(x, rng, strong=True)
    if bank is None:
        z = net(strong, [query])
        proto_loss = z.new_zeros(())
    else:
        z, features = manifold(net, strong, query)
        proto_loss = bank.loss(features, (neg, pos), learn=False)
    terms = F.binary_cross_entropy_with_logits(z[0], soft, reduction="none")
    fore = terms[pos].mean()
    bg = terms[neg].mean() if neg.any() else fore
    return .65 * fore + .35 * bg + .06 * proto_loss


def fit(modality, arm, seed):
    protocol = read(OUT / "protocol.json")
    assert protocol["code"] == sha(Path(__file__)) and protocol["cohort"] == sha(MANIFEST)
    cohort = read(MANIFEST)
    selected = set(cohort["labels"]["20"])
    labeled_rows = [r for r in cohort["train"] if r["code"] in selected]
    only_report = cohort["unlabelled"]
    assert len(labeled_rows) == 7 and len(only_report) == 36
    store = LobeStore(limit=56)
    text = modality == "B"
    semi, use_uot, prototypes = OPTIONS[arm]
    dest = OUT / modality / str(seed) / arm
    dest.mkdir(exist_ok=True, parents=True)
    receipt = dest / "development.json"
    ck = dest / "model.pt"
    if receipt.exists():
        assert read(receipt)["model"] == sha(ck)
        return
    start = PREVIOUS / "20" / str(seed) / "initial" / (modality + ".pt")
    assert sha(start) == read(start.with_suffix(".json"))["sha"]
    origins = {}
    if semi:
        for row in only_report:
            info = read(ROI / (modality + "_" + row["code"] + ".json"))
            assert info["source"] == row["sha"] and info["role"] == "unlabelled"
            origins[row["code"]] = info["choices"][protocol["candidate"][modality]]["origin"]
    configure(seed)
    net = image.FindingNet().to(image.DEVICE)
    net.load_state_dict(torch.load(start, map_location=image.DEVICE, weights_only=True))
    teacher = deepcopy(net).eval().requires_grad_(False)
    optimizer = torch.optim.AdamW(net.parameters(), lr=.0006, weight_decay=.0001)
    both = make_schedule(len(labeled_rows), STEPS, seed + 22591), make_schedule(len(only_report), STEPS, seed + 8828)
    bank = Prototype() if prototypes else None
    cap = max(3., 3 * float(np.median([r["volume_ml"] for r in labeled_rows])))
    quota = int(min(4., .65 * cap) * 1000 / image.GRID**3)
    records = []
    prior = None
    for i, (s, u) in enumerate(zip(*both, strict=True), 1):
        lab = labeled_rows[s]
        query = lab["query"] if text else ""
        rng = np.random.default_rng(seed * 16529 + i)
        obj, frac = labeled(net, store.get(lab, True), query, rng, bank)
        prior = frac if prior is None else .99 * prior + .01 * frac
        diagnostics = (0, 0, None, None, None)
        if semi:
            row = only_report[u]
            opposite = row["query"] if text else ""
            rng_u = np.random.default_rng(seed * 7321 + i)
            sample, diagnostics = teacher_target(teacher, store.get(row, False), opposite, origins[row["code"]],
                                                  rng_u, prior, use_uot, protocol["transport"], quota)
            if sample is not None:
                obj += .45 * min(1, i / 96) * unsupervised(net, sample, opposite, rng_u, bank)
        update(net, optimizer, obj)
        with torch.no_grad():
            for a, b in zip(teacher.parameters(), net.parameters(), strict=True):
                a.lerp_(b, 1 - EMA)
        records.append(dict(accepted=diagnostics[0], count=diagnostics[1], relaxation=diagnostics[2],
                            before=diagnostics[3], after=diagnostics[4], prior=prior, loss=float(obj.detach())))
        if i % 96 == 0:
            print(modality, arm, seed, i, sum(v["accepted"] for v in records[-96:]), records[-1]["loss"], flush=True)
    weight = checkpoint(ck, net)
    write(dest / "training.json", dict(records=records, source=sha(start), model=weight,
                                       mode=arm, kind=modality, seed=seed, cohort=sha(MANIFEST)))
    net.eval()
    scores = []
    for row in cohort["dev"]:
        query = row["query"] if text else ""
        result, _ = image.case_eval(net, row, store.get(row, True), query, cap)
        scores.append(dict(code=row["code"], value=result))
    write(receipt, dict(model=weight, score=scores, cap_ml=cap, protocol=sha(OUT / "protocol.json")))
    print("done", modality, arm, seed, flush=True)


def lock(candidate_a, candidate_b, mode):
    OUT.mkdir(exist_ok=True, parents=True)
    seal(OUT / "protocol.json", dict(cohort=sha(MANIFEST), code=sha(Path(__file__)),
                                     candidate={"A": candidate_a, "B": candidate_b}, transport=mode,
                                     labels=7, steps=STEPS, seeds=SEEDS, groups=OPTIONS,
                                     stage="development only", hypothesis="N0/U0/N1/U1 interaction; S group isolates unlabeled training and P isolates added prototypes"))


if __name__ == "__main__":
    os.umask(0o077)
    parser = ArgumentParser()
    parser.add_argument("stage", choices=("lock", "train"))
    parser.add_argument("--modality", choices=("A", "B"))
    parser.add_argument("--arms", default="SNUP")
    parser.add_argument("--seed", choices=SEEDS, type=int, default=SEEDS[0])
    parser.add_argument("--candidate-a", choices=("single", "mean", "agreement", "local"))
    parser.add_argument("--candidate-b", choices=("single", "mean", "agreement", "local"))
    parser.add_argument("--transport", choices=("direct", "conditioned"), default="conditioned")
    args = parser.parse_args()
    if args.stage == "lock":
        assert args.candidate_a and args.candidate_b
        lock(args.candidate_a, args.candidate_b, args.transport)
    else:
        assert args.modality and args.arms and set(args.arms) <= set(OPTIONS)
        for group in args.arms:
            fit(args.modality, group, args.seed)
