"""Synthetic-only check of the three-arm CT/report update protocol. Prints JSON."""

import argparse
from copy import deepcopy
import json
import math
import os
import platform
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
from torch import nn
from torch.nn import functional as F


class SmokeNet(nn.Module):
    """Small 3-D model with a trainable report-conditioned feature map."""

    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(nn.Conv3d(2, 8, 3, padding=1), nn.ReLU())
        self.tokens = nn.Embedding(6, 8)
        self.decoder = nn.Sequential(nn.Conv3d(8, 8, 3, padding=1), nn.ReLU(), nn.Conv3d(8, 1, 1))

    def forward(self, volume, report):
        context = self.tokens(report).mean(dim=1)
        return self.decoder(self.encoder(volume) + context[:, :, None, None, None])


def synthetic_case(side, grid):
    z, y, x = grid
    center_y = torch.rand((), device=x.device) * 0.2
    mask = ((x - 0.42 * side).square() * 2 + (y - center_y).square() + z.square() < 0.18)
    if side == 0:
        mask = torch.zeros_like(mask)
    mask = mask.float()
    # A fixed laterality channel and a noisy synthetic attenuation channel.
    volume = torch.stack((mask * 0.65 + torch.randn_like(x) * 0.3, x))[None]
    # [effusion, left/right, small] or [no, effusion, absent marker].
    words = [1, 2 if side < 0 else 3, 4] if side else [5, 1, 0]
    return volume, mask[None, None], torch.tensor([words], device=x.device)


def labeled_loss(logits, mask):
    bce = F.binary_cross_entropy_with_logits(logits, mask)
    dims = (1, 2, 3, 4)
    probability = logits.sigmoid()
    pos = mask.sum(dims) > 0
    loss = 1 - (2 * (probability * mask).sum(dims) + 1) / (probability.sum(dims) + mask.sum(dims) + 1)
    return bce + (loss * pos).sum() / pos.sum().clamp_min(1)


def run(args):
    if args.patch_size < 8:
        raise ValueError("patch-size must be at least 8")
    device = torch.device(("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device)
    torch.manual_seed(args.seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_num_threads(min(4, torch.get_num_threads()))

    def sync():
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    started = time.perf_counter()
    axis = torch.linspace(-1, 1, args.patch_size, device=device)
    grid = torch.meshgrid(axis, axis, axis, indexing="ij")
    train = [synthetic_case(side, grid) for side in (-1, 1, 0, -1, 1, 0)]
    candidate, _, true_text = synthetic_case(-1, grid)
    _, _, donor_text = synthetic_case(1, grid)
    reference = [synthetic_case(side, grid) for side in (-1, 1, 0)]
    empty = torch.zeros_like(true_text)
    online = SmokeNet().to(device)
    teacher = deepcopy(online).eval().requires_grad_(False)
    optim = torch.optim.Adam(online.parameters(), lr=0.002)
    for i, (volume, mask, text) in enumerate(train):
        optim.zero_grad(set_to_none=True)
        loss = labeled_loss(online(volume, text if i % 2 == 0 else empty), mask)
        loss.backward()
        optim.step()
        with torch.no_grad():
            for ema, current in zip(teacher.parameters(), online.parameters()):
                ema.mul_(0.9).add_(current, alpha=0.1)

    snapshot = deepcopy(online.state_dict())
    optim_snapshot = deepcopy(optim.state_dict())  # Includes populated Adam moments.
    with torch.no_grad():
        targets = {name: teacher(candidate, text).sigmoid() for name, text in
                   (("true", true_text), ("matched_donor", donor_text), ("empty", empty))}
        # Identical candidate view for the intervention; identical mixed replays thereafter.
        candidate_view = candidate + torch.cat((0.03 * torch.randn_like(candidate[:, :1]),
                                                torch.zeros_like(candidate[:, 1:])), dim=1)
        replay = []
        for i in range(3):
            unlabeled, _, txt = synthetic_case((-1, 1, 0)[i], grid)
            replay.append((train[i], unlabeled, teacher(unlabeled, txt).sigmoid()))

    def arm(target):
        online.load_state_dict(snapshot)
        optim.load_state_dict(deepcopy(optim_snapshot))
        torch.manual_seed(args.seed + 1)
        sync()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        start = time.perf_counter()
        optim.zero_grad(set_to_none=True)
        first = F.binary_cross_entropy_with_logits(online(candidate_view, empty), target)
        first.backward()
        optim.step()
        for (volume, mask, _), other, cached in replay:
            optim.zero_grad(set_to_none=True)
            loss = labeled_loss(online(volume, empty), mask)
            loss = loss + F.binary_cross_entropy_with_logits(online(other, empty), cached)
            loss.backward()
            optim.step()
        sync()
        eval_start = time.perf_counter()
        with torch.no_grad():
            scores = [labeled_loss(online(vol, empty), mask) for vol, mask, _ in reference]
            macro_loss = torch.stack(scores).mean().item()
        sync()
        if not math.isfinite(macro_loss):
            raise RuntimeError("Non-finite reference loss")
        return {"update_s": eval_start - start, "reference_s": time.perf_counter() - eval_start,
                "synthetic_reference_loss": macro_loss,
                "peak_allocated_mib": torch.cuda.max_memory_allocated(device) / 2**20 if device.type == "cuda" else None}

    first_run = arm(targets["true"])
    first_params = {key: val.detach().cpu().clone() for key, val in online.state_dict().items()}
    duplicate = arm(targets["true"])
    max_difference = max((val.cpu() - first_params[key]).abs().max().item() for key, val in online.state_dict().items())
    if max_difference != 0 or first_run["synthetic_reference_loss"] != duplicate["synthetic_reference_loss"]:
        raise RuntimeError("Identical-target replay failed determinism check")
    differences = {name: (tar - targets["true"]).abs().max().item() for name, tar in targets.items()}
    if any(not math.isfinite(value) or value <= 0 for key, value in differences.items() if key != "true"):
        raise RuntimeError("Text intervention produced no target change")
    rows = {"true": duplicate, "matched_donor": arm(targets["matched_donor"]), "empty": arm(targets["empty"])}
    return {"purpose": "synthetic engineering check only; no real CT, no scientific efficacy result",
            "status": "pass", "device": str(device), "hardware": torch.cuda.get_device_name(device) if device.type == "cuda" else platform.machine(),
            "torch": str(torch.__version__), "python": platform.python_version(), "cuda_runtime": torch.version.cuda,
            "seed": args.seed, "patch_shape": [args.patch_size] * 3, "steps_per_arm": 4, "synthetic_reference_cases": 3,
            "identical_target_max_parameter_difference": max_difference, "target_max_difference_vs_true": differences,
            "arms": rows, "total_s": time.perf_counter() - started}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--patch-size", type=int, default=24)
    parser.add_argument("--seed", type=int, default=891)
    print(json.dumps(run(parser.parse_args()), indent=2, allow_nan=False))
