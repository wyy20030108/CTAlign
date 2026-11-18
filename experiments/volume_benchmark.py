"""Single-device synthetic 3-D CT/report training benchmark. No clinical data."""

import argparse
import copy
import json
import math
import platform
import statistics
import time
from contextlib import nullcontext

import torch
from torch import nn
from torch.nn import functional as F


CHANNELS = (24, 48, 96, 192)


def conv_block(n_in, n_out):
    return nn.Sequential(
        nn.Conv3d(n_in, n_out, 3, padding=1, bias=False),
        nn.GroupNorm(8, n_out),
        nn.LeakyReLU(0.1, inplace=True),
        nn.Conv3d(n_out, n_out, 3, padding=1, bias=False),
        nn.GroupNorm(8, n_out),
        nn.LeakyReLU(0.1, inplace=True),
    )


class ReportUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.ModuleList(conv_block(i, o) for i, o in zip((1,) + CHANNELS, CHANNELS))
        deep = CHANNELS[-1]
        self.embedding = nn.Embedding(512, deep)
        self.text_encoder = nn.GRU(deep, deep, batch_first=True)
        self.cross_attention = nn.MultiheadAttention(deep, 4, batch_first=True)
        self.norm = nn.LayerNorm(deep)
        self.up = nn.ModuleList(nn.ConvTranspose3d(i, o, 2, stride=2)
                                for i, o in zip(CHANNELS[:0:-1], CHANNELS[-2::-1]))
        self.decoder = nn.ModuleList(conv_block(c * 2, c) for c in CHANNELS[-2::-1])
        self.output = nn.Conv3d(CHANNELS[0], 1, 1)

    def forward(self, x, token_ids):
        skips = []
        for index, block in enumerate(self.encoder):
            x = block(x)
            if index < len(self.encoder) - 1:
                skips.append(x)
                x = F.max_pool3d(x, 2)
        size = x.shape
        voxels = x.flatten(2).transpose(1, 2)
        text, _ = self.text_encoder(self.embedding(token_ids))
        attended, _ = self.cross_attention(voxels, text, text, need_weights=False)
        x = self.norm(voxels + attended).transpose(1, 2).reshape(size)
        for up, block, skip in zip(self.up, self.decoder, reversed(skips)):
            x = block(torch.cat((up(x), skip), dim=1))
        return self.output(x)


def benchmark(args):
    if args.patch_size < 16 or args.patch_size % 8 or args.steps < 1 or args.warmup < 0:
        raise ValueError("Use patch-size >= 16 divisible by 8, steps >= 1 and warmup >= 0.")
    device = torch.device(args.device)
    gpu = device.type == "cuda"
    torch.set_num_threads(min(4, torch.get_num_threads()))
    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = False
    student = ReportUNet().to(device).train()
    teacher = copy.deepcopy(student).eval().requires_grad_(False)
    optimizer = torch.optim.AdamW(student.parameters(), lr=1e-4)
    scaler = torch.cuda.amp.GradScaler(enabled=gpu, init_scale=1024.0)
    size = (1, 1, args.patch_size, args.patch_size, args.patch_size)
    labeled = torch.randn(size, device=device)
    unlabeled = torch.randn_like(labeled)
    truth = (labeled > 1.5).float()
    report = torch.randint(1, 512, (2, 64), device=device)
    empty = torch.zeros_like(report)
    durations, skipped, measured_skipped = [], 0, 0
    loss_value = None
    sync = torch.cuda.synchronize if gpu else lambda: None
    sync()
    if gpu:
        torch.cuda.reset_peak_memory_stats(device)
    for step in range(args.warmup + args.steps):
        start = time.perf_counter()
        amp = torch.cuda.amp.autocast(dtype=torch.float16) if gpu else nullcontext()
        weak = unlabeled + 0.02 * torch.randn_like(unlabeled)
        strong = unlabeled + 0.15 * torch.randn_like(unlabeled)
        optimizer.zero_grad(set_to_none=True)
        with amp:
            with torch.no_grad():
                q_text = teacher(weak, report[1:]).sigmoid()
                q_empty = teacher(weak, empty[1:]).sigmoid()
                # Alternate interventions only to exercise both paths; no fitted selector here.
                target = q_text if step % 2 == 0 else q_empty
            tokens = torch.cat((report[:1] if step % 2 == 0 else empty[:1], empty[1:]))
            logits = student(torch.cat((labeled, strong)), tokens)
            labeled_logits, unlabeled_logits = logits.float().chunk(2)
            probability = labeled_logits.sigmoid()
            dice_loss = 1 - (2 * (probability * truth).sum() + 1) / (probability.sum() + truth.sum() + 1)
            loss = F.binary_cross_entropy_with_logits(labeled_logits, truth) + dice_loss
            loss += F.binary_cross_entropy_with_logits(unlabeled_logits, target.float())
        scaler.scale(loss).backward()
        scale_before = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()
        if scaler.get_scale() >= scale_before:
            with torch.no_grad():
                for p, s in zip(teacher.parameters(), student.parameters()):
                    p.lerp_(s, 0.01)
        else:
            skipped += 1
            if step >= args.warmup:
                measured_skipped += 1
        sync()
        duration = time.perf_counter() - start
        loss_value = loss.item()
        if not math.isfinite(loss_value):
            raise RuntimeError("Non-finite loss in synthetic training.")
        if step >= args.warmup:
            durations.append(duration)
    result = {
        "status": "pass" if not measured_skipped else "amp_step_skipped",
        "scope": "Synthetic in-memory patch training only. No file I/O, real reports, clinical masks, fitted selection rule, three-arm replay or whole-volume reference inference.",
        "hardware": torch.cuda.get_device_name(device) if gpu else platform.machine(),
        "gpu_total_mib": torch.cuda.get_device_properties(device).total_memory / 2**20 if gpu else None,
        "device": str(device), "torch": str(torch.__version__), "python": platform.python_version(),
        "cuda_runtime": torch.version.cuda, "seed": args.seed, "amp": "fp16" if gpu else "off",
        "patch_shape": list(size[2:]), "batch": {"labeled": 1, "unlabeled": 1},
        "architecture": {"channels": list(CHANNELS), "conv_per_stage": 2, "attention_heads": 4,
                         "text_encoder": "one-layer GRU", "report_tokens": 64, "teacher_forwards_per_step": 2,
                         "student_parameters": sum(p.numel() for p in student.parameters())},
        "warmup_steps": args.warmup, "measured_steps": args.steps,
        "step_seconds": durations, "median_step_s": statistics.median(durations),
        "mean_step_s": statistics.mean(durations), "per_second_student_patches": 2 / statistics.mean(durations),
        "peak_allocated_mib": torch.cuda.max_memory_allocated(device) / 2**20 if gpu else None,
        "peak_reserved_mib": torch.cuda.max_memory_reserved(device) / 2**20 if gpu else None,
        "amp_skipped_steps": skipped, "measured_skipped_steps": measured_skipped, "last_synthetic_loss": loss_value,
    }
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--patch-size", type=int, default=96)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--seed", type=int, default=891)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    result = benchmark(parser.parse_args())
    print(json.dumps(result, indent=2, allow_nan=False))
    raise SystemExit(0 if result["status"] == "pass" else 2)
