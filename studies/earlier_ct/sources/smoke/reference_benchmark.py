"""Synthetic, image-only sliding-window benchmark for the ReportUNet student."""

import argparse
from contextlib import nullcontext
import itertools
import json
import math
import statistics
import time

import torch

from volume_benchmark import ReportUNet


def window_starts(length, patch, stride):
    starts = list(range(0, length - patch + 1, stride))
    last = length - patch
    if starts[-1] != last:
        starts.append(last)
    return starts


def benchmark(args):
    shape, patch = tuple(args.shape), args.patch_size
    if len(shape) != 3 or patch < 16 or patch % 8 or min(shape) < patch:
        raise ValueError("Require three dimensions >= the patch size, and patch >= 16 divisible by 8.")
    if args.repeats < 1 or args.warmup < 0 or args.repeats + args.warmup < 2:
        raise ValueError("Need at least one measured and two total volumes to test repeat agreement.")
    device = torch.device(args.device)
    gpu = device.type == "cuda"
    stride = patch * 3 // 4
    starts = [window_starts(size, patch, stride) for size in shape]
    windows = list(itertools.product(*starts))
    torch.manual_seed(args.seed)
    torch.set_num_threads(min(4, torch.get_num_threads()))
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    model = ReportUNet().eval().to(device)
    volume = torch.randn((1, 1, *shape), device=device)
    empty_report = torch.zeros((1, 64), dtype=torch.long, device=device)
    sync = torch.cuda.synchronize if gpu else lambda: None
    sync()
    if gpu:
        torch.cuda.reset_peak_memory_stats(device)

    baseline, max_error = None, 0.0
    durations = []
    with torch.inference_mode():
        for run in range(args.warmup + args.repeats):
            sync()
            begin = time.perf_counter()
            summed = torch.zeros_like(volume, dtype=torch.float32)
            count = torch.zeros(shape, dtype=torch.int16, device=device)
            amp = torch.cuda.amp.autocast(dtype=torch.float16) if gpu else nullcontext()
            with amp:
                for coords in windows:
                    region = tuple(slice(i, i + patch) for i in coords)
                    cube = volume[(..., *region)]
                    prediction = model(cube, empty_report).float().sigmoid()
                    summed[(..., *region)].add_(prediction)
                    count[region].add_(1)
            min_count, max_count = count.amin().item(), count.amax().item()
            if min_count < 1:
                raise RuntimeError("Sliding-window inference left uncovered voxels.")
            summed.div_(count)
            sync()
            elapsed = time.perf_counter() - begin
            if run >= args.warmup:
                durations.append(elapsed)
            # Copy out after timing. Avoid retaining a second full image on GPU between runs.
            current = summed.cpu()
            if not torch.isfinite(current).all():
                raise RuntimeError("Non-finite synthetic prediction.")
            if baseline is None:
                baseline = current
            else:
                max_error = max(max_error, (baseline - current).abs().max().item())
            del summed, count, prediction
    return {
        "status": "pass" if max_error <= 1e-6 else "repeat_mismatch",
        "scope": "Synthetic on-device volume, untrained student, empty report, forward and average fusion only; excludes disk loading, transfer, training and reference-panel utility estimation.",
        "device": str(device), "hardware": torch.cuda.get_device_name(device) if gpu else "CPU",
        "torch": str(torch.__version__), "seed": args.seed, "amp": "fp16" if gpu else "off",
        "shape_dhw": shape, "patch_shape": [patch] * 3, "window_batch": 1,
        "nominal_overlap": 0.25, "stride": stride, "last_window_anchored": True,
        "window_starts_dhw": starts, "windows_per_volume": len(windows),
        "covered_voxels": math.prod(shape), "min_windows_per_voxel": min_count,
        "max_windows_per_voxel": max_count, "probability_fusion_dtype": "float32",
        "max_repeat_abs_probability_difference": max_error, "tolerance": 1e-6,
        "warmup_volumes": args.warmup, "measured_volumes": args.repeats,
        "seconds_per_volume": durations, "median_s_per_volume": statistics.median(durations),
        "peak_allocated_mib": torch.cuda.max_memory_allocated(device) / 2**20 if gpu else None,
        "peak_reserved_mib": torch.cuda.max_memory_reserved(device) / 2**20 if gpu else None,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--shape", type=int, nargs=3, default=(160, 256, 256), metavar=("D", "H", "W"))
    parser.add_argument("--patch-size", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=891)
    result = benchmark(parser.parse_args())
    print(json.dumps(result, indent=2, allow_nan=False))
    raise SystemExit(0 if result["status"] == "pass" else 2)
