"""Matched-report, empty-report and identical-target replay checks in reserved T patients."""
import copy
import csv
import hashlib
import json
import math
import os
import random
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import torch
from torch.nn import functional as F
from audit_pairs import canonical_section, identity
from fetch_four import sha256, write_json
from fetch_manifest import FILES
from real_pair_smoke import metadata_for, read_cube, text_ids
from report_strata import describe_report
from volume_benchmark import ReportUNet


def step(net, optimizer, image, target, empty):
    optimizer.zero_grad(set_to_none=True)
    loss = F.binary_cross_entropy_with_logits(net(image, empty), target)
    if not torch.isfinite(loss):
        raise RuntimeError("Nonfinite loss during counterfactual replay")
    loss.backward()
    optimizer.step()


def word_count(report):
    return sum(len(s.split()) for s in report.values())


def read_reports(allowed):
    texts = {}
    for rel in FILES["ct_rate"][2]:
        with (Path("metadata/ct_rate") / rel).open(encoding="utf-8-sig", newline="") as f:
            for row in csv.DictReader(f):
                name = row["VolumeName"].strip()
                if name in allowed:
                    if name in texts:
                        raise ValueError("Duplicate primary report")
                    texts[name] = {key: canonical_section(row[key]) for key in ("Findings_EN", "Impressions_EN")}
    if set(texts) != allowed:
        raise ValueError("Missing training report")
    return texts


def fingerprint(state):
    digest = hashlib.sha256()
    for key, item in state.items():
        digest.update(key.encode("utf-8"))
        digest.update(item.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def main():
    seed = 891
    os.umask(0o077)
    torch.manual_seed(seed)
    random_state = random.Random(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.use_deterministic_algorithms(True, warn_only=True)
    root = Path("data/four_train")
    output = Path("phase1/replay")
    output.mkdir(parents=True, exist_ok=True)
    split = Path("phase1/cohort/split_v1/b10/T.csv")
    with split.open(newline="") as f:
        allowed = {r["volume"] for r in csv.DictReader(f)}
    reservation = json.loads((root / "pilot_T_reservation.json").read_text())
    names = reservation["volumes"]
    if reservation["role"] != "T_paid_engineering_pilot" or len(names) != 4 or not set(names) <= allowed:
        raise ValueError("Candidates must be the four T reservations in this split")
    reports = read_reports(allowed)
    tokens = {name: text_ids(row)[0] for name, row in reports.items()}
    states = {name: describe_report(row["Findings_EN"], row["Impressions_EN"])["state"] for name, row in reports.items()}
    donors = {}
    for n in names:
        matches = sorted(k for k in allowed if identity(k)[1] != identity(n)[1] and states[k] == states[n]
                         and abs(math.log(word_count(reports[k]) / word_count(reports[n]))) <= 0.25
                         and tokens[k] != tokens[n])
        donors[n] = random_state.sample(matches, min(2, len(matches)))
    records = json.loads((root / "four_header_audit.json").read_text())["cases"]
    params = metadata_for(set(names), root / "ct_rate/dataset/metadata")
    by_name = {row["name"]: row for row in records}
    images = [read_cube(root / "ct_rate" / by_name[n]["ct_release_path"], by_name[n], 64, *params[n]).cuda()
              for n in names]
    text_tensor = lambda n: torch.tensor([tokens[n]], device="cuda")
    empty = torch.zeros_like(text_tensor(names[0]))
    net = ReportUNet().cuda().train()
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-4)
    # Populate Adam moments before taking the common checkpoint.
    with torch.no_grad():
        warm_target = net(images[0], text_tensor(names[0])).sigmoid()
    step(net, optimizer, (images[0] + 0.02 * torch.randn_like(images[0])).clamp(0,1), warm_target, empty)
    checkpoint = copy.deepcopy(net.state_dict())
    moments = copy.deepcopy(optimizer.state_dict())
    if not moments["state"]:
        raise RuntimeError("Adam checkpoint is empty")
    teacher = copy.deepcopy(net).eval().requires_grad_(False)
    with torch.no_grad():
        cached = [teacher(x, empty).sigmoid() for x in images]
        strong = [(x + 0.02 * torch.randn_like(x)).clamp(0,1) for x in images]

    results = []
    for index, n in enumerate(names):
        x = images[index]
        order = [(index + i) % len(names) for i in range(4)]
        with torch.no_grad():
            truth = teacher(x, text_tensor(n)).sigmoid()
            targets = {"paired": truth, "same_tensor": truth.clone(),
                       "duplicate_report": teacher(x, text_tensor(n)).sigmoid(), "empty": cached[index]}
            for j, donor in enumerate(donors[n]):
                targets[f"matched_{j}"] = teacher(x, text_tensor(donor)).sigmoid()
        weights, first_states, last_states = {}, {}, {}
        for arm, target in targets.items():
            net.load_state_dict(checkpoint)
            optimizer.load_state_dict(copy.deepcopy(moments))
            for i, k in enumerate(order):
                step(net, optimizer, strong[k], target if i == 0 else cached[k], empty)
                if i == 0:
                    first_states[arm] = fingerprint(net.state_dict())
            last_states[arm] = fingerprint(net.state_dict())
            weights[arm] = net.output.weight.detach().cpu().clone()
        if not all(states_map["paired"] == states_map["same_tensor"] == states_map["duplicate_report"]
                   for states_map in (first_states, last_states)):
            raise RuntimeError("Identical targets produced different student state dictionaries")
        gaps = {arm: (target - truth).abs().mean().item() for arm, target in targets.items()}
        distances = {arm: (w - weights["paired"]).abs().max().item() for arm, w in weights.items()}
        if not all(math.isfinite(a) for a in (*gaps.values(), *distances.values())):
            raise RuntimeError("Nonfinite control diagnostic")
        results.append({"donor_count":len(donors[n]), "identical_all_parameters_at_K1_and_K4":True,
                        "teacher_probability_mae_from_paired":gaps, "K4_output_weight_linf_from_paired":distances})
    write_json(output / "donors.private.json", donors)
    write_json(output / "control_results.json", {
        "stage":"Real-input intervention implementation control in T; no reference losses or efficacy estimates.",
        "seed":seed, "T_split_sha256":sha256(split), "CT_only_student":True, "patch":64,
        "optimizer":"Adam with shared nonempty moments; four updates; one intervention, three shared cached replay targets",
        "matcher":"Different T patient; same report-derived effusion state and chest CT protocol; absolute log word-count ratio <= 0.25, distinct encoded report; up to 2 donors without replacement.",
        "pilot_text_encoder":"section-wise hashed token IDs, 31 lexical tokens per section",
        "candidate_count":len(results), "cases_in_reservation_order":results,
        "limitations":"Common randomly initialized model after one self-distillation warmup. No complete or partial mask input, no R1/R2, no clinical claim."})
    print("replay_controls_pass", len(results), sum(len(x) for x in donors.values()), flush=True)


if __name__ == "__main__":
    main()
