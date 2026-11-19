"""Compare the new supervised recipe to the original sliding window on unchanged cases."""
from argparse import ArgumentParser
from pathlib import Path
import json
import os
import zipfile

import numpy as np
import torch

from atelectasis_sliding import GRID, OUT as SLIDING, score
from diag_localize import seal
from frozen_priors import read, sha, write
import thesis_ssl as training
from thesis_model import FindingNet, DEVICE, case_eval, cohort_pilot


BRIDGE = training.ROOT / "original_budget_bridge"
PROTOCOL = BRIDGE / "protocol.json"
KEYS = ("dice", "recall", "fp_ml", "miss")


def prepare():
    cases = cohort_pilot()
    selected = [row["code"] for row in cases["selected"]]
    old = {str(seed): sha(SLIDING / "b10" / str(seed) / "results.json") for seed in training.SEEDS}
    dev = [row["code"] for row in cases["development"]]
    seal(PROTOCOL, dict(labeled=selected, development=dev, historical=old, seeds=training.SEEDS,
                        updates=384, threshold=0.5, implementation=sha(Path(__file__))))


def evaluate(seed):
    fixed = read(PROTOCOL)
    assert fixed["implementation"] == sha(Path(__file__))
    cases = cohort_pilot()
    target = cases["selected"]
    dev = cases["development"]
    assert [row["code"] for row in target] == fixed["labeled"]
    assert [row["code"] for row in dev] == fixed["development"]
    source = SLIDING / "b10" / str(seed) / "results.json"
    assert sha(source) == fixed["historical"][str(seed)]
    baseline = read(source)
    assert set(baseline["train_codes"]) == set(fixed["labeled"])
    old = {r["code"]: r for r in baseline["rows"]}
    assert set(old) == set(fixed["development"])
    training.OUT = BRIDGE
    training.WARM = fixed["updates"]
    store = training.LobeStore(limit=12)
    weight = training.warm("matched", seed, "B", cases, target, store)
    assert len(read(weight.with_suffix(".json"))["loss"]) == fixed["updates"]
    model = FindingNet().to(DEVICE).eval()
    model.load_state_dict(torch.load(weight, weights_only=True, map_location=DEVICE))
    dest = BRIDGE / str(seed)
    dest.mkdir(exist_ok=True)
    cap = max(3.0, 3 * float(np.median([r["coords"] * GRID**3 / 1000 for r in target])))
    rows = []
    for row in dev:
        code = row["code"]
        case = store.get(row, True)
        prediction = dest / (code + ".npz")
        result, tiles = case_eval(model, row, case, row["query"], cap, prediction)
        prior_file = source.parent / (code + ".npz")
        assert sha(prior_file) == old[code]["prediction_sha"]
        with np.load(prior_file) as a, np.load(prediction) as b:
            assert tuple(a["shape"]) == tuple(b["shape"]) == case.truth.shape
            prior = np.unpackbits(a["B1"], count=case.truth.size).reshape(case.truth.shape).astype(bool)
            new = np.unpackbits(b["prediction"], count=case.truth.size).reshape(case.truth.shape).astype(bool)
            before = score(prior, case.truth)
            after = score(new, case.truth)
        historical = old[code]["metrics"]["B1"]["0.5"]
        assert all(before[k] == historical[k] and after[k] == result["0.5"][k] for k in KEYS)
        rows.append(dict(code=code, sliding=before, revised=after, tiles=tiles, sha=sha(prediction)))
        print(seed, code, *(round(after[k], 4) for k in KEYS), flush=True)
    write(dest / "results.json", dict(seed=seed, rows=rows, checkpoint=sha(weight), protocol=sha(PROTOCOL)))


def release():
    fixed = read(PROTOCOL)
    patients = []
    runs = [read(BRIDGE / str(seed) / "results.json") for seed in fixed["seeds"]]
    assert all(r["protocol"] == sha(PROTOCOL) for r in runs)
    for code in fixed["development"]:
        grouped = [next(r for r in run["rows"] if r["code"] == code) for run in runs]
        patients.append(dict(code=code, **{
            name: {m: float(np.mean([row[name][m] for row in grouped])) for m in KEYS}
            for name in ("sliding", "revised")}))
    bootstrap = np.random.default_rng(89190).integers(len(patients), size=(10000, len(patients)))
    comparison = {}
    for key in KEYS:
        old, new = (np.array([row[name][key] for row in patients]) for name in ("sliding", "revised"))
        difference = new - old
        comparison[key] = dict(sliding=float(old.mean()), revised=float(new.mean()), delta=float(difference.mean()),
                               ci95=np.quantile(difference[bootstrap].mean(axis=1), [0.025, 0.975]).tolist())
    out = BRIDGE / "summary.json"
    write(out, dict(protocol=fixed, per_patient=patients, comparison=comparison))
    archive = BRIDGE / "supervised_bridge.zip"
    paths = [p for p in BRIDGE.rglob("*") if p.is_file() and p.suffix != ".zip"]
    entries = {str(p.relative_to(BRIDGE)): p for p in paths}
    entries["code/thesis_supervised_bridge.py"] = Path(__file__)
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as z:
        for name, path in entries.items():
            z.write(path, name)
        z.writestr("SHA256.json", json.dumps({n: sha(p) for n, p in entries.items()}))
    print(json.dumps(comparison), sha(archive), flush=True)


if __name__ == "__main__":
    os.umask(0o077)
    parser = ArgumentParser()
    parser.add_argument("mode", choices=("lock", "run", "release"))
    parser.add_argument("--seed", type=int, choices=training.SEEDS)
    args = parser.parse_args()
    if args.mode == "lock":
        BRIDGE.mkdir(exist_ok=True)
        prepare()
    elif args.mode == "run":
        evaluate(args.seed)
    else:
        release()
