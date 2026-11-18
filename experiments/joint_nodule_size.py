"""Correct report-derived ellipsoid sizes using only the visible training masks."""
from pathlib import Path
from statistics import median

import numpy as np

from diag_localize import seal
from frozen_priors import read, sha
from joint_nodule_cohort import DEST
from joint_nodule_gate import report_volume


SIZE_CALIBRATION=DEST/"report_size_calibration.json"


def seal_size(cohort_path):
    cohort=read(cohort_path)
    labeled=[r for r in cohort["train"] if r["code"] in cohort["labels"]["20"]]
    volumes=[(report_volume(r["query"]),r["coords"]) for r in labeled]
    ratios=[float(np.log(observed/prediction)) for prediction,observed in volumes if prediction is not None]
    n=len(ratios)
    factor=float(np.clip(np.exp(median(ratios)*n/(n+6)),.25,4)) if n>=3 else 1.
    dev=[(report_volume(r["query"]),r["coords"]) for r in cohort["dev"]]
    pairs=[(x,y) for x,y in dev if x is not None]
    centre=median(y for _,y in volumes)
    metrics={name:float(median(abs(np.log(adjust(x)/observed)) for x,observed in pairs))
             for name,adjust in (("uncorrected",float),("constant",lambda _:centre),("corrected",lambda x:x*factor))}
    calibration=dict(factor=factor,training_measured=n,development_measured=len(pairs),development_log_error=metrics,
                     labels=[r["code"] for r in labeled],cohort=sha(cohort_path),code=sha(Path(__file__)),
                     rule="Median label-to-ellipsoid log ratio, shrunk by n/(n+6) and clipped to [0.25,4].")
    seal(SIZE_CALIBRATION,calibration)
    return calibration


def get_calibrator(cohort_path):
    cfg=read(SIZE_CALIBRATION)
    assert cfg["code"]==sha(Path(__file__)) and cfg["cohort"]==sha(cohort_path)
    def apply(query):
        value=report_volume(query)
        return None if value is None else cfg["factor"]*value
    return apply
