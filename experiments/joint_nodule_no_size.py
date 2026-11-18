"""Isolate Adaptive UOT from the extra report-derived size prior on the nodule task."""
from argparse import ArgumentParser
from pathlib import Path
import os

import joint_factorial as experiment
import joint_nodule_gate as geometry
from diag_localize import seal
from frozen_priors import read,sha
from joint_nodule_run import COHORT,DEST,ROIS,TRAIN,WARM
from thesis_ssl import SEEDS


OUT=DEST/"factorial_nosize"


def run(stage,seed):
    role=read(DEST/"lock.json")
    assert role["cohort"]==sha(COHORT)
    geometry_file=TRAIN/"shape_protocol.json"
    assert read(geometry_file)["geometry"]==sha(Path(geometry.__file__))
    experiment.MANIFEST,experiment.PREVIOUS,experiment.ROI,experiment.OUT=COHORT,WARM,ROIS,OUT
    if stage=="lock":
        experiment.lock("agreement","agreement","conditioned")
        seal(OUT/"ablation.json",dict(parent=sha(TRAIN/"protocol.json"),geometry=sha(geometry_file),
              disabled="only report-derived size prior",arm="text with Adaptive UOT",program=sha(Path(__file__))))
        return
    assert read(OUT/"ablation.json")["program"]==sha(Path(__file__))
    geometry.report_volume=lambda query:None
    geometry.activate(experiment)
    experiment.fit("B","U",seed)


if __name__=="__main__":
    os.umask(0o077)
    p=ArgumentParser()
    p.add_argument("stage",choices=("lock","train"))
    p.add_argument("--seed",type=int,choices=SEEDS,default=SEEDS[0])
    a=p.parse_args()
    run(a.stage,a.seed)
