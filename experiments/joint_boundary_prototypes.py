"""Test whether local hard negatives make PRCV prototype refinement more useful."""
from argparse import ArgumentParser
from pathlib import Path
import os

import torch.nn.functional as F

from diag_localize import seal
from frozen_priors import read,sha
import joint_external_refine as base
import joint_factorial as experiment
import joint_nodule_gate as gate
import joint_nodule_growth_refine as growth
from joint_nodule_run import COHORT,DEST
from joint_nodule_size import get_calibrator
from thesis_ssl import SEEDS


OUT=DEST/"boundary_prototypes"


class BoundaryPrototype(experiment.Prototype):
    def loss(self,features,masks,learn):
        background,foreground=masks
        if foreground.any():
            surrounding=F.max_pool3d(foreground[None,None].float(),kernel_size=7,stride=1,padding=3)[0,0].bool()
            hard=background&surrounding
            if int(hard.sum())>=24:
                background=hard
        return super().loss(features,(background,foreground),learn)


def run(stage,kind,seed):
    experiment.MANIFEST,experiment.PREVIOUS,experiment.OUT,experiment.ROI=COHORT,base.WARM,OUT,base.ROI
    if stage=="lock":
        experiment.lock("agreement","agreement","conditioned")
        seal(OUT/"ablation.json",dict(base=sha(base.ROOT/"protocol.json"),code=sha(Path(__file__)),
              changed="Prototype samples from near-boundary background when at least 24 voxels exist. All other training and evaluation remain the frozen matched configuration."))
        return
    cfg=read(OUT/"ablation.json")
    assert cfg["code"]==sha(Path(__file__)) and cfg["base"]==sha(base.ROOT/"protocol.json")
    base.install()
    growth.ROI=base.ROI
    gate.report_volume=get_calibrator(COHORT)
    experiment.teacher_target=growth.target
    experiment.Prototype=BoundaryPrototype
    experiment.fit(kind,"P",seed)


if __name__=="__main__":
    os.umask(0o077)
    p=ArgumentParser()
    p.add_argument("stage",choices=("lock","train"))
    p.add_argument("--kind",choices=("A","B"),default="B")
    p.add_argument("--seed",type=int,choices=SEEDS,default=SEEDS[0])
    depth=p.parse_args()
    run(depth.stage, depth.kind, depth.seed)
