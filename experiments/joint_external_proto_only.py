"""Test prototype regularization without transport under the same local teacher."""
from argparse import ArgumentParser
from pathlib import Path
import os

from diag_localize import seal
from frozen_priors import read,sha
import joint_external_refine as base
import joint_external_residual as edge
import joint_factorial as experiment
import joint_nodule_growth_refine as growth
from joint_nodule_run import COHORT,DEST
from thesis_ssl import SEEDS


OUT=DEST/"external_proto_only"


def run(mode,seed):
    experiment.MANIFEST,experiment.PREVIOUS,experiment.OUT,experiment.ROI=COHORT,base.WARM,OUT,base.ROI
    experiment.OPTIONS["R"]=(True,False,True)
    if mode=="lock":
        experiment.lock("agreement","agreement","conditioned")
        seal(OUT/"ablation.json",dict(source=sha(Path(__file__)),base=sha(base.ROOT/"protocol.json"),
              boundary_model=sha(Path(edge.__file__)),arm="B with prototype regularization and without transport"))
    else:
        scope=read(OUT/"ablation.json")
        assert scope["base"]==sha(base.ROOT/"protocol.json")
        assert scope["source"]==sha(Path(__file__)) and scope["boundary_model"]==sha(Path(edge.__file__))
        edge.install()
        growth.ROI=base.ROI
        experiment.teacher_target=growth.target
        experiment.fit("B","R",seed)


if __name__=="__main__":
    os.umask(0o077)
    p=ArgumentParser()
    p.add_argument("stage",choices=("lock","train"))
    p.add_argument("--seed",type=int,choices=SEEDS,default=SEEDS[0])
    a=p.parse_args()
    run(a.stage,a.seed)
