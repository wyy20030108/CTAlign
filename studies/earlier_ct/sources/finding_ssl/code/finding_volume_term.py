"""Matched follow-up: constrain predicted soft finding volume on labelled training scans."""
import argparse,os,sys
from pathlib import Path
import torch
from torch.nn import functional as F
import finding_train as tr
import finding_eval as ev
import finding_unpaired as up

def run(seed,action,arms):
    root=tr.OUT/"volume_term"
    standard=tr.supervised_loss
    def with_volume(z,y):
        pred=z.sigmoid().sum((1,2,3))
        target=y.sum((1,2,3))
        size=F.smooth_l1_loss(torch.log(pred+1),torch.log(target+1))
        return standard(z,y)+.3*size
    tr.supervised_loss=with_volume;tr.OUT=root;ev.OUT=root
    root.mkdir(exist_ok=True,parents=True)
    for arm in arms:
        if arm=="unpaired":
            tr.OUT=root;ev.OUT=root
            up.main(seed,action)
        elif action=="train":
            tr.OUT=root;tr.train(seed,[arm])
        else:
            ev.OUT=root;ev.evals(seed,[arm])

if __name__=="__main__":
    os.umask(0o077);p=argparse.ArgumentParser();p.add_argument("seed",type=int,choices=tr.SEEDS)
    p.add_argument("action",choices=("train","evaluate"));p.add_argument("arms",nargs="+",choices=(*tr.ARMS,"unpaired"));a=p.parse_args();run(a.seed,a.action,a.arms)
