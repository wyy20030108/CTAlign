"""Attach frozen external lung-lobe priors to the same train/evaluation pipeline."""
from pathlib import Path
import numpy as np
from scipy import ndimage as nd
import torch

from thesis_model import Case,CaseStore,DEVICE
from thesis_expand_screen import ROOT
from thesis_cohort import site
from frozen_priors import read,sha

CODES={"left_upper":1,"left_lingular":1,"left_lower":2,"right_upper":3,"right_middle":4,"right_lower":5}
CACHE=ROOT/"lobe_anatomy"

class LobeCase(Case):
    def __init__(self,case,lobes):
        super().__init__(case.code,case.ct,case.truth,case.detail,case.side_masks,case.axes,case.anatomical,case.positive)
        self.lobes=lobes
        self.allowed={}
    def features(self,region,text):
        x=super().features(region,text)
        label=CODES.get(site({"query":text}))
        if label is not None:
            if label not in self.allowed:
                self.allowed[label]=nd.binary_dilation(self.lobes==label,iterations=4)
            exact=torch.from_numpy((self.lobes[region]==label).astype(np.float32))[None].to(DEVICE)
            near=torch.from_numpy(self.allowed[label][region].astype(np.float32))[None].to(DEVICE)
            # Preserve full ipsilateral coverage in case the external lobe model misses the lesion.
            x[:,7]=.35*x[:,7]+.45*exact+.20*near
        return x

class LobeStore(CaseStore):
    def get(self,row,labeled):
        value=super().get(row,labeled)
        if isinstance(value,LobeCase):return value
        file=CACHE/(row["code"]+".npz")
        rec=read(file.with_suffix(".json"))
        assert rec["source_sha"]==row["sha"] and sha(file)==rec["prediction_sha"]
        with np.load(file) as a:lobes=a["lobes"]
        assert value.ct.shape==lobes.shape
        new=LobeCase(value,lobes)
        self.cache[(row["code"],labeled)]=new
        return new

