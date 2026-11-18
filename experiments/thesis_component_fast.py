"""Numerically identical faster connected-component reduction for full 3D logits."""
import numpy as np
from scipy import ndimage as nd
from atelectasis_b0 import GRID

CONN=np.ones((3,3,3),bool)
def segment(scores,threshold,cap_ml):
    high=scores>=np.log(threshold/(1-threshold))
    lohi=[]
    for axis in range(3):
        idx=np.flatnonzero(high.any(axis=tuple(i for i in range(3) if i!=axis)))
        if not len(idx):return high
        lohi.append(slice(int(idx[0]),int(idx[-1])+1))
    crop=tuple(lohi)
    foreground=high[crop]
    labels,n=nd.label(foreground,CONN)
    ids=labels[foreground]
    sizes=np.bincount(ids,minlength=n+1)
    good=np.flatnonzero(sizes>=6)
    if not len(good):
        high.fill(False);return high
    selected=scores[crop]
    peaks=np.full(n+1,-np.inf)
    np.maximum.at(peaks,ids,selected[foreground])
    winner=good[np.argmax(peaks[good]+.14*np.log1p(sizes[good]))]
    high.fill(False)
    high[crop]=labels==winner
    limit=int(cap_ml*1000/GRID**3)
    points=np.flatnonzero(high)
    if len(points)>limit:
        keep=np.argpartition(scores.ravel()[points],-limit)[-limit:]
        high.fill(False);high.ravel()[points[keep]]=True
    return high

