"""Check loss of thin/small labelled structures from a round trip through full-CT grids."""
import os
import numpy as np
import torch
from torch.nn import functional as F
from finding_train import SRC,OUT,read,write
def main():
    torch.set_num_threads(3)
    records=[]
    for g in ("fallback_train","fallback_val","swap_dev"):
        for r in read(SRC/(g+".private.json")):
            with np.load(r["path"]) as z:m=torch.from_numpy(z["masks"].astype(np.float32))[:,None]
            for side in (80,128,160):
                low=F.adaptive_avg_pool3d(m,(side,)*3)
                back=F.interpolate(low,size=m.shape[2:],mode="trilinear",align_corners=False)>=.5
                for i in (0,1):
                    truth=m[i].bool();p=back[i];n=int(p.sum())
                    records.append(dict(code=r["code"],group=g,query=i,side=side,truth_voxels=int(truth.sum()),
                                        recovered_voxels=n,dice=float(2*(p&truth).sum()/(p.sum()+truth.sum()))))
    write(OUT/"grid_qc.json",dict(scope="Reference-mask round trip; geometric diagnostic, not a performance bound or a model result.",cases=records))
    print("verified grid reconstruction",len(records),flush=True)
if __name__=="__main__":
    os.umask(0o077);main()
