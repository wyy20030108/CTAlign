"""Supplementary training-only threshold calibration and a coarse-grid reconstruction diagnostic."""
import argparse,os,time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from finding_train import OUT,SEEDS,QueryNet,configure,read,write,sha,load_split

PROBS=np.r_[np.arange(1,20)/20.,.98,.99]
CUTS=np.log(PROBS/(1-PROBS)).astype(np.float32)
def dice_curve(indices,mask,n):
    pos=np.bincount(indices[mask],minlength=len(PROBS)+1)[1:][::-1].cumsum()[::-1]
    return 2*pos/(mask.sum()+n)

@torch.inference_mode()
def curves(net,row,arm,geometry):
    with np.load(row["path"]) as v:m=v["masks"].astype(bool)
    docs=["",""] if arm=="image" else row["queries"]
    logits=net(row["x"].cuda(),docs)
    correct=[];wrong=[];upper=[];vol=[]
    for i in range(2):
        up=F.interpolate(logits[i:i+1,None],row["shape"],mode="trilinear",align_corners=False)[0,0].cpu().numpy()
        ids=np.searchsorted(CUTS,up.ravel(),side="right").astype(np.uint8)
        counts=np.bincount(ids,minlength=len(PROBS)+1)[1:][::-1].cumsum()[::-1]
        correct.append(dice_curve(ids,m[i].ravel(),counts))
        wrong.append(dice_curve(ids,m[1-i].ravel(),counts))
        vol.append([int(m[i].sum()),int(counts[9])])
        if geometry:
            reconstructed=F.interpolate(row["frac"][i:i+1,None],row["shape"],mode="trilinear",align_corners=False)[0,0].numpy()
            pred=reconstructed>=.5
            upper.append(float(2*(pred&m[i]).sum()/(pred.sum()+m[i].sum())))
    return dict(code=row["code"],correct=np.mean(correct,axis=0).tolist(),swapped=np.mean(wrong,axis=0).tolist(),
                voxel_counts_true_and_pred05=vol,reconstruction_at_half=upper)

def main(part):
    sets={g:load_split(g) for g in ("fallback_train","fallback_val","swap_dev")}
    for seed in SEEDS:
        configure(seed)
        for arm in (("unpaired",) if part=="unpaired" else ("image","supervised","semi")):
            base=OUT/"unpaired_U"/str(seed) if arm=="unpaired" else OUT/str(seed)
            name="semi" if arm=="unpaired" else arm
            note=base/(name+".json");ck=base/(name+".pt")
            while not note.exists():time.sleep(2)
            target=base/(name+"_threshold.json")
            if target.exists(): assert sha(Path(__file__))==read(target)["script"];continue
            assert sha(ck)==read(note)["sha256"]
            net=QueryNet().cuda().eval();net.load_state_dict(torch.load(ck,map_location="cpu",weights_only=True)["net"])
            rows={}
            for g,data in sets.items():
                rows[g]=[curves(net,r,arm,seed==SEEDS[0] and arm=="image") for r in data]
                if g=="fallback_train":
                    # Select and freeze before calculating any evaluation curve; ties go to the lower threshold.
                    best=int(np.argmax(np.mean([r["correct"] for r in rows[g]],axis=0)))
                    write(base/(name+"_threshold_selection.json"),dict(index=best,threshold=PROBS[best],selection_group=g,patients=len(data)))
            write(target,dict(script=sha(Path(__file__)),seed=seed,arm=arm,thresholds=PROBS.tolist(),selected=best,rows=rows))
            print("threshold verified",seed,arm,round(PROBS[best],2),flush=True)
if __name__=="__main__":
    os.umask(0o077);p=argparse.ArgumentParser();p.add_argument("part",choices=("base","unpaired"));main(p.parse_args().part)
