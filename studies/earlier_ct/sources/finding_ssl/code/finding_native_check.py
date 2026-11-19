"""Test native inference calibration on training patients and actual memorisation of two locked training crops."""
import itertools,os,sys
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from finding_native import OUT,get,configure,QueryNet,image,pick,slices,windows,read,write,sha

@torch.inference_mode()
def grid(net,r):
    dims=r["ct"].shape;sums=np.zeros((2,*dims),np.float32);hits=np.zeros(dims,np.uint8)
    for start in itertools.product(*(windows(n) for n in dims)):
        sl=slices(start);sums[(slice(None),*sl)]+=net(image(r,sl),r["queries"]).cpu().numpy();hits[sl]+=1
    assert hits.min()>0;return sums/hits[None]

def calibration(arm):
    out=OUT/"native"/arm;net=QueryNet().cuda().eval()
    net.load_state_dict(torch.load(out/"model.pt",map_location="cpu",weights_only=True))
    ts=np.array([.001,.005,.01,.025,.05,.1,.15,.2,.25,.3,.35,.4,.45,.5,.6,.7,.8,.9,.95],np.float32)
    cuts=np.log(ts/(1-ts))
    runs={}
    for group in ("fallback_train","fallback_val","swap_dev"):
        rows=[]
        for r in get(group):
            z=grid(net,r);correct=[];wrong=[]
            for j in range(2):
                idx=np.searchsorted(cuts,z[j].ravel(),side="right").astype(np.uint8)
                counts=np.bincount(idx,minlength=len(ts)+1)[1:][::-1].cumsum()[::-1]
                for mode,mask in ((correct,r["m"][j]),(wrong,r["m"][1-j])):
                    inter=np.bincount(idx[mask.ravel()],minlength=len(ts)+1)[1:][::-1].cumsum()[::-1]
                    mode.append(2*inter/(counts+mask.sum()))
            rows.append(dict(code=r["code"],correct=np.mean(correct,axis=0).tolist(),swapped=np.mean(wrong,axis=0).tolist()))
            if group!="fallback_train":
                previous=read(out/(r["code"]+".json"))["scores"]["paired"]
                assert abs(np.mean([v["dice"] for v in previous])-rows[-1]["correct"][13])<1e-8
        runs[group]=rows
        if group=="fallback_train":
            best=int(np.argmax(np.mean([r["correct"] for r in rows],axis=0)))
            write(out/"native_selection.json",dict(p_actual=float(ts[best]),train_only=True,selected_index=best))
        print(arm,group,best,flush=True)
    write(out/"native_calibration.json",dict(thresholds=ts.tolist(),best=best,rows=runs,script=sha(Path(__file__))))

def memorise():
    train=get("fallback_train");r=train[0];rng=np.random.default_rng(891)
    sls=[pick(r,rng,j) for j in range(2)]
    net=QueryNet().cuda();opt=torch.optim.AdamW(net.parameters(),lr=.002)
    def calculate(step):
        total=0;metrics=[]
        for j,sl in enumerate(sls):
            m=torch.from_numpy(np.ascontiguousarray(r["m"][j][sl])).cuda()
            z=net(image(r,sl),[r["queries"][j]])[0]
            p=z.sigmoid(); soft=(2*p[m].sum()+1)/(p.sum()+m.sum()+1)
            # Strongly labelled two-crop positive control; no clinical performance inference.
            total=total+F.binary_cross_entropy_with_logits(z,m.float())-torch.log(soft)
            pred=z>=0;metrics.append(float(2*(pred&m).sum()/(pred.sum()+m.sum())))
        return total*.5,metrics
    vals={}
    for s in range(257):
        loss,metrics=calculate(s)
        if s in (0,32,128,256):vals[s]=metrics
        if s<256:
            opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),5,error_if_nonfinite=True);opt.step()
    write(OUT/"native/overfit_control.json",dict(code=r["code"],scores=vals,seed=891,n_train_patients=1,
        n_fixed_crops=2,scope="Same two fixed target-centred training crops used for scoring. Simpler loss for this engineering positive control.",script=sha(Path(__file__))))
    print("two-crop control",vals,flush=True)
if __name__=="__main__":
    os.umask(0o077);configure(891)
    if sys.argv[1]=="control":memorise()
    else:calibration(sys.argv[1])
