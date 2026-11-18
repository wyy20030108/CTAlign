"""Evaluate locked endpoints at 2 mm, recording native-grid binary predictions for recount."""
import argparse,os,re,time
import numpy as np
import torch
from torch.nn import functional as F
from finding_train import OUT,SRC,SEEDS,ARMS,QueryNet,load_split,configure,read,write,sha
from pathlib import Path

def measure(p,m):
    pred=int(p.sum());truth=int(m.sum());both=int(np.count_nonzero(p&m))
    return dict(pred_voxels=pred,true_voxels=truth,intersection=both,dice=2*both/(pred+truth),coverage=both/truth)

@torch.inference_mode()
def infer(net,row,queries,mask,arm):
    key=["",""] if arm=="image" else queries
    z=net(row["x"].cuda(),key)
    mode=[]
    for i in range(2):
        high=F.interpolate(z[i:i+1,None],size=row["shape"],mode="trilinear",align_corners=False)[0,0]
        assert torch.isfinite(high).all()
        mode.append((high>=0).cpu().numpy())
    return np.stack(mode)

def evals(seed,arms):
    configure(seed)
    sets={r:load_split(r) for r in ("fallback_val","swap_dev")}
    train=read(SRC/"fallback_train.private.json");u=read(SRC/"unlabelled_train.private.json")
    from audit_pairs import identity
    names=[r["name"] for v in sets.values() for r in v]+[r["name"] for r in train]+[r["name"] for r in u]
    assert len(names)==len({identity(n)[1] for n in names})==66
    cp={}
    for g in sets:cp.update(read(Path("phase3_swap/clinical")/(g+".private.json")))
    for arm in arms:
        base=OUT/str(seed);record=base/(arm+".json")
        while not record.exists():time.sleep(2)
        trained=read(record);ck=base/(arm+".pt");assert trained["sha256"]==sha(ck)
        state=torch.load(ck,map_location="cpu",weights_only=True)
        assert state["step"]==256 and state["seed"]==seed and state["arm"]==arm
        net=QueryNet().cuda().eval();net.load_state_dict(state["net"])
        out=base/(arm+"_dev");out.mkdir(exist_ok=True)
        for g,items in sets.items():
            for r in items:
                code=r["code"];dest=out/(code+".json");pack=out/(code+".npz")
                if dest.exists():
                    assert read(dest)["map_sha256"]==sha(pack) and read(dest)["eval_sha256"]==sha(Path(__file__))
                    continue
                with np.load(r["path"]) as f:m=f["masks"];aff=f["affine"]
                conditions={"paired":r["queries"],"template":r["templates"],"empty":["",""]}
                if code in cp:conditions["clinical"]=cp[code]
                scores={};packed={};cache={}
                for mode,pair in conditions.items():
                    k=tuple(pair) if arm!="image" else ("","")
                    if k not in cache:
                        cache[k]=infer(net,r,list(k),m,arm)
                    p=cache[k]
                    scores[mode]=[measure(pp,mm) for pp,mm in zip(p,m)]
                    packed[mode]=np.packbits(p,bitorder="little")
                    if mode in ("paired","clinical","template"):
                        scores[mode+"_swapped"]=[measure(pp,mm) for pp,mm in zip(p[::-1],m)]
                np.savez_compressed(pack,affine=aff,shape=m.shape,**packed)
                write(dest,dict(code=code,group=g,arm=arm,seed=seed,checkpoint_sha256=sha(ck),
                    map_sha256=sha(pack),eval_sha256=sha(Path(__file__)),scores=scores,
                    location_index=any(re.search(r"\b(?:series|serial|image|slice)\s*\d",q,re.I) for q in r["queries"])))
            print("evaluated",seed,arm,g,len(items),flush=True)
if __name__=="__main__":
    os.umask(0o077);p=argparse.ArgumentParser();p.add_argument("seed",type=int,choices=SEEDS);p.add_argument("arms",nargs="+",choices=ARMS);a=p.parse_args();evals(a.seed,a.arms)
