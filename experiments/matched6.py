"""Matched known-location learning with patient-separated supervision and evaluation."""
import argparse, os, re
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from finding_train import SRC, SEEDS, QueryNet, configure, read, write, sha
from diag_localize import boxes, cube_sums, cases, seal
from finding_native import image, slices
from diag_common import loss_fn

OUT=Path("phase6_matched")
GROUPS=("fallback_train","fallback_val","swap_dev")
RULE=dict(seeds=list(SEEDS),epochs=64,patients=16,updates=1024,threshold=.5,learning_rate=.002,
          optimizer="AdamW; max gradient norm 5",loss="phase5 positive_weight; equal weight for each target across the union of cores",
          field="same ground-truth-constructed, unordered pair of 32-cube cores with 64-cube context per CT; duplicated voxels average logits; no box-query assignment input",
          primary="patient macro whole-target Dice and correct minus swapped; 10 then 8 reused development patients; seeds averaged within patient",
          advance="in each split whole-target improvement >=10 percentage points vs historical supervised oracle on identical patients, plus lower endpoint of paired patient bootstrap 95% interval for correct minus swapped above zero. Engineering screen only.",
          controls="empty, disease and explicit laterality template, clinical source where available, patient/target volume and explicit left/right sensitivity; S030 flagged train; S002 A flagged train")
def get_rows(group):
    path=SRC/(group+".private.json")
    assert sha(path)==read(SRC/"receipt.json")["sha256_lists"][path.name]
    cf=Path("phase3_swap/clinical")/(group+".private.json")
    clinical=read(cf) if cf.exists() else {}
    items=[]
    for r in read(path):
        p=Path(r["path"]);assert sha(p)==r["sha256"]
        with np.load(p) as z:ct,m=z["ct"],z["masks"].astype(bool)
        shape=np.array(ct.shape);lattice=boxes(shape)
        starts=sorted(set(tuple(lattice[np.argmax(cube_sums(mask,lattice))]) for mask in m))
        flat=[];xs=[];inner=[]
        data=dict(ct=ct,ax=[np.linspace(-1,1,n,dtype=np.float32) for n in shape])
        for s in starts:
            pos=np.array(s);begin=np.clip(pos-16,0,shape-64)
            xs.append(image(data,slices(begin)).cpu())
            inner.append(tuple(slice(int(i),int(i)+32) for i in pos-begin))
            ijk=np.indices((32,32,32))+pos[:,None,None,None]
            flat.append(np.ravel_multi_index(ijk,shape).ravel())
        ids,inv,counts=np.unique(np.concatenate(flat),return_inverse=True,return_counts=True)
        q=r["queries"];sides=[" ".join(sorted(set(re.findall(r"\b(?:left|right)\b",t.lower())))) for t in q]
        items.append(dict(code=r["code"],source=sha(p),x=xs,inner=inner,shape=shape.tolist(),starts=np.asarray(starts).tolist(),
            ids=ids,inv=torch.tensor(inv).cuda(),counts=torch.tensor(counts).cuda(),m=torch.tensor(m.reshape(2,-1)[:,ids]).cuda(),
            total=m.sum((1,2,3)).tolist(),q=q,templates=[(t+" "+s).strip() for t,s in zip(r["templates"],sides)],
            sides=sides,clinical=clinical.get(r["code"])))
    return items
def forward(net,r,qs):
    cores=[]
    for x,inner in zip(r["x"],r["inner"]):
        cores.append(net(x.cuda(),qs)[(slice(None),*inner)].flatten(1))
    z=torch.cat(cores,dim=1)
    return z.new_zeros((len(qs),len(r["ids"]))).index_add(1,r["inv"],z)/r["counts"]
def run(seed,stage):
    configure(seed);dest=OUT/str(seed);dest.mkdir(exist_ok=True)
    ck=dest/"model.pt"
    if stage=="train":
        rows=get_rows(GROUPS[0]); net=QueryNet().cuda();opt=torch.optim.AdamW(net.parameters(),lr=.002);rng=np.random.default_rng(seed);history=[]
        for epoch in range(RULE["epochs"]):
            vals=[]
            for i in rng.permutation(len(rows)):
                r=rows[i];z=forward(net,r,r["q"]);loss=loss_fn(z,r["m"],"positive_weight");opt.zero_grad(set_to_none=True);loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(),5,error_if_nonfinite=True);opt.step();vals.append(loss.item())
            history.append(float(np.mean(vals)))
            if (epoch+1)%16==0:print(seed,epoch+1,history[-1],flush=True)
        torch.save(net.state_dict(),ck);write(dest/"training.json",dict(hash=sha(ck),loss=history))
    else:
        assert sha(ck)==read(dest/"training.json")["hash"]
        if stage=="swap_dev":assert (OUT/"development_lock.json").exists()
        rows=get_rows(stage);net=QueryNet().cuda().eval();net.load_state_dict(torch.load(ck,weights_only=True))
        folder=dest/stage;folder.mkdir(exist_ok=True);results=[]
        with torch.inference_mode():
            for r in rows:
                q=[*r["q"],"",*r["templates"],*(r["clinical"] or [])]
                # Query order has no information about candidate order; each query sees every core.
                pred=(torch.cat([forward(net,r,q[j:j+2]) for j in range(0,len(q),2)])>=0).cpu().numpy()
                m=r["m"].cpu().numpy()
                f=folder/(r["code"]+".npz")
                np.savez_compressed(f,pred=np.packbits(pred,axis=1),mask=np.packbits(m,axis=1),width=m.shape[1],starts=r["starts"],shape=r["shape"],total=r["total"])
                with np.load(f) as z:
                    pp=np.unpackbits(z["pred"],axis=1,count=int(z["width"]));mm=np.unpackbits(z["mask"],axis=1,count=int(z["width"]))
                    assert np.array_equal(pp,pred) and np.array_equal(mm,m)
                tp=(pp[:,None]&mm[None]).sum(-1);n=pp.sum(1);total=np.array(r["total"]);inside=mm.sum(1);assert (inside>0).all()
                results.append(dict(code=r["code"],total=r["total"],in_core=inside.tolist(),n=n.tolist(),intersect=tp.tolist(),dice=(2*tp/(n[:,None]+total)).tolist(),
                    local_dice=(2*tp/(n[:,None]+inside)).tolist(),sides=r["sides"],clinical=bool(r["clinical"]),source=r["source"],prediction=sha(f)))
        write(folder/"scores.json",results)
if __name__=="__main__":
    import sys
    os.umask(0o077);OUT.mkdir(exist_ok=True)
    if sys.argv[1]=="lock":
        assert len(cases())==18
        seal(OUT/"protocol.json",dict(rules=RULE,script=sha(Path(__file__))))
    else:
        assert read(OUT/"protocol.json")["script"]==sha(Path(__file__))
        run(int(sys.argv[1]),sys.argv[2])

