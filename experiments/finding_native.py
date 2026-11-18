"""Exploratory native 2-mm patch pilot with full-field evaluation at a fixed threshold."""
import copy,itertools,os,sys
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from finding_train import QueryNet,configure,load_split,photo,SRC,OUT,sha,write,read,fingerprint
from finding_eval import measure
from diagnose_whole_T import windows

def get(role):
    data=load_split(role)
    for r,meta in zip(data,read(SRC/(role+".private.json")),strict=True):
        r.pop("x")
        path=r.get("path",meta.get("file"))
        with np.load(path) as f:
            r["ct"]=f["ct_hu" if role=="unlabelled_train" else "ct"]
            if "masks" in f:
                r["m"]=f["masks"];r["anchors"]=[np.flatnonzero(v) for v in r["m"]]
        r["ax"]=[np.linspace(-1,1,n,dtype=np.float32) for n in r["ct"].shape]
    return data

def slices(start):return tuple(slice(int(n),int(n)+64) for n in start)
def image(r,sl):
    ct=r["ct"][sl].astype(np.float32)
    xyz=np.meshgrid(*(a[s] for a,s in zip(r["ax"],sl)),indexing="ij")
    x=np.stack((np.clip((ct+1000)/1200,0,1),np.clip((ct+150)/400,0,1),*xyz))
    return torch.from_numpy(x).cuda()[None]

def pick(r,rng,anchor=None):
    shape=np.array(r["ct"].shape)
    if anchor is None:
        start=[rng.integers(n-64+1) for n in shape]
    else:
        point=np.array(np.unravel_index(rng.choice(r["anchors"][anchor]),shape))
        start=np.clip(point-rng.integers(12,53,3),0,shape-64)
    return slices(start)

def objective(logit,mask):
    terms=[]
    for z,m in zip(logit,mask):
        probs=z.sigmoid();neg=z[~m];k=max(1,len(neg)//100)
        base=F.binary_cross_entropy_with_logits(z,m.float())
        if len(neg):base=base+.12*F.softplus(neg.topk(k).values).mean()
        if m.any():
            dice=(2*probs[m].sum()+1)/(probs.sum()+m.sum()+1)
            base=base+.5*(1-dice)+.12*F.softplus(-z[m]).mean()
        terms.append(base)
    return torch.stack(terms).mean()

def train(arm,out,t,u):
    rng=np.random.default_rng(891);torch.manual_seed(891)
    order=np.concatenate([rng.permutation(16) for _ in range(16)]);us=np.concatenate([rng.permutation(32) for _ in range(8)])
    draws=rng.integers(2**31,size=(256,4))
    net=QueryNet().cuda();net.base.bias.data.fill_(-4.)
    sig=fingerprint(net.state_dict());teacher=copy.deepcopy(net).eval().requires_grad_(False)
    opt=torch.optim.AdamW(net.parameters(),lr=.001,weight_decay=.0001)
    trace=[]; used=set()
    for i,(ti,ui) in enumerate(zip(order,us)):
        r=t[ti];g=np.random.default_rng(draws[i,0])
        anchor=None if i%4==3 else int(g.integers(2))
        sl=pick(r,g,anchor)
        xt=photo(image(r,sl),g)
        label=torch.from_numpy(np.ascontiguousarray(r["m"][(slice(None),*sl)])).cuda()
        s=objective(net(xt,r["queries"]),label);c=s.new_zeros(())
        if arm=="semi":
            d=u[ui]; used.add(d["name"]);ug=np.random.default_rng(draws[i,1])
            weak=image(d,pick(d,ug));strong=photo(weak,ug)
            query=[d["sentences"][int(draws[i,2])%len(d["sentences"])]]
            with torch.no_grad(): target=teacher(weak,query).sigmoid().flatten()
            pred=net(strong,query).sigmoid().flatten();diff=(pred-target).square();k=len(diff)//100
            high=diff[target.topk(k).indices].sum();c=.5*(high/k+(diff.sum()-high)/(len(diff)-k))
        loss=s+.5*min((i+1)/64,1)*c;opt.zero_grad(set_to_none=True);loss.backward()
        grad=torch.nn.utils.clip_grad_norm_(net.parameters(),5,error_if_nonfinite=True);assert grad>0 and torch.isfinite(loss)
        opt.step()
        if arm=="semi":
            with torch.no_grad():
                for a,b in zip(teacher.parameters(),net.parameters(),strict=True):a.lerp_(b,.01)
        trace.append([s.item(),c.item()])
    tmp=out/"model.part";torch.save(net.state_dict(),tmp);tmp.replace(out/"model.pt")
    write(out/"training.json",dict(initial=sig,final=fingerprint(net.state_dict()),checkpoint=sha(out/"model.pt"),seed=891,steps=256,
          script=sha(Path(__file__)),U_patients=len(used),loss=trace))

@torch.inference_mode()
def evaluate(net,r,docs):
    shape=r["ct"].shape
    summed=np.zeros((2,*shape),np.float32);hits=np.zeros(shape,np.uint8)
    for start in itertools.product(*(windows(n) for n in shape)):
        sl=slices(start)
        logits=net(image(r,sl),docs)
        summed[(slice(None),*sl)]+=logits.cpu().numpy();hits[sl]+=1
    assert hits.min()>0 and np.isfinite(summed).all()
    return summed>=0 # sign of the average equals the sign of the sum

def main(arm):
    os.umask(0o077);configure(891);out=OUT/"native"/arm;out.mkdir(parents=True,exist_ok=True)
    if not (out/"training.json").exists():train(arm,out,get("fallback_train"),get("unlabelled_train"))
    entry=read(out/"training.json");assert entry["checkpoint"]==sha(out/"model.pt") and entry["script"]==sha(Path(__file__))
    net=QueryNet().cuda().eval();net.load_state_dict(torch.load(out/"model.pt",weights_only=True))
    assert fingerprint(net.state_dict())==entry["final"]
    rows=[]
    for g in ("fallback_val","swap_dev"):
        cp=read(Path("phase3_swap/clinical")/(g+".private.json"))
        for r in get(g):
            code=r["code"];f=out/(code+".npz");record=out/(code+".json")
            if not record.exists():
                scores={};maps={};conditions={"paired":r["queries"],"template":r["templates"]}
                if code in cp:conditions["clinical"]=cp[code]
                for mode,qs in conditions.items():
                    p=evaluate(net,r,qs);m=r["m"];maps[mode]=np.packbits(p,bitorder="little")
                    scores[mode]=[measure(mm,tt) for mm,tt in zip(p,m)]
                    scores[mode+"_swapped"]=[measure(mm,tt) for mm,tt in zip(p[::-1],m)]
                np.savez_compressed(f,shape=r["m"].shape,**maps)
                write(record,dict(code=code,group=g,scores=scores,sha256=sha(f)))
            row=read(record);assert sha(f)==row["sha256"]
            with np.load(f) as v:
                m=r["m"];assert tuple(v["shape"])==m.shape
                for mode in set(v.files)-{"shape"}:
                    p=np.unpackbits(v[mode],bitorder="little",count=m.size).reshape(m.shape).astype(bool)
                    assert [measure(mm,tt) for mm,tt in zip(p,m)]==row["scores"][mode]
                    assert [measure(mm,tt) for mm,tt in zip(p[::-1],m)]==row["scores"][mode+"_swapped"]
            rows.append(row);print(arm,code,flush=True)
    assert len(rows)==18
    write(out/"summary.json",dict(rows=rows,definition="Native-grid patch-trained model. Three of four labelled steps use a target-centred training crop. Entire-CT sliding windows at evaluation, mean logit and threshold 0. One seed; change also includes mining and loss."))
if __name__=="__main__":
    assert sys.argv[1] in ("supervised","semi");main(sys.argv[1])
