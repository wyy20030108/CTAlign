"""Memorisation check on one common native-grid field per training patient."""
import argparse,os,time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from finding_train import QueryNet,configure,read,write,sha,fingerprint
from finding_native import image

ROOT=Path("phase5_diag");OUT=ROOT/"common_clean";SELECTION=ROOT/"selection_clean.json"
RULE=dict(seed=891,net="unchanged phase4 QueryNet",modes=["reference","positive_weight"],
    margin_voxels=8,threshold=.5,visits=[256,512],optimizer="AdamW 0.002 max gradient norm 5",
    reference="BCE minus log soft Dice, macro across the two targets",
    positive_weight="Same loss plus 0.5*mean positive logistic loss for each target",
    selection="Independent of model outcomes; see selection.json. One model per treatment, all selected patients fitted jointly.",
    criterion="For every training target at threshold 0.5: correct Dice >=0.50, correct minus each off-diagonal comparator >=0.20 (row and column). Engineering gate only.",
    assessment="A, B and empty all see identical common tensor. Field is the union bounding box of both known masks plus context, rounded to a multiple of four. Empty text is a diagnostic lacking explicit training target.")

def common_field(shape,m):
    pos=np.argwhere(np.any(m,axis=0));lo=pos.min(0);hi=pos.max(0)+1
    sz=(np.maximum(hi-lo+16,32)+3)//4*4
    sz=np.minimum(sz,np.asarray(shape)//4*4)
    start=np.clip((lo+hi-sz)//2,0,np.asarray(shape)-sz)
    assert np.all(lo>=start) and np.all(hi<=start+sz)
    return start,tuple(slice(a,a+b) for a,b in zip(start,sz))

def load_selected():
    selection=SELECTION
    codes=read(selection)["codes"]
    manifest=Path("phase3_swap/fallback/fallback_train.private.json")
    assert sha(manifest)==read(manifest.parent/"receipt.json")["sha256_lists"][manifest.name]
    roster={r["code"]:r for r in read(manifest)};assert "S002" in codes and 2<=len(set(codes))==len(codes)<=4
    items=[]
    for code in codes:
        r=roster[code];path=Path(r["path"]);assert sha(path)==r["sha256"]
        with np.load(path) as v:ct,m,aff=v["ct"],v["masks"],v["affine"]
        start,sl=common_field(ct.shape,m)
        cropped=m[(slice(None),*sl)]
        assert np.all(cropped.sum((1,2,3))==m.sum((1,2,3))) and np.all(cropped.sum((1,2,3))>0)
        x=image(dict(ct=ct,ax=[np.linspace(-1,1,n,dtype=np.float32) for n in ct.shape]),sl)
        crop_aff=aff.copy();crop_aff[:3,3]=(aff@np.r_[start,1])[:3]
        items.append(dict(code=code,x=x,m=torch.from_numpy(np.ascontiguousarray(cropped)).cuda(),aff=crop_aff,
                          start=start.tolist(),queries=r["queries"],ct=ct[sl],source_sha=sha(path)))
    return items

def loss_fn(z,m,mode):
    terms=[]
    for l,y in zip(z,m):
        p=l.sigmoid();soft=(2*p[y].sum()+1)/(p.sum()+y.sum()+1)
        term=F.binary_cross_entropy_with_logits(l,y.float())-soft.log()
        if mode=="positive_weight":term=term+.5*F.softplus(-l[y]).mean()
        terms.append(term)
    return torch.stack(terms).mean()

def score(pred,m):
    return [[2*int((x&y).sum())/int(x.sum()+y.sum()) for y in m] for x in pred]

@torch.inference_mode()
def inspect_cases(net,items,dest,save):
    cases=[];net.eval()
    for row in items:
        m=row["m"].cpu().numpy();qs=row["queries"]
        probs=net(row["x"],[*qs,""]).sigmoid().cpu().numpy()
        assert np.isfinite(probs).all()
        matrix=score(probs>=.5,m)
        passed=all(matrix[j][j]>=.5 and matrix[j][j]-max(matrix[j][1-j],matrix[1-j][j])>=.2 for j in (0,1))
        rec=dict(code=row["code"],matrix=matrix,passed=passed,shape=list(m.shape[1:]),start=row["start"],
                 predicted_voxels=(probs>=.5).sum(axis=(1,2,3)).tolist(),true_voxels=m.sum(axis=(1,2,3)).tolist(),
                 source_sha256=row["source_sha"],inside_mean=[[float(p[y].mean()) for y in m] for p in probs])
        if save:
            f=dest/(row["code"]+".npz")
            np.savez_compressed(f,probability=probs,affine=row["aff"],start=np.asarray(row["start"]))
            with np.load(f) as z:assert score(z["probability"]>=.5,m)==matrix and np.array_equal(z["affine"],row["aff"])
            rec["map_sha256"]=sha(f)
        cases.append(rec)
    net.train()
    return cases

def main(mode):
    os.umask(0o077);configure(891);dest=OUT/mode;dest.mkdir(parents=True,exist_ok=True)
    if (dest/"summary.json").exists(): raise FileExistsError("Completed treatment already exists")
    items=load_selected();net=QueryNet().cuda()
    opt=torch.optim.AdamW(net.parameters(),lr=.002);init=fingerprint(net.state_dict());rng=np.random.default_rng(891)
    write(dest/"protocol.json",dict(rules=RULE,selection_sha=sha(SELECTION),script=sha(Path(__file__)),
            common_init=init,cases=[r["code"] for r in items],treatment=mode))
    history={0:inspect_cases(net,items,dest,False)};start=time.monotonic()
    for epoch in range(1,513):
        for j in rng.permutation(len(items)):
            row=items[j];loss=loss_fn(net(row["x"],row["queries"]),row["m"],mode)
            opt.zero_grad(set_to_none=True);loss.backward()
            grad=torch.nn.utils.clip_grad_norm_(net.parameters(),5,error_if_nonfinite=True)
            assert torch.isfinite(loss)
            opt.step()
        if epoch in (32,128,256,512):
            vals=inspect_cases(net,items,dest,False);history[epoch]=vals
            print(mode,epoch,[[round(v["matrix"][k][k],3) for k in (0,1)] for v in vals],flush=True)
            write(dest/"progress.json",dict(visits=epoch,cases=vals))
            if epoch>=256 and all(v["passed"] for v in vals):break
    final=inspect_cases(net,items,dest,True)
    ck=dest/"model.pt";torch.save(net.state_dict(),ck)
    assert fingerprint(torch.load(ck,map_location="cpu",weights_only=True))==fingerprint(net.state_dict())
    write(dest/"summary.json",dict(mode=mode,visits=epoch,steps=epoch*len(items),gate=all(r["passed"] for r in final),
          cases=final,history=history,initial=init,weights_sha256=sha(ck),script_sha256=sha(Path(__file__)),
          elapsed_seconds=time.monotonic()-start))

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("mode",choices=RULE["modes"]);main(p.parse_args().mode)
