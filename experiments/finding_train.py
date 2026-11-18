"""Whole-CT text-conditioned segmenter for a specified ReX finding."""
import argparse,copy,hashlib,json,os,re,sys
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from frozen_priors import read,sha,write
from replay_controls import fingerprint
from audit_pairs import identity
from swap_data import template

SRC=Path("phase3_swap/fallback")
OUT=Path("phase4_finding")
SEEDS=(891,1901,2707)
ARMS=("image","supervised","semi")
DESIGN=dict(seeds=SEEDS,arms=ARMS,epochs=16,updates_per_epoch=16,input_grid=[80]*3,adamw_lr=.001,
    image="CT HU lung and soft-tissue windows + fixed RAS normalized coordinates. Entire CT resized for input; evaluation mapped back to original 2 mm grid.",
    text="Hashed word tokens, bidirectional GRU and explicit signed left/right indicator. Test uses requested finding text; image-only receives identical null tokens.",
    fitting="Both queries from one labelled patient per update. Soft Dice against occupancy plus class-balanced logistic loss at occupied and unoccupied coarse cells. Each target is the published finding-specific mask.",
    semi="One unlabelled patient per update cycling through all 32 every two epochs. A visible clinical sentence is held fixed for the EMA teacher and photometrically augmented student. Mean squared score discrepancy in teacher top 1% and remaining voxels, with equal strata weight; ramp to 0.5 over 64 steps. No U masks.",
    evaluation="Fixed terminal online weights, threshold 0.5. Patient mean of two finding-specific Dice on the 2 mm grid. Compare correct, swapped and disease-template queries. Patient is the resampling unit; development only.")

def blocks(cin,cout):
    return nn.Sequential(nn.Conv3d(cin,cout,3,padding=1),nn.GroupNorm(4,cout),nn.SiLU(),
                         nn.Conv3d(cout,cout,3,padding=1),nn.GroupNorm(4,cout),nn.SiLU())

def tokens(query):
    words=re.findall(r"[a-z]+|\d+(?:\.\d+)?",query.lower())
    ids=[2+int(hashlib.sha256(w.encode()).hexdigest()[:8],16)%4094 for w in words] or [1]
    left,right="left" in words,"right" in words
    return ids,float(right)-float(left)

class QueryNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb=nn.Embedding(4096,32,padding_idx=0)
        self.gru=nn.GRU(32,24,bidirectional=True,batch_first=True)
        self.a=blocks(5,12);self.b=blocks(12,24);self.c=blocks(24,48)
        self.d=blocks(48+24,24);self.e=blocks(24+12,12)
        self.film=nn.Linear(49,96);self.head=nn.Linear(49,13);self.base=nn.Conv3d(12,1,1)
        self.side_weight=nn.Parameter(torch.zeros(()))
    def forward(self,ct,queries):
        # A single image with multiple candidate queries shares the complete visual encoder.
        encoded=[tokens(q) for q in queries]; seq=[torch.tensor(t,device=ct.device) for t,_ in encoded]
        packed=nn.utils.rnn.pack_padded_sequence(self.emb(nn.utils.rnn.pad_sequence(seq,batch_first=True)),
                  [len(v) for v in seq],batch_first=True,enforce_sorted=False)
        _,hs=self.gru(packed);side=ct.new_tensor([s for _,s in encoded]).view(-1,1)
        txt=torch.cat((hs[0],hs[1],side),dim=1)
        a=self.a(ct);b=self.b(F.avg_pool3d(a,2));c=self.c(F.avg_pool3d(b,2))
        gain,bias=self.film(txt).chunk(2,dim=1)
        c=c*(1+gain[:,:,None,None,None])+bias[:,:,None,None,None]
        def join(z,skip):
            return torch.cat((F.interpolate(z,size=skip.shape[2:],mode="trilinear",align_corners=False),
                              skip.expand(len(queries),-1,-1,-1,-1)),dim=1)
        e=self.e(join(self.d(join(c,b)),a))
        w=self.head(txt)
        return self.base(e)[:,0]+(e*w[:,:12,None,None,None]).sum(1)/12**.5+w[:,12,None,None,None]+self.side_weight*side[:,:,None,None]*ct[0,2]

def configure(seed):
    torch.set_num_threads(2);torch.manual_seed(seed);torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True;torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False

def load_split(role):
    receipt=read(SRC/"receipt.json");p=SRC/(role+".private.json")
    assert sha(p)==receipt["sha256_lists"][p.name]
    rows=read(p);result=[]
    axis=torch.linspace(-1,1,80);coords=torch.stack(torch.meshgrid(axis,axis,axis,indexing="ij"))
    for row in rows:
        u=role=="unlabelled_train";path=Path(row["file"] if u else row["path"])
        assert sha(path)==row["npz_sha256" if u else "sha256"]
        with np.load(path,allow_pickle=False) as z:
            ct=z["ct_hu" if u else "ct"];shape=ct.shape
            assert np.allclose(z["affine"][:3,:3],2*np.eye(3))
            small=F.interpolate(torch.from_numpy(ct.astype(np.float32))[None,None],size=(80,)*3,mode="trilinear",align_corners=False)[0]
            x=torch.cat((((small+1000)/1200).clamp(0,1),((small+150)/400).clamp(0,1),coords))[None]
            obj=dict(name=row["name"],x=x,shape=shape)
            if u:
                assert set(z.files)=={"affine","ct_hu"}
                sections=row["report"];sentences=[s.strip() for s in re.split(r'(?<=[.!?])\s+|[\r\n]+',sections["Findings_EN"]) if s.strip()]
                candidate=[s for s in sentences if template(s)!="abnormal finding" and not re.search(r'\b(no|not|without|absent|history|previous)\b',s,re.I)]
                obj.update(sentences=candidate or sentences or [sections["Impressions_EN"]],uses_finding_filter=bool(candidate))
            else:
                mask=torch.from_numpy(z["masks"].astype(np.float32))[:,None]
                # Fractional occupancy is retained even if a small finding fits within one cell.
                frac=F.adaptive_avg_pool3d(mask,(80,)*3)[:,0]
                assert torch.all(frac.sum((1,2,3))>0)
                obj.update(frac=frac,queries=row["queries"],templates=row["templates"],code=row["code"],path=path)
            result.append(obj)
    return result

def photo(x,rng):
    gain=float(rng.uniform(.88,1.12));offset=float(rng.uniform(-.045,.045))
    return torch.cat(((x[:,:2]*gain+offset).clamp(0,1),x[:,2:]),1)

def supervised_loss(z,y):
    p=z.sigmoid(); yy=y.flatten(1); pp=p.flatten(1)
    dice=(2*(pp*yy).sum(1)+1e-5)/(pp.sum(1)+yy.sum(1)+1e-5)
    # Average the background with its natural volume, weight positive term by fractional occupancy.
    pos=(F.softplus(-z).flatten(1)*yy).sum(1)/yy.sum(1)
    neg=(F.softplus(z).flatten(1)*(1-yy)).sum(1)/(1-yy).sum(1)
    return (1-dice+.25*pos+.75*neg).mean()

def train(seed,arms):
    configure(seed);t=load_split("fallback_train");u=load_split("unlabelled_train")
    assert len(t)==16 and len(u)==32
    assert not {identity(r["name"])[1] for r in t}&{identity(r["name"])[1] for r in u}
    rng=np.random.default_rng(seed)
    order=list(np.concatenate([rng.permutation(16) for _ in range(16)]))
    unorder=list(np.concatenate([rng.permutation(32) for _ in range(8)]))
    draws=rng.integers(2**31,size=(256,3))
    init=QueryNet().cuda(); sig=fingerprint(init.state_dict())
    dest=OUT/str(seed);dest.mkdir(parents=True,exist_ok=True)
    for arm in arms:
        ck=dest/(arm+".pt");note=dest/(arm+".json")
        if note.exists(): assert sha(ck)==read(note)["sha256"] and read(note)["script"]==sha(Path(__file__));continue
        net=copy.deepcopy(init);ema=copy.deepcopy(net).eval().requires_grad_(False)
        opt=torch.optim.AdamW(net.parameters(),lr=.001,weight_decay=.0001)
        trace=[];seen=set()
        for i,(ti,ui) in enumerate(zip(order,unorder)):
            row=t[ti];xt=photo(row["x"].cuda(),np.random.default_rng(draws[i,0]))
            q=row["queries"] if arm!="image" else ["",""]
            # Avoid repeated image-only decoder work while retaining the two independent target losses.
            logits=net(xt,q if arm!="image" else [""])
            sup=supervised_loss(logits.expand(2,-1,-1,-1),row["frac"].cuda())
            loss=sup;cons=sup.new_zeros(())
            if arm=="semi":
                d=u[ui];seen.add(d["name"]);uquery=[d["sentences"][int(draws[i,1])%len(d["sentences"])]]
                weak=d["x"].cuda();strong=photo(weak,np.random.default_rng(draws[i,2]))
                with torch.no_grad():target=ema(weak,uquery).sigmoid().flatten()
                estimated=net(strong,uquery).sigmoid().flatten()
                k=max(1,len(target)//100);top=target.topk(k).indices
                err=(estimated-target).square();high=err[top].sum()
                cons=.5*(high/k+(err.sum()-high)/(len(err)-k))
                loss=sup+.5*min(1,(i+1)/64)*cons
            opt.zero_grad(set_to_none=True);loss.backward()
            grad=torch.nn.utils.clip_grad_norm_(net.parameters(),5,error_if_nonfinite=True)
            assert torch.isfinite(loss) and grad.item()>0
            opt.step()
            if arm=="semi":
                with torch.no_grad():
                    for a,b in zip(ema.parameters(),net.parameters(),strict=True):a.lerp_(b,.01)
            trace.append([sup.item(),cons.item()])
        tmp=ck.with_suffix(".part");torch.save(dict(net=net.state_dict(),arm=arm,seed=seed,step=256),tmp);tmp.replace(ck)
        assert fingerprint(torch.load(ck,map_location="cpu",weights_only=True)["net"])==fingerprint(net.state_dict())
        write(note,dict(seed=seed,arm=arm,initial=sig,sha256=sha(ck),script=sha(Path(__file__)),trace=trace,steps=len(trace),
                   n_U=len(seen),terminal_weights=fingerprint(net.state_dict()),U_with_positive_sentence=sum(r["uses_finding_filter"] for r in u)))
        print("finished",seed,arm,flush=True)

if __name__=="__main__":
    os.umask(0o077);p=argparse.ArgumentParser();p.add_argument("seed",type=int,choices=SEEDS);p.add_argument("arms",nargs="+",choices=ARMS);args=p.parse_args()
    train(args.seed,args.arms)
