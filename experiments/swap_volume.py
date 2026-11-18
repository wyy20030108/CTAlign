"""Equal-volume full-CT query exchange, with a true clinical-sentence subset."""
import os,sys,time,itertools
from pathlib import Path
import numpy as np
import torch
from swap_eval import clinical_pair,auc,windows,correction,collate,FullReportUNet,FullReportCodec,TEXT,out,read,write_json,read_reports,sha256

@torch.inference_mode()
def maps(net,codec,ct,prior,queries,a,b):
    starts=list(itertools.product(*(windows(n) for n in ct.shape)))
    support=a|b
    n=max(1,round(ct.size*.01))
    fields={}
    for q in dict.fromkeys(queries):
        sums=np.zeros(ct.shape,np.float32);hits=np.zeros(ct.shape,np.uint8)
        doc=codec.encode(q,"") if q else ()
        for i in range(0,len(starts),4):
            sls=[tuple(slice(j,j+64) for j in xyz) for xyz in starts[i:i+4]]
            batch=np.stack([ct[sl] for sl in sls]).astype(np.float32)
            x=torch.from_numpy(np.clip((batch+1000)/1200,0,1)).cuda()[:,None]
            vals=correction(net,x,collate([doc]*len(sls),"cuda")).cpu().numpy()[:,0]
            for sl,v in zip(sls,vals): sums[sl]+=v;hits[sl]+=1
        assert hits.min()>0
        sums/=hits
        margin=(sums+prior).ravel()
        threshold=np.partition(margin,-n)[-n]
        top=margin>threshold
        ties=n-int(top.sum())
        if ties:top[np.flatnonzero(margin==threshold)[:ties]]=True
        assert top.sum()==n
        fields[q]=(top.reshape(ct.shape)[support],sums[support])
    return fields,n

def run(role):
    torch.set_num_threads(2);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
    folder=out/"volume_final";folder.mkdir(exist_ok=True)
    source=out/"prior"/(role+".json")
    while not source.exists():time.sleep(2)
    priors=read(source)
    rows=[r for r in read(out/"index.private.json")["pairs"] if r["role"]==role and r["code"] in priors]
    docs=read_reports({r["name"] for r in rows})
    v=read(TEXT/"vocabulary.private.json");codec=FullReportCodec(v["words"],v["chunk_size"])
    for seed in (891,1901,2707):
        base=Path("phase2/bounded")/str(seed);ck=base/"paired.pt";assert sha256(ck)==read(base/"paired.json")["checkpoint_sha256"]
        net=FullReportUNet(codec.vocab_size).cuda().eval();net.load_state_dict(torch.load(ck,map_location="cpu",weights_only=True)["teacher"])
        for r in rows:
            code=r["code"];res=folder/f"{code}_{seed}.json"
            if res.exists():
                assert read(res)["script"]==sha256(Path(__file__))
                continue
            p=out/(code+".npz");f=out/"prior"/(code+".npz")
            assert sha256(p)==read(out/(code+".json"))["sha256"] and sha256(f)==priors[code]["file_sha256"]
            with np.load(p) as z,np.load(f) as zz:
                ct=z["ct"];x,y=z["masks"].astype(bool);a=x&~y;b=y&~x
                assert np.array_equal(z["affine"],zz["affine"]);pr=zz["margin"]
            controls={"finding":r["queries"],"template":r["templates"],"empty":["",""]}
            cp=clinical_pair(docs[r["name"]],r["queries"])
            if cp:controls["clinical"]=cp
            fields,k=maps(net,codec,ct,pr,[q for pair in controls.values() for q in pair],a,b)
            side=a[a|b];scores={}
            for mode,(qa,qb) in controls.items():
                (ta,ma),(tb,mb)=fields[qa],fields[qb]
                correct=(ta[side].mean()+tb[~side].mean())/2
                swapped=(ta[~side].mean()+tb[side].mean())/2
                aucval=auc((ma-mb)[side],(ma-mb)[~side])
                scores[mode]=dict(correct_equal_volume=float(correct),swapped_equal_volume=float(swapped),delta=float(correct-swapped),
                                  preference_auc=aucval,residual_difference=float(np.abs(ma-mb).mean()))
            assert scores["empty"]["delta"]==0 and scores["empty"]["preference_auc"]==.5
            result=dict(code=code,role=role,seed=seed,k_voxels=k,volume_voxels=ct.size,scores=scores,script=sha256(Path(__file__)))
            if role=="swap_dev":
                orig=read(out/str(seed)/(code+".json"))
                # Skipping non-overlapping tiles changes GPU batch composition. Measure its
                # numerical effect on ranking rather than using the masks to choose inference batches.
                result["roi_batch_auc_difference"]=scores["finding"]["preference_auc"]-orig["finding"]["auc"]
            if set(r["sides"])=={"left","right"}:
                xx=np.nonzero(a|b)[0]*(1 if r["sides"][0]=="right" else -1)
                result["coordinate_control_auc"]=auc(xx[side],xx[~side])
            write_json(res,result);print(seed,code,flush=True)
if __name__=="__main__":
    os.umask(0o077);run(sys.argv[1])
