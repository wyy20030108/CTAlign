"""Frozen same-CT two-query challenge. Endpoints conditional on the two published regions."""
import argparse,itertools,os,re,time
from pathlib import Path
import numpy as np
from scipy.stats import rankdata
import torch
from bounded_residual import FullReportCodec,FullReportUNet,collate,correction,TEXT
from diagnose_whole_T import windows
from swap_data import out,read,sha256,write_json,read_reports,norm

def auc(a,b):
    ranks=rankdata(np.concatenate((a,b)))
    return float((ranks[:len(a)].sum()-len(a)*(len(a)+1)/2)/(len(a)*len(b)))

def clinical_pair(report,texts):
    sentences=list(dict.fromkeys(x.strip() for s in report.values() for x in re.split(r'(?<=[.!?])\s+|[\r\n]+',s) if x.strip()))
    matches=[[s for s in sentences if norm(t) in norm(s)] for t in texts]
    if all(len(m)==1 for m in matches) and matches[0]!=matches[1]:
        return [m[0] for m in matches]

@torch.inference_mode()
def test(net,codec,ct,masks,r,doc):
    a,b=masks.astype(bool)
    common=a&b
    a=a&~common;b=b&~common
    support=a|b
    sls=[tuple(slice(j,j+64) for j in xyz) for xyz in itertools.product(*(windows(n) for n in ct.shape))]
    sls=[s for s in sls if support[s].any()]
    queries=r["queries"]+r["templates"]+[""]
    cp=clinical_pair(doc,r["queries"])
    if cp: queries+=cp
    maps={}
    for q in dict.fromkeys(queries):
        sums=np.zeros(ct.shape,np.float32); hits=np.zeros(ct.shape,np.int16)
        encoded=codec.encode(q,"") if q else ()
        for i in range(0,len(sls),4):
            slices=sls[i:i+4]
            x=np.stack([ct[s] for s in slices]).astype(np.float32)
            x=torch.from_numpy(np.clip((x+1000)/1200,0,1)).cuda()[:,None]
            val=correction(net,x,collate([encoded]*len(slices),"cuda")).cpu().numpy()[:,0]
            for sl,v in zip(slices,val):
                sums[sl]+=v; hits[sl]+=1
        assert np.all(hits[support]>0)
        maps[q]=sums[support]/hits[support]
    side=a[support]
    def contrast(qa,qb):
        # The same frozen CT prior cancels exactly in the difference of the two pre-sigmoid scores.
        d=maps[qa]-maps[qb]
        u=auc(d[side],d[~side]); rev=auc(-d[side],-d[~side])
        assert abs(u+rev-1)<1e-9
        return {"auc":u,"preference_margin":float(d[side].mean()-d[~side].mean()),
                "mean_abs_change":float(np.abs(d).mean()),"swap_auc":rev}
    result=dict(finding=contrast(*r["queries"]),template=contrast(*r["templates"]),empty=contrast("",""),
                clinical=contrast(*cp) if cp else None)
    assert result["empty"]["auc"]==.5
    if set(r["sides"])=={"left","right"}:
        coords=np.nonzero(support)[0]*(1 if r["sides"][0]=="right" else -1)
        result["anatomical_side_control"]=auc(coords[side],coords[~side])
    return result

def main(seed):
    torch.set_num_threads(2);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
    base=Path("phase2/bounded")/str(seed)
    ck=base/"paired.pt";sign=sha256(ck);assert sign==read(base/"paired.json")["checkpoint_sha256"]
    state=torch.load(ck,map_location="cpu",weights_only=True)
    v=read(TEXT/"vocabulary.private.json");codec=FullReportCodec(v["words"],v["chunk_size"])
    net=FullReportUNet(codec.vocab_size).cuda().eval();net.load_state_dict(state["teacher"])
    dest=out/str(seed);dest.mkdir(exist_ok=True)
    rows=[r for r in read(out/"index.private.json")["pairs"] if r["role"]=="swap_dev"]
    docs=read_reports({r["name"] for r in rows})
    for r in rows:
        code=r["code"]; store=dest/(code+".json")
        if store.exists():
            assert read(store)["source_code"]==sha256(Path(__file__))
            continue
        receipt=out/(code+".json")
        while not receipt.is_file(): time.sleep(2)
        rec=read(receipt)
        if rec["status"]!="ok": continue
        p=out/(code+".npz");assert sha256(p)==rec["sha256"]
        with np.load(p) as data: val=test(net,codec,data["ct"],data["masks"],r,docs[r["name"]])
        val.update(code=code,seed=seed,checkpoint=sign,data_sha256=rec["sha256"],source_code=sha256(Path(__file__)))
        write_json(store,val);print(seed,code,flush=True)

if __name__=="__main__":
    os.umask(0o077);p=argparse.ArgumentParser();p.add_argument("seed",type=int);main(p.parse_args().seed)
