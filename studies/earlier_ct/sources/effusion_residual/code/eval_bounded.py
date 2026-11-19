"""CT-only sliding-window evaluation and independent recount on already used development patients."""
import argparse
import itertools
import os
from pathlib import Path
import numpy as np
import torch
from bounded_residual import DIGEST, PTH, P2, TEXT, FullReportUNet, correction
from diagnose_whole_T import windows
from frozen_priors import read, sha, write
from trace_t7 import aggregate, score


@torch.inference_mode()
def predict(net, ct, margin):
    assert ct.shape==margin.shape and min(ct.shape)>=64
    sums=np.zeros(ct.shape,np.float32)
    visits=np.zeros(ct.shape,np.uint16)
    starts=list(itertools.product(*(windows(n) for n in ct.shape)))
    for i in range(0,len(starts),4):
        regions=[tuple(slice(a,a+64) for a in xyz) for xyz in starts[i:i+4]]
        batch=np.stack([ct[r] for r in regions]).astype(np.float32)
        x=torch.from_numpy(np.clip((batch+1000)/1200,0,1)).cuda()[:,None]
        residual=correction(net,x).squeeze(1).cpu().numpy()
        if not np.isfinite(residual).all() or np.abs(residual).max()>2:
            raise ValueError("Residual exceeded its prespecified bounds")
        for r,part in zip(regions,residual,strict=True):
            sums[r]+=part
            visits[r]+=1
    assert visits.min()>0
    delta=sums/visits
    mask=(margin.astype(np.float32)+delta)>0
    assert not np.any(mask & (margin<=-2)) and not np.any(~mask & (margin>2))
    return mask


def evaluate(seed):
    torch.set_num_threads(2)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cuda.matmul.allow_tf32=False
    assert sha(PTH)==DIGEST
    plan=read(PTH)
    assert seed in plan["seeds"]
    run=P2/"bounded"/str(seed)
    comp=read(P2/"development_7/comparator.json")
    assert comp["protocol_sha256"]==DIGEST
    manifest=read(Path("phase1/cross_patient_T7/input_manifest.private.json"))["cases"]
    cases=[]
    for row,base in zip(manifest,comp["cases"],strict=True):
        code=row["code"]
        assert code==base["code"]
        pack=Path("phase1/data/T_partial")/(row["volume"].removesuffix(".nii.gz")+".npz")
        mpath=P2/"development_7"/(code+".npz")
        assert sha(pack)==row["file_sha256"] and sha(mpath)==base["margin_sha256"]
        with np.load(pack,allow_pickle=False) as p,np.load(mpath,allow_pickle=False) as m:
            ct,known,margin,aff=p["ct_hu"],p["positive_mask"].astype(bool),m["margin"],p["affine"]
            assert np.array_equal(aff,m["affine"])
        voxel=abs(np.linalg.det(aff[:3,:3]))/1000
        assert ct.shape==known.shape==margin.shape and score(margin>0,known,voxel)==base["basel"]
        cases.append((code,ct,known,margin,aff,voxel))
    out=run/"development_7"
    out.mkdir(exist_ok=True)
    for arm in plan["arms"]:
        ck=run/(arm+".pt")
        signed=read(run/(arm+".json"))
        assert sha(ck)==signed["checkpoint_sha256"]
        store=torch.load(ck,map_location="cpu",weights_only=True)
        config=read(run/"design.json")
        assert store["design"]==config and store["arm"]==arm and store["steps"]==128
        assert config["model_code_sha256"]==sha(TEXT/"full_report_model.py") and config["protocol_sha256"]==DIGEST
        net=FullReportUNet(config["vocab_size"]).cuda().eval()
        net.load_state_dict(store["student"])
        dest=out/arm
        dest.mkdir(exist_ok=True)
        rows=[]
        for code,ct,known,margin,aff,voxel in cases:
            predicted=predict(net,ct,margin)
            path=dest/(code+".npz")
            np.savez_compressed(path,bits=np.packbits(predicted,bitorder="little"),affine=aff,shape=ct.shape)
            with np.load(path,allow_pickle=False) as f:
                assert tuple(f["shape"])==ct.shape and np.array_equal(f["affine"],aff)
                saved=np.unpackbits(f["bits"],bitorder="little",count=ct.size).reshape(ct.shape).astype(bool)
            assert np.array_equal(predicted,saved)
            base=margin>0
            row=score(saved,known,voxel)
            added=saved & ~base
            removed=base & ~saved
            row.update(code=code,map_sha256=sha(path),gained_known=int((added&known).sum()),
                       lost_known=int((removed&known).sum()),added_ml=float(added.sum()*voxel),
                       removed_ml=float(removed.sum()*voxel))
            rows.append(row)
        write(out/(arm+".json"),{"arm":arm,"seed":seed,"cases":rows,"summary":aggregate(rows),
              "protocol_sha256":DIGEST,"checkpoint_sha256":signed["checkpoint_sha256"],"script_sha256":sha(Path(__file__))})
        print("recounted 7 predictions",seed,arm,flush=True)


if __name__=="__main__":
    os.umask(0o077)
    p=argparse.ArgumentParser()
    p.add_argument("seed",type=int)
    evaluate(p.parse_args().seed)
