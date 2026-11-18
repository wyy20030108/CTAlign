"""Matched report-source training with frozen, CT-only spatial margins."""
import argparse
import copy
import os
from pathlib import Path
import sys

import numpy as np
import torch
from torch.nn import functional as F
from frozen_priors import read, roster, sha, write
from replay_controls import fingerprint, read_reports
from train_partial_baselines import allowed, digest, schedule
from audit_pairs import identity

TEXT = Path("phase1/full_report_trainonly")
sys.path.insert(0,str(TEXT.resolve()))
from full_report_model import FullReportCodec, FullReportUNet, collate

P2 = Path("phase2")
PTH = P2/"residual_protocol.json"
DIGEST = "59f418d74db5c634e117cea0bbe9123cea06595d6faf048a77536d7ef36e3e05"


def get_cases():
    prior_dir = P2/"prior_train"
    priors = {}
    for i in range(2):
        receipt = read(prior_dir/f"shard{i}.json")
        assert receipt["script_sha256"] == sha(Path("frozen_priors.py"))
        assert not priors.keys() & receipt["cases"].keys()
        priors.update(receipt["cases"])
    assert len(priors)==38
    hints = Path("phase1/baselines/sparse_hu_v1")
    hint_index = {r["volume"]:r for r in read(hints/"seed_counts.private.json")}
    donors_file = Path("phase1/donor_support/training_donors.private.json")
    donors = {r["volume"]:r["tiers"]["coarse"]["selected"] for r in read(donors_file)}
    allowed_u = allowed(Path("phase1/cohort/split_v1/b10/U_train.csv"))
    cases = {"T":[],"U":[]}
    names = set()
    active_donors = set()
    for code,name,path,signature in roster(Path.cwd()):
        item = priors[code]
        assert sha(path)==item["source_sha256"]==signature
        pfile = prior_dir/(code+".npz")
        assert sha(pfile)==item["file_sha256"]
        role = code[0]
        with np.load(path,allow_pickle=False) as v, np.load(pfile,allow_pickle=False) as m:
            ct, margin = v["ct_hu"],m["margin"]
            assert np.array_equal(m["affine"],v["affine"]) and ct.shape==margin.shape and min(ct.shape)>=64
            assert int((margin>0).sum())==item["hard_foreground"]
            marks = v["positive_mask"].astype(bool) if role=="T" else None
            if role=="U":
                assert set(v.files)=={"ct_hu","affine"}
        if role=="T":
            coords = hints/"hints_private"/(name.removesuffix(".nii.gz")+".npz")
            assert sha(coords)==hint_index[name]["hint_coordinates_sha256"]
            neg = np.zeros(ct.shape,dtype=bool)
            with np.load(coords,allow_pickle=False) as v:
                for g in ("gas","dense"):
                    neg[tuple(v[g].T)]=True
            assert not np.any(neg & marks)
            known = np.flatnonzero(marks)
            assert known.size
            other = []
        else:
            known,neg = None,None
            other = donors[name]
            assert set(other)<=allowed_u and all(identity(n)[1]!=identity(name)[1] for n in other)
            active_donors.update(other)
        names.add(name)
        band = np.flatnonzero(np.abs(margin)<=2)
        cases[role].append(dict(code=code,name=name,ct=ct,margin=margin,marks=marks,known=known,
                                neg=neg,band=band,donors=other))
    held_ids = {identity(r["volume"])[1] for r in read(Path("phase1/cross_patient_T7/input_manifest.private.json"))["cases"]}
    assert not {identity(n)[1] for n in names|active_donors} & held_ids
    vocab_file = TEXT/"vocabulary.private.json"
    assert sha(vocab_file)==read(TEXT/"fulltext_validation.json")["vocabulary_sha256"]
    vocab = read(vocab_file)
    codec = FullReportCodec(vocab["words"],vocab["chunk_size"])
    docs = {k:codec.encode(v["Findings_EN"],v["Impressions_EN"]) for k,v in read_reports(names|active_donors).items()}
    return cases,codec,docs


def patch(case, seed, aim, size):
    rng = np.random.default_rng(seed)
    shape = np.asarray(case["ct"].shape)
    candidates = case["known"] if aim=="positive" else case["band"]
    if candidates.size:
        center = np.asarray(np.unravel_index(rng.choice(candidates),shape))
        center += rng.integers(-12,13,size=3)
    else:
        center = shape//2
    start = np.clip(center-size//2,0,shape-size)
    sl = tuple(slice(a,a+size) for a in start)
    def tensor(a):
        return torch.from_numpy(np.ascontiguousarray(a)).cuda()[None,None]
    x = np.clip((case["ct"][sl].astype(np.float32)+1000)/1200,0,1)
    weak = tensor(x)
    strong = tensor(np.clip(x*rng.uniform(.90,1.10)+rng.uniform(-.04,.04)
                         +rng.normal(0,.025,x.shape).astype(np.float32),0,1).astype(np.float32))
    margin = tensor(case["margin"][sl].astype(np.float32))
    pos = tensor(case["marks"][sl]) if case["marks"] is not None else None
    neg = tensor(case["neg"][sl]) if case["neg"] is not None else None
    if aim=="positive":
        assert pos.any()
    return (weak,strong,margin,pos,neg),start.tolist()


def get_batches(cases,seed,plan):
    steps = schedule(cases,plan["updates"],seed)
    u_visits = [0]*len(cases["U"])
    t_visits = [0]*len(cases["T"])
    batches = []
    ledger = []
    for step in steps:
        ti,ui = step["T"],step["U"]
        t,u = cases["T"][ti],cases["U"][ui]
        aim = "positive" if t_visits[ti]%2==0 else "boundary"
        tpatch,tstart = patch(t,step["crop_T"],aim,plan["patch"])
        upatch,ustart = patch(u,step["crop_U"],"boundary",plan["patch"])
        other = u["donors"][u_visits[ui]%len(u["donors"])] if u["donors"] else None
        batches.append((tpatch,upatch,None if step["drop_T_text"] else t["name"],u["name"],other))
        ledger.append([step,t["code"],u["code"],tstart,ustart,aim,other])
        u_visits[ui]+=1
        t_visits[ti]+=1
    assert set(u_visits)=={4} and sum(s["drop_T_text"] for s in steps)==64
    return batches,ledger


def correction(net,ct,tokens=None):
    return 2 * torch.tanh(net(ct,tokens))


def train(seed):
    assert sha(PTH)==DIGEST
    plan=read(PTH)
    assert seed in plan["seeds"]
    out=P2/"bounded"/str(seed)
    out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(2)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cuda.matmul.allow_tf32=False
    cases,codec,docs=get_cases()
    batches,ledger=get_batches(cases,seed,plan)
    initial=FullReportUNet(codec.vocab_size).cuda()
    torch.nn.init.zeros_(initial.output.weight)
    torch.nn.init.zeros_(initial.output.bias)
    signature=fingerprint(initial.state_dict())
    config={"protocol_sha256":DIGEST,"schedule_sha256":digest(ledger),"seed":seed,"vocab_size":codec.vocab_size,
            "initial_sha256":signature,"model_code_sha256":sha(TEXT/"full_report_model.py"),
            "source_sha256":sha(Path(__file__)),"T":6,"U":32,"supported_updates":sum(r[-1] is not None for r in ledger),
            "frozen_priors":[sha(P2/"prior_train"/f"shard{i}.json") for i in range(2)]}
    if (out/"design.json").exists():
        assert read(out/"design.json")==config
    write(out/"design.json",config)
    def tokens(name):
        return collate([docs[name]],"cuda") if name is not None else None
    for arm in plan["arms"]:
        ck=out/(arm+".pt")
        record=out/(arm+".json")
        if record.exists():
            assert sha(ck)==read(record)["checkpoint_sha256"]
            continue
        net=copy.deepcopy(initial).train()
        teacher=copy.deepcopy(initial).eval().requires_grad_(False)
        opt=torch.optim.Adam(net.parameters(),lr=1e-4)
        stats=[]
        for tx,ux,t_name,u_name,other in batches:
            _,xt,at,pos,neg=tx
            weak,strong,au,_,_=ux
            source={"paired":u_name,"empty":None,"matched":other}[arm] if other is not None else None
            opt.zero_grad(set_to_none=True)
            rt=correction(net,xt,tokens(t_name))
            terms=[]
            if pos.any():
                terms.append(F.softplus(-(at+rt)[pos]).mean())
            if neg.any():
                terms.append(F.softplus((at+rt)[neg]).mean())
            supervised=torch.stack(terms).mean() if terms else rt.sum()*0
            with torch.no_grad():
                target=(au+correction(teacher,weak,tokens(source))).sigmoid()
            pred=(au+correction(net,strong)).sigmoid()
            band=au.abs()<=2
            err=(pred-target).square()
            cons=err[band].mean() if band.any() else err.mean()
            loss=supervised+.05*rt.square().mean()+cons
            loss.backward()
            grad=torch.nn.utils.clip_grad_norm_(net.parameters(),float("inf"),error_if_nonfinite=True)
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite training loss")
            opt.step()
            with torch.no_grad():
                for a,b in zip(teacher.parameters(),net.parameters(),strict=True):
                    a.lerp_(b,.01)
            stats.append([loss.item(),cons.item(),grad.item()])
        tmp=ck.with_suffix(".tmp")
        torch.save({"student":net.state_dict(),"teacher":teacher.state_dict(),"design":config,"arm":arm,
                    "steps":len(stats)},tmp)
        tmp.replace(ck)
        state=torch.load(ck,weights_only=True,map_location="cpu")
        assert fingerprint(state["student"])==fingerprint(net.state_dict())!=signature
        write(record,{"checkpoint_sha256":sha(ck),"student_sha256":fingerprint(state["student"]),
                      "steps":len(stats),"gradient_nonzero_steps":sum(s[2]>0 for s in stats),
                      "mean_U_mse":float(np.mean([s[1] for s in stats])),"last_loss":stats[-1][0]})
        print("verified training",seed,arm,len(stats),flush=True)
    assert len({read(out/(a+".json"))["student_sha256"] for a in plan["arms"]})==3


if __name__=="__main__":
    os.umask(0o077)
    p=argparse.ArgumentParser()
    p.add_argument("seed",type=int)
    train(p.parse_args().seed)
