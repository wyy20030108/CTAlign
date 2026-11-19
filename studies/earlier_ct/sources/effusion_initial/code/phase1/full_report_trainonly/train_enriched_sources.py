"""Fixed full-text source comparisons with prospectively enriched U training scans."""
import copy
import os
from collections import Counter
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from full_report_model import FullReportCodec,FullReportUNet,collate
from matched_two_crop import load_locked,prepare
from train_partial_baselines import read,write_json,sha256,digest,schedule
from replay_controls import read_reports,fingerprint
from test_ai_supervision import cues,weak_maps


def main():
    os.umask(0o077)
    torch.set_num_threads(2)
    torch.manual_seed(891)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cuda.matmul.allow_tf32=False
    root=Path("phase1/full_report_trainonly")
    dest=root/"U32_source_comparison"
    dest.mkdir(parents=True,exist_ok=True)
    vocab=read(root/"vocabulary.private.json")
    codec=FullReportCodec(vocab["words"],vocab["chunk_size"])
    cases,_,common,lock,groups,_=load_locked(Path("phase1/baselines/partial_v1"),Path("phase1/baselines/sparse_hu_v1"))
    tnames=[r["volume"] for r in lock["roles"]["T"]]
    source=Path("phase1/data/U_report_strata32/manifest.private.json")
    u_rows=read(source)["rows"]
    u_names=[r["name"] for r in u_rows]
    states={r["name"]:r["stratum"] for r in u_rows}
    if len(set(u_names))!=32 or set(u_names)&{r["volume"] for r in lock["roles"]["U"]}:
        raise ValueError("Incorrect enriched U membership")
    frozen=read(Path("phase1/donor_support/training_donors.private.json"))
    donors={r["volume"]:r["tiers"]["coarse"]["selected"] for r in frozen if r["volume"] in u_names}
    all_names=set(tnames+u_names+[n for v in donors.values() for n in v])
    full={k:codec.encode(v["Findings_EN"],v["Impressions_EN"]) for k,v in read_reports(all_names).items()}
    no_input=torch.zeros((1,1),device="cuda",dtype=torch.long)
    cases["U"]=[]
    for row in u_rows:
        path=Path(row["file"])
        if sha256(path)!=row["npz_sha256"]: raise ValueError("Invalid enriched U cache")
        with np.load(path,allow_pickle=False) as pack:
            if set(pack.files)!={"ct_hu","affine"} or not np.allclose(pack["affine"][:3,:3],np.eye(3)*2):
                raise ValueError("Wrong unlabelled spatial data")
            cases["U"].append((pack["ct_hu"],None,None,no_input))
    anchored=cues(cases,tnames)
    visits={k:0 for k in u_names}
    blocks=[]
    for block in range(4):
        sd=891+block
        steps=schedule(cases,32,sd)
        b,l,_=prepare(cases,steps,{**common,"seed":sd},lock,groups)
        weak_t=weak_maps(cases,anchored,steps,l,64)
        sources=[]
        for step in steps:
            key=u_names[step["U"]]; available=donors[key]
            sources.append(available[visits[key]%len(available)] if available else None)
            visits[key]+=1
        blocks.append((steps,b,weak_t,sources,digest(l)))
    if set(visits.values())!={4}: raise ValueError("Each U patient should contribute four updates")
    init=FullReportUNet(codec.vocab_size).cuda()
    initial=fingerprint(init.state_dict())
    if initial!=read(root/"three_arms/design.json")["initial_weights_sha256"]: raise ValueError("Inconsistent initialization")
    design={"T":len(tnames),"U":len(u_names),"steps":128,"epochs_over_U":4,"seed":891,"vocab_size":codec.vocab_size,
       "U_state_counts":dict(Counter(states.values())),"U_with_coarse_donors":sum(bool(v) for v in donors.values()),
       "U_source_sha256":sha256(source),"T_source_sha256":digest(lock["roles"]["T"]),"input_batch_sha256":[v[-1] for v in blocks],
       "donor_order_sha256":digest([v[3] for v in blocks]),"initial_weights_sha256":initial,"script_sha256":sha256(Path(__file__)),
       "scope":"Enriched training comparison with shared fixed external-AI auxiliary labels in six T. Same supervised complete text with 50% dropout in all arms. Only U teacher report is switched. Matched donor sampled by wording polarity and word count. If none is available all three conditions use empty text on that patient. U students and deployed models receive CT only. No R1/R2/E references used for fitting."}
    write_json(dest/"design.json",design)
    stats={}
    for arm in ("empty","paired","matched"):
        net=copy.deepcopy(init).train()
        teacher=copy.deepcopy(init).eval().requires_grad_(False)
        opt=torch.optim.Adam(net.parameters(),lr=1e-4)
        min_grad=float("inf");count=0
        for steps,batches,maps,selected,_ in blocks:
            for step,(x,pos,neg,strong,weak,active),h,other in zip(steps,batches,maps,selected,strict=True):
                t=full[tnames[step["T"]]]
                u=u_names[step["U"]]
                side=None if step["drop_T_text"] else collate([t,t],"cuda")
                k={"empty":None,"paired":u,"matched":other}[arm] if other else None
                opt.zero_grad(set_to_none=True)
                logit=net(x,side)
                p=F.softplus(-logit[:1][pos]).mean()
                labelled=.5*(p+F.softplus(logit[1:][neg]).mean()) if active else p
                aux=[]
                if (h==1).any():aux.append(F.softplus(-logit[h==1]).mean())
                if (h==2).any():aux.append(F.softplus(logit[h==2]).mean())
                weaklabel=torch.stack(aux).mean() if aux else logit.sum()*0
                with torch.no_grad(): q=teacher(weak,collate([full[k]],"cuda") if k else None).sigmoid()
                loss=labelled+.5*weaklabel+F.mse_loss(net(strong).sigmoid(),q)
                loss.backward()
                g=torch.nn.utils.clip_grad_norm_(net.parameters(),float("inf"),error_if_nonfinite=True).item()
                if not g>0 or not torch.isfinite(loss): raise ValueError("Nonfinite update")
                min_grad=min(min_grad,g)
                opt.step()
                with torch.no_grad():
                    for ema,online in zip(teacher.parameters(),net.parameters(),strict=True):ema.lerp_(online,.01)
                count+=1
        path=dest/(arm+".pt")
        tmp=path.with_suffix(".tmp")
        torch.save({"student":net.state_dict(),"teacher":teacher.state_dict(),"optimizer":opt.state_dict(),
                    "arm":arm,"design":design,"vocab_size":codec.vocab_size},tmp)
        tmp.replace(path)
        check=torch.load(path,weights_only=True,map_location="cpu")
        if fingerprint(check["student"])!=fingerprint(net.state_dict()):raise ValueError("Saved model differs")
        stats[arm]={"steps":count,"minimum_gradient_l2":min_grad,"sha256":sha256(path),
                    "student_sha256":fingerprint(net.state_dict()),"checkpoint_verified":True}
        print("enriched run completed",arm,count,flush=True)
    write_json(dest/"summary.json",{"design":design,"arms":stats})


if __name__=="__main__":
    main()
