"""Three matched full-report runs; only change the text shown to the EMA teacher for U."""
import copy
import json
import os
import time
from pathlib import Path
import torch
from torch.nn import functional as F
from full_report_model import FullReportCodec, FullReportUNet, collate
from matched_two_crop import load_locked, prepare
from replay_controls import fingerprint, read_reports
from train_partial_baselines import digest, read, sha256, write_json, tensors


def run():
    os.umask(0o077)
    out=Path(__file__).resolve().parent/"three_arms"
    out.mkdir(parents=True,exist_ok=True)
    root=out.parent
    audit=read(root/"fulltext_validation.json")
    vocab=read(root/"vocabulary.private.json")
    if sha256(root/"vocabulary.private.json")!=audit["vocabulary_sha256"]:
        raise ValueError("Vocabulary changed after the audit")
    codec=FullReportCodec(vocab["words"],vocab["chunk_size"])
    torch.manual_seed(891)
    torch.set_num_threads(2)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cuda.matmul.allow_tf32=False
    ref=Path("phase1/baselines/partial_v1")
    coords=Path("phase1/baselines/sparse_hu_v1")
    cases,steps,common,lock,groups,_=load_locked(ref,coords)
    batches,ledger,_=prepare(cases,steps,common,lock,groups)
    names={role:[r["volume"] for r in lock["roles"][role]] for role in ["T","U"]}
    frozen=read(Path("phase1/donor_support/training_donors.private.json"))
    donors={r["volume"]:r["tiers"]["coarse"]["selected"] for r in frozen if r["volume"] in names["U"]}
    texts=read_reports(set(names["T"]+names["U"]+[d for choices in donors.values() for d in choices]))
    enc={k:codec.encode(v["Findings_EN"],v["Impressions_EN"]) for k,v in texts.items()}
    # Cycle the preregistered three donors, without drawing randomness inside an experimental arm.
    visits={n:0 for n in names["U"]}
    draw=[]
    for step in steps:
        n=names["U"][step["U"]]
        options=donors[n]
        draw.append(options[visits[n]%len(options)] if options else None)
        visits[n]+=1
    init=FullReportUNet(codec.vocab_size).cuda()
    design={"cohort_sha256":digest(lock),"actual_tensor_ledger_sha256":digest(ledger),
        "frozen_donor_sequence_sha256":digest(draw),"U_cases_with_donors":sum(bool(v) for v in donors.values()),
        "U_cases":len(names["U"]),"T_cases":len(names["T"]),"T_drop_steps":sum(s["drop_T_text"] for s in steps),
        "seed":891,"steps":len(steps),"vocab_size":codec.vocab_size,
        "vocab_sha256":audit["vocabulary_sha256"],"initial_weights_sha256":fingerprint(init.state_dict()),
        "branches":["empty","paired","matched"],"protocol":"Same T text and dropout in all branches, same supervised sparse positives and HU-derived weak negatives, actual T/U patches and augmentation. For U only change EMA teacher text. Online U student sees CT and learned null in every branch.",
        "training_code_sha256":sha256(Path(__file__)),"model_code_sha256":sha256(root/"full_report_model.py")}
    if (out/"design.json").exists() and read(out/"design.json")!=design:
        raise ValueError("Existing design changed")
    write_json(out/"design.json",design)
    write_json(out/"U_donors.private.json",dict(draw=draw,by_case=donors))
    metrics={}
    for arm in design["branches"]:
        model=copy.deepcopy(init).train()
        teacher=copy.deepcopy(init).eval().requires_grad_(False)
        opt=torch.optim.Adam(model.parameters(),lr=1e-4)
        rows=[]
        for i,((x,fg,neg,strong,weak,has_neg),step) in enumerate(zip(batches,steps,strict=True)):
            t=names["T"][step["T"]]
            u=names["U"][step["U"]]
            labelled=None if step["drop_T_text"] else collate([enc[t],enc[t]],"cuda")
            source={"empty":None,"paired":u,"matched":draw[i]}[arm]
            report=collate([enc[source]],"cuda") if source else None
            start=time.perf_counter()
            opt.zero_grad(set_to_none=True)
            forward=model(x,labelled)
            pos=F.softplus(-forward[:1][fg]).mean()
            neg_loss=F.softplus(forward[1:][neg]).mean() if has_neg else None
            with torch.no_grad():
                target=teacher(weak,report).sigmoid()
            cons=F.mse_loss(model(strong).sigmoid(),target)
            loss=((pos+neg_loss)*0.5 if has_neg else pos)+cons
            loss.backward()
            grad=torch.nn.utils.clip_grad_norm_(model.parameters(),float("inf"),error_if_nonfinite=True).item()
            if not grad>0 or not torch.isfinite(loss): raise ValueError("Invalid training step")
            opt.step()
            with torch.no_grad():
                for ema,student in zip(teacher.parameters(),model.parameters(),strict=True):
                    ema.lerp_(student,.01)
            torch.cuda.synchronize()
            rows.append([loss.item(),grad,time.perf_counter()-start])
        output=out/(arm+".pt")
        temporary=out/(arm+".tmp")
        torch.save({"student":model.state_dict(),"teacher":teacher.state_dict(),"optimizer":opt.state_dict(),
                    "arm":arm,"design":design,"vocab_size":codec.vocab_size},temporary)
        temporary.replace(output)
        loaded=torch.load(output,weights_only=True)
        restored=FullReportUNet(codec.vocab_size).cuda().eval()
        restored.load_state_dict(loaded["student"])
        model.eval()
        xprobe=tensors(cases["U"][0],10000,64,False)[0]
        with torch.inference_mode():
            if not torch.equal(restored(xprobe),model(xprobe)):
                raise ValueError("Reloaded CT-only logits changed")
        metrics[arm]={"saved_sha256":sha256(output),"loss_first":rows[0][0],"loss_last":rows[-1][0],
                "gradient_min":min(r[1] for r in rows),"steps":len(rows),"training_compute_s":sum(r[2] for r in rows),
                "reload_equal":True,"student_sha256":fingerprint(model.state_dict())}
        del restored,loaded,model,teacher,opt
        print("complete",arm,len(rows),flush=True)
    if len({r["student_sha256"] for r in metrics.values()})!=3:
        raise ValueError("Teacher source interventions produced identical trained models")
    write_json(out/"summary.json",{"design":design,"arms":metrics,"clinical_metrics_available":False})


if __name__=="__main__":
    run()
