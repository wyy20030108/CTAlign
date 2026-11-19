"""Four-step matched interventions evaluated against independent image-only algorithmic R1 references."""
import copy
import hashlib
import itertools
import os
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from diagnose_whole_T import windows
from full_report_model import FullReportCodec,FullReportUNet,collate
from matched_two_crop import load_locked,prepare
from replay_controls import fingerprint,read_reports
from train_partial_baselines import read,write_json,sha256,digest


@torch.inference_mode()
def reference_loss(net,reference):
    ct,reference_mask=reference
    if min(ct.shape)<64: raise ValueError("Unsupported small reference scan")
    starts=[windows(n) for n in ct.shape]
    locations=list(itertools.product(*starts))
    total=np.zeros(ct.shape,dtype=np.float32)
    hits=np.zeros(ct.shape,dtype=np.uint16)
    for i in range(0,len(locations),4):
        cuts=[tuple(slice(a,a+64) for a in q) for q in locations[i:i+4]]
        data=np.stack([ct[c] for c in cuts]).astype("float32")
        data=np.clip((data+1000.)/1200.,0,1)
        prob=net(torch.from_numpy(data).cuda()[:,None]).sigmoid().squeeze(1).cpu().numpy()
        for key,values in zip(cuts,prob,strict=True):
            hits[key]+=1
            total[key]+=values
    if hits.min()==0: raise ValueError("Incomplete whole-volume inference")
    prob=np.clip(total/hits,1e-6,1-1e-6)
    # Work in float64 for averaging small between-intervention differences.
    if np.any(reference_mask):
        fg_loss=-np.mean(np.log(prob[reference_mask]),dtype=np.float64)
        bg_loss=-np.mean(np.log1p(-prob[~reference_mask]),dtype=np.float64)
        common=float(prob.sum(dtype=np.float64)+reference_mask.sum())
        dice=(2*prob[reference_mask].sum(dtype=np.float64)+1)/(common+1)
        return float(.5*(.5*(fg_loss+bg_loss))+.5*(1-dice))
    return float(-np.mean(np.log1p(-prob),dtype=np.float64))


def main():
    os.umask(0o077)
    torch.manual_seed(891)
    torch.set_num_threads(2)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cuda.matmul.allow_tf32=False
    root=Path("phase1/full_report_trainonly")
    out=root/"R1_algorithmic_replay"
    out.mkdir(exist_ok=True)
    src=root/"three_arms/paired.pt"
    if sha256(src)!=read(root/"three_arms/summary.json")["arms"]["paired"]["saved_sha256"]:
        raise ValueError("Wrong starting checkpoint")
    ck=torch.load(src,weights_only=True,map_location="cpu")
    words=read(root/"vocabulary.private.json")
    codec=FullReportCodec(words["words"],words["chunk_size"])
    cases,steps,common,lock,groups,_=load_locked(Path("phase1/baselines/partial_v1"),Path("phase1/baselines/sparse_hu_v1"))
    batches,ledger,_=prepare(cases,steps,common,lock,groups)
    if ck["design"]["actual_tensor_ledger_sha256"]!=digest(ledger):
        raise ValueError("Training replay data changed")
    names={role:[r["volume"] for r in lock["roles"][role]] for role in ["T","U"]}
    source_rows=read(Path("phase1/donor_support/training_donors.private.json"))
    donors={r["volume"]:r["tiers"]["coarse"]["selected"] for r in source_rows if r["volume"] in names["U"]}
    if min(len(v) for v in donors.values())<2:raise ValueError("Two donors per prespecified candidate required")
    texts=read_reports(set(names["T"]+names["U"]+[n for r in donors.values() for n in r]))
    tokens={n:codec.encode(v["Findings_EN"],v["Impressions_EN"]) for n,v in texts.items()}
    chosen=sorted(names["U"],key=lambda n:hashlib.sha256(("replay-origin-891:"+n).encode()).digest())[:2]
    panel=read(Path("phase1/R1_proxy4/manifest.private.json"))
    if len(panel)!=4:raise ValueError("Incomplete R1 engineering panel")
    references=[]
    for item in panel:
        path=Path(item["file"])
        if sha256(path)!=item["cache_sha256"]:raise ValueError("Reference derivative changed")
        with np.load(path,allow_pickle=False) as f: references.append((f["ct_hu"],f["AI_reference"].astype(bool)))
    plan={"snapshot_sha256":sha256(src),"script_sha256":sha256(Path(__file__)),"reference_panel_sha256":digest(panel),
          "source":"CT-only external algorithm generates R1 reference, no clinician adjudication; early engineering response only",
          "training_snapshot":"U paired-report three-arm run at 32 steps, before auxiliary Basel training; optimizer state restored",
          "candidate_selection":"First 2 hash-ranked among the 8 eligible, previously exposed U training patients. First two coarse disease-polarity and length matched donors from existing training-only roster.",
          "updates":"K=4, intervention only at step 1. Use exact same labelled objectives, actual patches, weak HU cues, augmentation and subsequent cached targets in every branch.",
          "evaluation":"Equal patient mean on first 4 R1 by frozen hash order. Whole volumes at 2mm; 64-cube stride48, mean overlap probabilities. For an AI-positive volume use equally weighted class-balanced BCE and soft Dice loss. For an AI-empty volume use background BCE. This measures AI agreement.",
          "K":4,"candidates":len(chosen),"reference_patients":len(panel),"branches":["paired","empty","donor1","donor2","duplicate"],
          "start_state_sha256":fingerprint(ck["student"])}
    write_json(out/"design.json",plan)
    teacher=FullReportUNet(codec.vocab_size).cuda().eval().requires_grad_(False)
    teacher.load_state_dict(ck["teacher"])
    results=[]
    for cnum,n in enumerate(chosen):
        first=next(i for i,s in enumerate(steps) if names["U"][s["U"]]==n)
        sequence=[(first+k)%len(steps) for k in range(4)]
        with torch.no_grad():
            weak=batches[first][4]
            targets={k:teacher(weak,collate([tokens[v]],"cuda") if v else None).sigmoid()
                     for k,v in {"paired":n,"empty":None,"donor1":donors[n][0],"donor2":donors[n][1],"duplicate":n}.items()}
            tail=[teacher(batches[j][4],collate([tokens[names["U"][steps[j]["U"]]]],"cuda")).sigmoid() for j in sequence[1:]]
        observed={}
        for arm in plan["branches"]:
            net=FullReportUNet(codec.vocab_size).cuda().train();net.load_state_dict(ck["student"])
            opt=torch.optim.Adam(net.parameters(),lr=1e-4);opt.load_state_dict(copy.deepcopy(ck["optimizer"]))
            clone_start=fingerprint(net.state_dict())
            if clone_start!=plan["start_state_sha256"]: raise ValueError("Intervention did not start from common state")
            for k,j in enumerate(sequence):
                x,fg,neg,strong,_,active=batches[j]
                t=tokens[names["T"][steps[j]["T"]]]
                rt=None if steps[j]["drop_T_text"] else collate([t,t],"cuda")
                opt.zero_grad(set_to_none=True)
                pred=net(x,rt)
                pos=F.softplus(-pred[:1][fg]).mean()
                labelled=.5*(pos+F.softplus(pred[1:][neg]).mean()) if active else pos
                target=targets[arm] if k==0 else tail[k-1]
                loss=labelled+F.mse_loss(net(strong).sigmoid(),target)
                loss.backward()
                g=torch.nn.utils.clip_grad_norm_(net.parameters(),float("inf"),error_if_nonfinite=True)
                if not torch.isfinite(loss) or not g>0: raise ValueError("Bad intervention gradient")
                opt.step()
            net.eval()
            state=fingerprint(net.state_dict())
            if arm=="duplicate" and observed["paired"]["state_sha256"]==state:
                # An independent state check guarantees identical deterministic inference.
                vals=observed["paired"]["by_reference"].copy()
            else:
                vals=[reference_loss(net,r) for r in references]
            observed[arm]={"state_sha256":state,"by_reference":vals,"mean_loss":float(np.mean(vals))}
        if observed["duplicate"]!=observed["paired"]:raise ValueError("Duplicate report control failed")
        own=observed["paired"]["mean_loss"]
        results.append({"case":cnum+1,"K":4,"results":observed,
                        "proxy_delta_use":observed["empty"]["mean_loss"]-own,
                        "proxy_delta_pair":.5*(observed["donor1"]["mean_loss"]+observed["donor2"]["mean_loss"])-own})
        write_json(out/"running.private.json",results)
        print("complete AI-referenced candidate",cnum+1,flush=True)
    write_json(out/"summary.json",{"design":plan,"completed_branches":len(results)*5,
        "duplicate_state_exact_in_all_cases":True,
        "candidate_effects":[{k:r[k] for k in ("case","proxy_delta_use","proxy_delta_pair")} for r in results],
        "interpretation":"Two directions of short-update agreement with a frozen AI grader; no clinical utility estimate, no selector fit, no R2 or E."})


if __name__=="__main__":
    main()
