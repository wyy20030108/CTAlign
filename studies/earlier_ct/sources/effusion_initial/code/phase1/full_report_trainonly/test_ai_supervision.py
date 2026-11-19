"""Single prespecified exploratory comparison of auxiliary CT-only algorithmic labels in T."""
import copy
import os
import time
from pathlib import Path
import nibabel as nib
import numpy as np
import torch
from scipy.ndimage import distance_transform_edt
from torch.nn import functional as F
from full_report_model import FullReportCodec, FullReportUNet, collate
from matched_two_crop import load_locked, prepare
from train_partial_baselines import read, write_json, sha256, digest, tensors, schedule
from replay_controls import fingerprint, read_reports


def cues(cases,training_names):
    base=Path("phase1/ai_contours/partial_six")
    rows={r["volume"]:r for r in read(base/"input_ledger.private.json")}
    locked=read(base/"prediction_lock.json")["files"]
    if set(training_names)!=set(rows): raise ValueError("Only the six T individuals with independent AI contours belong here")
    result=[]
    for case,name in zip(cases["T"],training_names,strict=True):
        path=base/"ai_binary"/(rows[name]["case"]+"_AI_effusion.nii.gz")
        if sha256(path)!=locked[path.name]:
            raise ValueError("AI reference changed since initial, CT-only generation")
        img=nib.load(path)
        if img.shape!=case[0].shape or nib.aff2axcodes(img.affine)!=("R","A","S"):
            raise ValueError("Reference shape or axes differ")
        with np.load(rows[name]["source_npz"],allow_pickle=False) as volume:
            if not np.allclose(img.affine,volume["affine"],rtol=0,atol=.001):
                raise ValueError("AI mask and training image occupy different physical grids")
        ai=np.asarray(img.dataobj)>0
        known=case[1].astype(bool)
        # Distances only define a fixed exclusion band. They are not model confidence scores.
        pos=(distance_transform_edt(ai,sampling=2.)>2) & ~known
        neg=(distance_transform_edt(~(ai|known),sampling=2.)>8)
        hint=np.zeros(ai.shape,dtype=np.uint8)
        hint[pos]=1
        hint[neg]=2
        if (hint[known]!=0).any(): raise ValueError("AI cue overlaps expert-positive objective")
        result.append(hint)
    return result


def weak_maps(training,all_hints,steps,ledger,size):
    maps=[]
    for step,row in zip(steps,ledger,strict=True):
        t=step["T"]
        ct,mask,anchors,_=training["T"][t]
        rng=np.random.default_rng(step["crop_T"])
        center=np.unravel_index(rng.choice(anchors),ct.shape)
        first_start=np.clip(np.asarray(center)-size//2,0,np.asarray(ct.shape)-size)
        if not mask[center]: raise ValueError("Unexpected foreground crop anchor")
        cropped=[]
        for start in [first_start,row["second_start"]]:
            sl=tuple(slice(int(s),int(s)+size) for s in start)
            cropped.append(all_hints[t][sl])
        hint=torch.from_numpy(np.stack(cropped)[:,None]).cuda()
        maps.append(hint)
    return maps


def main():
    torch.manual_seed(891)
    torch.set_num_threads(2)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cuda.matmul.allow_tf32=False
    os.umask(0o077)
    base=Path("phase1/full_report_trainonly")
    out=base/"auxiliary_AI"
    out.mkdir(exist_ok=True)
    source_run=read(base/"three_arms/summary.json")
    vocab=read(base/"vocabulary.private.json")
    codec=FullReportCodec(vocab["words"],vocab["chunk_size"])
    cases,_,common,lock,groups,_=load_locked(Path("phase1/baselines/partial_v1"),Path("phase1/baselines/sparse_hu_v1"))
    tnames=[r["volume"] for r in lock["roles"]["T"]]
    unames=[r["volume"] for r in lock["roles"]["U"]]
    texts=read_reports(set(tnames+unames))
    docs={n:codec.encode(row["Findings_EN"],row["Impressions_EN"]) for n,row in texts.items()}
    model0=FullReportUNet(codec.vocab_size).cuda()
    if fingerprint(model0.state_dict())!=source_run["design"]["initial_weights_sha256"]:
        raise ValueError("Inconsistent initialization")
    # Predefine 4 blocks of 32 updates. All arms receive exactly the same batches.
    # Use distinct draws in later blocks, do not repeat a fixed crop for 128 steps.
    blocks=[]
    for k in range(4):
        seed=891+k
        sampled=schedule(cases,32,seed)
        local={**common,"seed":seed}
        batches,ledger,_=prepare(cases,sampled,local,lock,groups)
        blocks.append((sampled,ledger,batches))
    hints=cues(cases,tnames)
    packed=[weak_maps(cases,hints,steps,ledger,64) for steps,ledger,_ in blocks]
    modes=("no_aux","with_aux")
    design={"steps":128,"checkpoints_at":[32,128],"seed":891,"U_teacher_source":"paired full report in both arms",
        "T":6,"U":8,"vocab_size":codec.vocab_size,"vocab_sha256":source_run["design"]["vocab_sha256"],
        "cohort_sha256":digest(lock),"actual_tensor_ledger_sha256":[digest(l) for _,l,_ in blocks],
        "AI_label_source":"Independent Basel five-fold CT-only segmentation, 2-mm aligned; used only in the six development T patients.",
        "common_terms":"Expert positive BCE and sparse HU-derived weak negative BCE in matched T crops; MSE to EMA paired-report teacher targets in U. R1, R2, E and external data not used.",
        "auxiliary_term":"Only augmented arm adds 0.5 times BCE against surrogate voxels. Omit expert positives and uncertain band. AI positive interior >2mm from binary boundary; negatives >8mm outside union of expert and AI positives; class means equally weighted when both present. Both arms compute losses on identical voxels.",
        "interpretation":"Secondary treatment motivated by oversegmentation in prior exploratory runs. Any evaluation revisiting those seven T cases is developmental reuse.",
        "script_sha256":sha256(Path(__file__)),"initial_weights_sha256":fingerprint(model0.state_dict())}
    write_json(out/"design.json",design)
    scores={}
    for mode in modes:
        student=copy.deepcopy(model0).train()
        teacher=copy.deepcopy(student).eval().requires_grad_(False)
        opt=torch.optim.Adam(student.parameters(),lr=1e-4)
        result={}
        tick=time.perf_counter()
        for block,((steps,ledger,batches),maps) in enumerate(zip(blocks,packed,strict=True)):
            for i,((x,fg,neg,strong,weak,has_neg),step,h) in enumerate(zip(batches,steps,maps,strict=True)):
                t=docs[tnames[step["T"]]]
                u=docs[unames[step["U"]]]
                tokens=None if step["drop_T_text"] else collate([t,t],"cuda")
                opt.zero_grad(set_to_none=True)
                logits=student(x,tokens)
                human=F.softplus(-logits[:1][fg]).mean()
                known=(human+F.softplus(logits[1:][neg]).mean())*.5 if has_neg else human
                aux=[]
                if (h==1).any(): aux.append(F.softplus(-logits[h==1]).mean())
                if (h==2).any(): aux.append(F.softplus(logits[h==2]).mean())
                weak_loss=torch.stack(aux).mean() if aux else logits.sum()*0
                with torch.no_grad(): target=teacher(weak,collate([u],"cuda")).sigmoid()
                loss=known+F.mse_loss(student(strong).sigmoid(),target)+(0.5 if mode=="with_aux" else 0)*weak_loss
                loss.backward()
                g=torch.nn.utils.clip_grad_norm_(student.parameters(),float("inf"),error_if_nonfinite=True).item()
                if not g>0 or not torch.isfinite(loss): raise ValueError("Invalid auxiliary training update")
                opt.step()
                with torch.no_grad():
                    for a,s in zip(teacher.parameters(),student.parameters(),strict=True):a.lerp_(s,.01)
                number=block*32+i+1
                if number in design["checkpoints_at"]:
                    path=out/f"{mode}_{number}.pt"
                    tmp=path.with_suffix(".tmp")
                    torch.save({"student":student.state_dict(),"teacher":teacher.state_dict(),"optimizer":opt.state_dict(),
                                "vocab_size":codec.vocab_size,"mode":mode,"step":number,"design":design},tmp)
                    tmp.replace(path)
                    loaded=torch.load(path,weights_only=True)
                    pattern=fingerprint(student.state_dict())
                    if fingerprint(loaded["student"])!=pattern: raise ValueError("Restored weights differ")
                    result[str(number)]={"sha256":sha256(path),"student_sha256":pattern,"loss":loss.item(),"gradient_l2":g}
                    print(mode,number,flush=True)
            torch.cuda.synchronize()
        result["training_seconds"]=time.perf_counter()-tick
        scores[mode]=result
    write_json(out/"summary.json",{"design":design,"arms":scores})


if __name__=="__main__":
    main()
