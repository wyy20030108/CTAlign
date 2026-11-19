"""Reconstruct the archived S002 two-patch overfit test, probing failure of the small target."""
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.ndimage import distance_transform_edt
from scipy.special import expit
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.nn import functional as F

from finding_native import QueryNet, configure, get, image, pick, read, sha, write
from replay_controls import fingerprint

OUT = Path("phase5_diag/replay")
OLD = Path("phase4_finding/native/overfit_control.json")
CHECKPOINTS = {0,32,128,256}
VARIANTS = {"joint_probe":((0,1),1.),"small_alone":((0,),1.),"joint_epsilon":((0,1),1e-6)}

def diagnostics(z, m, start):
    flat=z.ravel()
    mask=m.ravel()
    # Exact attainable Dice over all thresholds on *this training crop*. Include tied logits as one group.
    order=np.argsort(-flat,kind="stable");sorted_z=flat[order]
    ends=np.flatnonzero(np.r_[sorted_z[:-1]!=sorted_z[1:],True])
    tp=mask[order].cumsum(dtype=np.int64)[ends]
    dice=2*tp/(ends+1+mask.sum());best=int(np.argmax(dice))
    pred=(z>=0);hit=int((pred&m).sum());n=int(pred.sum())
    peak=np.array(np.unravel_index(np.argmax(z),z.shape))
    near=float(distance_transform_edt(~m,sampling=2.)[tuple(peak)])
    pos=expit(z.astype(np.float64))
    stats=lambda vals:np.quantile(vals,[0,.5,.9,.99,1]).tolist()
    return dict(true=int(m.sum()),pred_at_half=n,intersection_at_half=hit,
        dice_at_half=2*hit/(n+int(m.sum())),peak_global_ijk=(peak+start).tolist(),
        peak_distance_to_true_mm=near,prob_inside_min_median_q90_q99_max=stats(pos[m]),
        prob_outside_min_median_q90_q99_max=stats(pos[~m]),
        oracle_best_probability=float(expit(float(sorted_z[ends[best]]))),
        oracle_best_logit=float(sorted_z[ends[best]]),oracle_best_dice=float(dice[best]),
        oracle_pred_positive=int(ends[best]+1),inside_soft_mass=float(pos[m].sum()),
        total_soft_mass=float(pos.sum()),average_precision=float(average_precision_score(mask,flat)),
        roc_auc=float(roc_auc_score(mask,flat)),prevalence=float(mask.mean()))

def grads(net,terms):
    vecs=[]
    for l in terms:
        gg=torch.autograd.grad(l,net.parameters(),allow_unused=True,retain_graph=True)
        vecs.append(torch.cat([(g if g is not None else torch.zeros_like(a)).reshape(-1)
                              for g,a in zip(gg,net.parameters(),strict=True)]).detach())
    norms=[v.norm().item() for v in vecs]
    return dict(norm_unweighted=norms,cosine=float(F.cosine_similarity(vecs[0],vecs[1],dim=0)))

def produce(r,sls,variant):
    configure(891); net=QueryNet().cuda()
    init=fingerprint(net.state_dict());opt=torch.optim.AdamW(net.parameters(),lr=.002)
    active,eps=VARIANTS[variant]
    old=read(OLD)
    xt=[image(r,sl) for sl in sls]
    masks=[r["m"][i][sl] for i,sl in enumerate(sls)]
    mt=[torch.from_numpy(np.ascontiguousarray(m)).cuda() for m in masks]
    starts=[np.array([x.start for x in sl]) for sl in sls]
    counts=[int(m.sum()) for m in r["m"]]
    snapshots={};trace=[];captures={}
    for step in range(257):
        total=0;metrics=[];terms=[];logits=[]
        for j in (0,1):
            m=mt[j];z=net(xt[j],[r["queries"][j]])[0];p=z.sigmoid()
            soft=(2*p[m].sum()+eps)/(p.sum()+m.sum()+eps)
            bce=F.binary_cross_entropy_with_logits(z,m.float())
            if j in active:total=total+bce-torch.log(soft)
            terms.append(bce-torch.log(soft));logits.append(z)
            pred=z>=0;metrics.append(float(2*(pred&m).sum()/(pred.sum()+m.sum())))
        loss=total*.5
        if step in CHECKPOINTS:
            q=grads(net,terms)
            q.update(loss_unweighted=[l.item() for l in terms],dice=metrics)
            q["equals_archived_Dice"]=(metrics==old["scores"][str(step)])
            case_stats=[]
            for j,z in enumerate(logits):
                zcpu=z.detach().cpu().numpy()
                case_stats.append(diagnostics(zcpu,masks[j],starts[j]))
                if step==256:captures[f"logits_{j}"]=zcpu
            q["per_target"]=case_stats;snapshots[str(step)]=q
        if step<256:
            opt.zero_grad(set_to_none=True);loss.backward()
            g=torch.nn.utils.clip_grad_norm_(net.parameters(),5,error_if_nonfinite=True)
            opt.step();trace.append([loss.item(),g.item()])
    for j,sl in enumerate(sls):
        captures[f"ct_{j}"]=r["ct"][sl]
        captures[f"mask_{j}"]=masks[j]
        captures[f"start_{j}"]=starts[j]
    store=OUT/(variant+".npz");np.savez_compressed(store,**captures)
    for j,m in enumerate(masks):
        with np.load(store) as saved:
            assert diagnostics(saved[f"logits_{j}"],saved[f"mask_{j}"],saved[f"start_{j}"])==snapshots["256"]["per_target"][j]
    result=dict(variant=variant,source=sha(Path(__file__)),original_source=sha(Path("finding_native_check.py")),
        original_receipt=sha(OLD),case=r["code"],init=init,final=fingerprint(net.state_dict()),steps=256,
        supervised_targets=list(active),smoothing=eps,crop_start_ijk=[s.tolist() for s in starts],
        original_foreground_counts=counts,crop_foreground_counts=[int(m.sum()) for m in masks],
        crop_fraction=[float(m.sum()/n) for m,n in zip(masks,counts)],output_sha256=sha(store),
        snapshots=snapshots,training_trace_loss_and_preclip_norm=trace)
    write(OUT/(variant+".json"),result)
    print(variant,snapshots["256"]["dice"],flush=True)

@torch.inference_mode()
def inspect_original(r,sls):
    starts=[np.array([x.start for x in sl]) for sl in sls]
    results=[]
    for n in range(1,6):
        folder=OUT if n==1 else OUT/f"unmodified_{n}"
        note=read(folder/"original_callable.json");net=QueryNet().cuda()
        net.load_state_dict(torch.load(folder/"original_callable.pt",map_location="cpu",weights_only=True))
        groups=[];arrays={}
        for j,sl in enumerate(sls):
            z=net(image(r,sl),[r["queries"][j]])[0].cpu().numpy();m=r["m"][j][sl]
            stats=diagnostics(z,m,starts[j])
            assert abs(stats["dice_at_half"]-note["scores"]["256"][j])<1e-6
            groups.append(stats)
            arrays[f"logits_{j}"]=z;arrays[f"mask_{j}"]=m
            arrays[f"ct_{j}"]=r["ct"][sl];arrays[f"start_{j}"]=starts[j]
        map_file=OUT/f"unmodified_maps_{n}.npz";np.savez_compressed(map_file,**arrays)
        with np.load(map_file) as v:
            for j in (0,1):assert diagnostics(v[f"logits_{j}"],v[f"mask_{j}"],v[f"start_{j}"])==groups[j]
        results.append(dict(repeat=n,curve=note["scores"],weights_sha256=sha(folder/"original_callable.pt"),
                          weights_fingerprint=fingerprint(net.state_dict()),maps_sha256=sha(map_file),final=groups))
    return results

def figure(reps):
    fig,axes=plt.subplots(2,4,figsize=(13.4,7.1))
    worst=min(reps,key=lambda rep:rep["final"][0]["dice_at_half"])["repeat"]
    keys=[f"unmodified_maps_{worst}","small_alone","joint_epsilon"]
    with np.load(OUT/(keys[0]+".npz")) as reference:
        for j in (0,1):
            ct=reference[f"ct_{j}"];m=reference[f"mask_{j}"]
            k=int(np.argmax(m.sum((0,1))))
            axes[j,0].imshow(ct[:,:,k].T,origin="lower",cmap="gray",vmin=-1000,vmax=200)
            axes[j,0].contour(m[:,:,k].T,levels=[.5],colors="lime",linewidths=1)
            axes[j,0].set_title(f"Target {j+1}: CT + reference")
            for i,key in enumerate(keys,1):
                with np.load(OUT/(key+".npz")) as v:p=expit(v[f"logits_{j}"][:,:,k].T.astype(np.float64))
                ax=axes[j,i]
                ax.imshow(ct[:,:,k].T,cmap="gray",origin="lower",vmin=-1000,vmax=200)
                heat=ax.imshow(p,cmap="magma",origin="lower",vmin=0,vmax=1,alpha=.72)
                ax.contour(m[:,:,k].T,levels=[.5],colors="lime",linewidths=1)
                if p.min()<.5<p.max():ax.contour(p,levels=[.5],colors="cyan",linewidths=.8)
                ax.set_title(key.replace("_"," "))
            for ax in axes[j]:ax.axis("off")
    fig.suptitle("S002 training crops at step 256 | green: annotated outline; cyan: p=0.5")
    fig.subplots_adjust(right=.91,top=.9,bottom=.035,wspace=.06,hspace=.18)
    bar=fig.add_axes([.925,.19,.013,.62]);fig.colorbar(heat,cax=bar,label="Predicted probability")
    fig.savefig(OUT/"S002_replay.png",dpi=140);plt.close(fig)

def main():
    os.umask(0o077);OUT.mkdir(exist_ok=True,parents=True);configure(891)
    old=read(OLD);assert old["script"]==sha(Path("finding_native_check.py"))
    r=get("fallback_train")[0];assert r["code"]==old["code"]=="S002"
    rng=np.random.default_rng(891);sls=[pick(r,rng,j) for j in range(2)]
    reps=inspect_original(r,sls)
    write(OUT/"independent_replays.json",dict(archive=old,repeat_results=reps,old_terminal_weights_available=False,
          observational_note="The original memorise function was reexecuted unchanged; intercepting its final file write captures the terminal state after training. All repeat results retained."))
    for v in VARIANTS:produce(r,sls,v)
    notes=[read(OUT/(n+".json")) for n in VARIANTS]
    assert len({d["init"] for d in notes})==1
    assert all(x["crop_start_ijk"]==notes[0]["crop_start_ijk"] for x in notes)
    figure(reps)
    device=torch.cuda.get_device_properties(0)
    deterministic_before=torch.are_deterministic_algorithms_enabled()
    # Isolated operator smoke check, after all model diagnostics. Does not attribute the empirical variance.
    try:
        torch.use_deterministic_algorithms(True)
        q=torch.ones((1,1,4,4,4),device="cuda",requires_grad=True)
        F.interpolate(q,size=(8,8,8),mode="trilinear",align_corners=False).sum().backward()
        interpolation_check="completed successfully"
    except RuntimeError as exc:
        interpolation_check=str(exc)[:800]
    finally:
        torch.use_deterministic_algorithms(deterministic_before)
    write(OUT/"receipt.json",dict(source=sha(Path(__file__)),case=r["code"],
           variants={v:sha(OUT/(v+".json")) for v in VARIANTS},png=sha(OUT/"S002_replay.png"),
           replays_sha=sha(OUT/"independent_replays.json"),original_callable_sha=sha(Path("diag_original.py")),
           original_fn_hash_matches_old=old["script"]==sha(Path("finding_native_check.py")),
           environment=dict(torch=str(torch.__version__),cuda=torch.version.cuda,cudnn=torch.backends.cudnn.version(),
               device=device.name,visible_device=os.environ.get("CUDA_VISIBLE_DEVICES"),
               cublas_workspace_config=os.environ.get("CUBLAS_WORKSPACE_CONFIG"),original_algorithms_flag=deterministic_before,
               cudnn_deterministic=torch.backends.cudnn.deterministic,cudnn_benchmark=torch.backends.cudnn.benchmark,
               isolated_strict_trilinear_backward=interpolation_check),
           scope="Five exact-protocol reruns; identical seed did not guarantee identical numerics. Matched training-crop diagnostics; oracle thresholds use the crop itself."))
if __name__=="__main__":main()
