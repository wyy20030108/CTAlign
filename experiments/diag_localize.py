"""Factor full-CT target retrieval from segmentation, with shared, image-only candidate windows."""
import argparse
import hashlib
import itertools
import json
import os
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from audit_pairs import identity
from finding_train import SRC, OUT, SEEDS, QueryNet, configure
from finding_native import image, slices, windows
from frozen_priors import read, sha, write

BASE = Path("phase5_diag/localization")
GROUPS = ("fallback_val", "swap_dev")
WIDTH = 32
STRIDE = 16
KS = (1, 2, 5)
DESIGN = {
    "targets": "36 specified findings in 18 reused development patients (10 + 8); study unit patient",
    "proposal_model": "phase4 three image-only whole-CT 80-cube checkpoints; average upsampled sigmoid scores; all receive an empty query",
    "proposal_windows": "Fixed 32^3 voxels (64 mm edge), stride 16 with final end-aligned window. Rank by cube mean, break ties by lexicographic starts, greedily retain disjoint cubes.",
    "passes": "Seal all 18 image-only candidate lists before segmenting; freeze all native-grid predictions before reading voxel masks for scoring.",
    "candidate_metrics": "Primary K=2, sensitivities K=1 and K=5. A hit means >=50% target voxels inside the union. Report any overlap, >=90%, mean covered mass and physical search fraction.",
    "mask_input": "ROI generation uses CT only, identical for both finding queries and all segmentation models. The shared full-FOV sliding-window segmentation takes CT and query, never the candidate centre, reference or layer number as location metadata.",
    "segmenters": "Frozen historical native supervised/semi use their patient-training-only threshold. If the conditional training gate passes, also run its first passing QueryNet alone at .5. Distinct training cohorts; between-model results are descriptive.",
    "endpoints": "Whole-field Dice, Dice of candidate-restricted prediction against whole target, and region-only Dice conditional on >=50% retrieved. Conditional score applies on the hit subset; also report the number of patients and findings.",
    "oracle": "After freezing predictions, select one candidate-grid box separately maximising coverage of each true target. Union the two and use the SAME region for both queries. This is a positive-location diagnostic, not automatic retrieval or a theoretical Dice bound.",
    "thresholds": "Mean overlapped tile logits. Keep prior clinical text and rephrased ReX findings separate. Do not tune K, box size or score thresholds against these patients."
}


def seal(path, value):
    if path.exists():
        if read(path) != value:
            raise ValueError("An existing lock differs: " + str(path))
    else:
        write(path, value)


def cases():
    checks = read(SRC / "receipt.json")["sha256_lists"]
    rows = []
    for group in GROUPS:
        f = SRC / (group + ".private.json")
        assert checks[f.name] == sha(f)
        for r in read(f):
            rows.append(dict(code=r["code"], group=group, path=r["path"], signature=r["sha256"]))
    training, development = set(), set()
    for role in ("fallback_train", "unlabelled_train", *GROUPS):
        f = SRC / (role + ".private.json")
        assert sha(f) == checks[f.name]
        for row in read(f):
            p = identity(row["name"])[1]
            assert p not in training | development
            (development if role in GROUPS else training).add(p)
    assert len(training) == 48 and len(development) == len({r["code"] for r in rows}) == len(rows) == 18
    return rows


def prepare():
    BASE.mkdir(parents=True, exist_ok=True)
    roster = cases()
    seal(BASE / "protocol.json", dict(design=DESIGN, code_sha256=sha(Path(__file__))))
    seal(BASE / "roster.private.json", roster)
    return roster


def starts(n):
    if n < WIDTH:
        raise ValueError("Unsupported volume smaller than a proposal")
    result = list(range(0, n-WIDTH+1, STRIDE))
    if result[-1] != n-WIDTH:
        result.append(n-WIDTH)
    return result


def boxes(shape):
    return np.asarray(list(itertools.product(*(starts(n) for n in shape))), dtype=np.int32)


def cube_sums(arr, locations):
    dtype = np.float64 if np.issubdtype(arr.dtype, np.floating) else np.int64
    integral = np.zeros(tuple(n+1 for n in arr.shape), dtype=dtype)
    buf = integral[1:, 1:, 1:]
    buf[...] = arr
    for ax in range(3):
        np.cumsum(buf, axis=ax, out=buf)
    i, j, k = locations.T
    I, J, K = (locations+WIDTH).T
    return (integral[I,J,K]-integral[i,J,K]-integral[I,j,K]-integral[I,J,k]
            +integral[i,j,K]+integral[i,J,k]+integral[I,j,k]-integral[i,j,k])


def greedy(available, scores):
    chosen = []
    for idx in np.argsort(-scores, kind="stable"):
        start = available[idx]
        if any(np.all(np.abs(start-available[c]) < WIDTH) for c in chosen):
            continue
        chosen.append(int(idx))
        if len(chosen) == max(KS):
            return chosen
    raise ValueError("Could not select five disjoint cubes")


def score_mask(shape, chosen):
    region = np.zeros(shape, dtype=bool)
    for loc in chosen:
        region[tuple(slice(int(start),int(start)+WIDTH) for start in loc)] = True
    return region


def synthetic_check():
    a = np.random.default_rng(891).random((99, 105, 78), dtype=np.float32)
    all_boxes = boxes(a.shape)
    fast = cube_sums(a, all_boxes)
    expected = [a[tuple(slice(s,s+WIDTH) for s in loc)].sum(dtype=np.float64) for loc in all_boxes]
    assert np.allclose(fast, expected, atol=1e-6, rtol=0)
    idx = greedy(all_boxes, fast)
    assert np.count_nonzero(score_mask(a.shape, all_boxes[idx])) == len(idx)*WIDTH**3


def image_weights():
    models, hashes = [], []
    for seed in SEEDS:
        f = OUT / str(seed) / "image.pt"
        h = sha(f)
        assert h == read(f.with_suffix(".json"))["sha256"]
        state = torch.load(f, map_location="cpu", weights_only=True)
        assert state["arm"] == "image" and state["seed"] == seed and state["step"] == 256
        net = QueryNet().cuda().eval()
        net.load_state_dict(state["net"])
        models.append(net)
        hashes.append(h)
    return models, hashes


@torch.inference_mode()
def propose():
    roster = prepare()
    models, model_hashes = image_weights()
    axis = torch.linspace(-1,1,80)
    coords = torch.stack(torch.meshgrid(axis,axis,axis,indexing="ij"))
    rows = []
    for r in roster:
        path = Path(r["path"])
        assert sha(path) == r["signature"]
        with np.load(path, allow_pickle=False) as z:
            ct,aff = z["ct"],z["affine"]
        assert np.allclose(aff[:3,:3],2*np.eye(3))
        shape = ct.shape
        scaled = F.interpolate(torch.from_numpy(ct.astype(np.float32))[None,None],size=(80,)*3,
                               mode="trilinear",align_corners=False)[0]
        x = torch.cat((((scaled+1000)/1200).clamp(0,1),((scaled+150)/400).clamp(0,1),coords))[None].cuda()
        total = np.zeros(shape, dtype=np.float32)
        for net in models:
            logit = net(x, [""])
            z = F.interpolate(logit[:,None],shape,mode="trilinear",align_corners=False)
            total += z.sigmoid().cpu().numpy()[0,0]
        total /= len(models)
        assert np.isfinite(total).all()
        locs = boxes(shape)
        scores = cube_sums(total,locs)/WIDTH**3
        idx = greedy(locs,scores)
        rows.append(dict(code=r["code"],shape=list(shape),starts=locs[idx].tolist(),scores=scores[idx].tolist(),
                         n_eligible=len(locs),input_sha256=r["signature"]))
    seal(BASE/"candidate_lock.json",dict(cases=rows,models=model_hashes,roster_hash=sha(BASE/"roster.private.json"),
        protocol_hash=sha(BASE/"protocol.json"),labels_decoded_during_proposal=False))
    print("sealed CT-only proposals",len(rows),flush=True)


def threshold_and_weights(arm):
    if arm in ("supervised","semi"):
        base=OUT/"native"/arm
        threshold=read(base/"native_selection.json")["p_actual"]
        path=base/"model.pt"
        assert sha(path)==read(base/"training.json")["checkpoint"]
    else:
        base=Path("phase5_diag/common_clean")/arm.removeprefix("new_")
        path=base/"model.pt"
        threshold=.5
    ck=torch.load(path,map_location="cpu",weights_only=True)
    if "net" in ck:
        ck=ck["net"]
    net=QueryNet().cuda().eval()
    net.load_state_dict(ck)
    return net,float(threshold),sha(path)


@torch.inference_mode()
def fullfield(net,ct,queries):
    record=dict(ct=ct,ax=[np.linspace(-1,1,n,dtype=np.float32) for n in ct.shape])
    sums=np.zeros((len(queries),*ct.shape),np.float32)
    counts=np.zeros(ct.shape,np.uint8)
    for loc in itertools.product(*(windows(n) for n in ct.shape)):
        sl=slices(loc)
        vals=net(image(record,sl),queries).cpu().numpy()
        sums[(slice(None),*sl)]+=vals
        counts[sl]+=1
    assert counts.min()>0 and np.isfinite(sums).all()
    sums/=counts[None]
    return sums


def segment(arm):
    roster=prepare()
    proposals=read(BASE/"candidate_lock.json")
    assert [r["code"] for r in proposals["cases"]]==[r["code"] for r in roster]
    directory=BASE/arm
    directory.mkdir(exist_ok=True)
    queries,clinical={},{}
    for group in GROUPS:
        for row in read(SRC/(group+".private.json")):
            queries[row["code"]]=row["queries"]
        clinical.update(read(Path("phase3_swap/clinical")/(group+".private.json")))
    net,threshold,ck=threshold_and_weights(arm)
    cut=float(np.log(np.float32(threshold)/(1-np.float32(threshold))))
    records=[]
    for r in roster:
        code=r["code"]
        with np.load(r["path"],allow_pickle=False) as v:
            ct=v["ct"]
        modes={"finding":queries[code]}
        if code in clinical:
            modes["clinical"]=clinical[code]
        pack=directory/(code+".npz")
        receipt=directory/(code+"_prediction.json")
        if receipt.is_file():
            old=read(receipt)
            assert old["sha256"]==sha(pack) and old["checkpoint"]==ck and old["protocol"]==sha(BASE/"protocol.json")
        else:
            data={}
            for key,qs in modes.items():
                logits=fullfield(net,ct,qs)
                data[key]=np.packbits(logits>=cut,bitorder="little")
                if key=="finding" and arm in ("supervised","semi"):
                    source=OUT/"native"/arm/(code+".npz")
                    assert read(source.with_suffix(".json"))["sha256"]==sha(source)
                    with np.load(source) as prev:
                        assert np.array_equal(prev["paired"],np.packbits(logits>=0,bitorder="little"))
            tmp=pack.with_suffix(".tmp.npz")
            np.savez_compressed(tmp,shape=np.asarray(ct.shape),**data)
            tmp.replace(pack)
            write(receipt,dict(code=code,sha256=sha(pack),checkpoint=ck,protocol=sha(BASE/"protocol.json")))
        records.append(read(receipt))
    seal(directory/"prediction_lock.json",dict(arm=arm,threshold=threshold,checkpoint_sha256=ck,candidate_hash=sha(BASE/"candidate_lock.json"),
                                                 cases=records,n_clinical=len(set(queries)&set(clinical))))
    print("sealed full-FOV predictions",arm,len(records),flush=True)


def target_metrics(pred,truth,region):
    n=int(truth.sum())
    present=truth&region
    restricted=pred&region
    n_inside=int(present.sum());n_pred=int(restricted.sum());hit=int(np.count_nonzero(pred&present))
    return dict(truth_voxels=n,truth_in_region=n_inside,predicted_in_region=n_pred,true_positives=hit,
                volume_coverage=n_inside/n,hit50=2*n_inside>=n,any=n_inside>0,hit90=10*n_inside>=9*n,
                dice_whole=2*hit/(n+n_pred),dice_inside=2*hit/(n_inside+n_pred) if n_inside else None)


def score(arm):
    roster=prepare()
    directory=BASE/arm
    lock=read(directory/"prediction_lock.json");proposals=read(BASE/"candidate_lock.json")["cases"]
    assert lock["candidate_hash"]==sha(BASE/"candidate_lock.json")
    records=[]
    for r,proposal,pred_record in zip(roster,proposals,lock["cases"],strict=True):
        code=r["code"]
        assert code==proposal["code"]==pred_record["code"]
        assert sha(Path(r["path"]))==r["signature"]
        with np.load(r["path"]) as z:masks=z["masks"].astype(bool)
        shape=masks.shape[1:];coords=np.asarray(proposal["starts"])
        regions={f"K{k}":score_mask(shape,coords[:k]) for k in KS}
        assert all(regions[f"K{k}"].sum()==k*WIDTH**3 for k in KS)
        lattice=boxes(shape)
        oracle=[lattice[int(np.argmax(cube_sums(m,lattice)))] for m in masks]
        regions.update(oracle_2=score_mask(shape,oracle),whole=np.ones(shape,dtype=bool))
        pack=directory/(code+".npz");assert sha(pack)==pred_record["sha256"]
        row=dict(code=code,group=r["group"],oracle_boxes=[l.tolist() for l in oracle],fov_voxels=int(masks[0].size),
                 region_voxels={k:int(v.sum()) for k,v in regions.items()},modes={})
        with np.load(pack) as z:
            assert tuple(z["shape"])==shape
            for mode in sorted(set(z.files)-{"shape"}):
                binary=np.unpackbits(z[mode],count=masks.size,bitorder="little").reshape(masks.shape).astype(bool)
                row["modes"][mode]={}
                for key,reg in regions.items():
                    row["modes"][mode][key]={
                        "correct":[target_metrics(p,t,reg) for p,t in zip(binary,masks)],
                        "swapped":[target_metrics(p,t,reg) for p,t in zip(binary[::-1],masks)]
                    }
        records.append(row)
    if arm in ("supervised","semi"):
        previous=read(OUT/"native"/arm/"native_calibration.json");chosen=previous["best"]
        ref={r["code"]:r for g in GROUPS for r in previous["rows"][g]}
        for rec in records:
            for condition in ("correct","swapped"):
                measured=np.mean([v["dice_whole"] for v in rec["modes"]["finding"]["whole"][condition]])
                assert abs(measured-ref[rec["code"]][condition][chosen])<1e-8
    summary={}
    for mode in ("finding","clinical"):
        available=[r for r in records if mode in r["modes"]]
        summary[mode]={}
        for reg in ("whole","K1","K2","K5","oracle_2"):
            flat=[m for r in available for m in r["modes"][mode][reg]["correct"]]
            eligible=[r for r in available if any(m["hit50"] for m in r["modes"][mode][reg]["correct"])]
            cond=[np.mean([m["dice_inside"] for m in r["modes"][mode][reg]["correct"] if m["hit50"]]) for r in eligible]
            def macro(condition):
                return float(np.mean([np.mean([m["dice_whole"] for m in r["modes"][mode][reg][condition]]) for r in available]))
            summary[mode][reg]=dict(patients=len(available),targets=len(flat),hit50_targets=sum(m["hit50"] for m in flat),
              hit50_patients=len(eligible),hit90_targets=sum(m["hit90"] for m in flat),any_hit_targets=sum(m["any"] for m in flat),
              mean_true_coverage=float(np.mean([m["volume_coverage"] for m in flat])),
              mean_fov_fraction=float(np.mean([r["region_voxels"][reg]/r["fov_voxels"] for r in available])),
              end_to_end_dice=macro("correct"),swapped_end_to_end_dice=macro("swapped"),
              conditional_within_roi_dice=float(np.mean(cond)) if cond else None)
    write(directory/"results.json",dict(arm=arm,prediction_lock_sha256=sha(directory/"prediction_lock.json"),threshold=lock["threshold"],
          summary=summary,cases=records,previous_full_fov_threshold_checked=(arm in ("supervised","semi"))))
    print("scored",arm,json.dumps(summary["finding"]["K2"]),flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("action",choices=("prepare","propose","segment","score"))
    parser.add_argument("--arm",default="supervised")
    a=parser.parse_args()
    if a.arm not in ("supervised","semi") and not a.arm.startswith("new_"):
        raise ValueError("Expected a known frozen native model or a new gated model")
    os.umask(0o077)
    synthetic_check()
    if a.action=="prepare":prepare()
    elif a.action=="propose":
        configure(891);propose()
    elif a.action=="segment":
        configure(891);segment(a.arm)
    else:score(a.arm)
