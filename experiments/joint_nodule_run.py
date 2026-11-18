"""Materialize the nodule roles and reuse the matched factorial experiment."""
from argparse import ArgumentParser
from itertools import combinations
from pathlib import Path
import os

import numpy as np
from scipy import ndimage as ndi

from audit_pairs import identity
from atelectasis_locate_refine import lung_field
from diag_localize import seal
from frozen_priors import read, sha
from joint_nodule_cohort import CASE, DEST
import joint_nodule_gate as shape
from joint_nodule_size import SIZE_CALIBRATION, get_calibrator, seal_size
import joint_factorial as experiment
import joint_roi as proposals
import thesis_ssl as warmup
from thesis_cohort import choose_stratified
from thesis_lobe_model import LobeStore
from thesis_lobes import prepare
from thesis_model import GRID


COHORT = DEST / "cohort.private.json"
WARM = DEST / "warm"
TRAIN = DEST / "factorial_matched"
ROIS = DEST / "candidate_rois"
AUDIT = DEST / "candidate_audit"


def usable_lungs(path):
    with np.load(path) as f:
        ct=f["ct"]
    try:
        lung_field(ct)
    except ValueError:
        return False
    return True


def build():
    data = read(CASE)
    cohort = {}
    excluded = {}
    for label, rows in data.items():
        role = {"train":"train", "dev":"dev", "reserved":"test"}[label]
        cohort[role] = []
        excluded[role] = []
        for raw in rows:
            code = raw["code"]
            receipt = DEST / "receipts" / (code+".json")
            if not receipt.is_file():
                assert (receipt.with_suffix(".pending.json")).is_file()
                excluded[role].append((code,"publisher transfer incomplete"))
                continue
            r = read(receipt)
            if r["status"]!="eligible" or max(r.get("bounding_mm",[100]))>36:
                excluded[role].append((code,r.get("reason","maximum physical extent")))
                continue
            p = DEST / "arrays" / (code+".npz")
            assert sha(p)==r["sha"]
            if not usable_lungs(p):
                excluded[role].append((code,"lung field unobtainable"))
                continue
            if r["components"]>1:
                with np.load(p) as f:
                    labels,_ = ndi.label(f["mask"],np.ones((3,3,3),bool))
                sizes = np.bincount(labels.ravel())[1:]
                if sizes.max()/sizes.sum()<.80:
                    excluded[role].append((code,"dispersed reference"))
                    continue
            cohort[role].append(dict(raw, path=str(p), sha=r["sha"],coords=r["voxels"],volume_ml=r["voxels"]*GRID**3/1000))
    assert len(cohort["train"])>=7 and len(cohort["dev"])>=8 and len(cohort["test"])>=12
    chosen = choose_stratified(cohort["train"],7,"nodule-7-visible-v2")
    u = DEST / "report_only"
    no_masks = []
    for row in read(u/"source.private.json"):
        f = u/"receipts"/(row["code"]+".json")
        if not f.is_file():
            continue
        r = read(f)
        p = u/"arrays"/(row["code"]+".npz")
        assert sha(p)==r["processed_sha"] and not r["contains_mask"]
        with np.load(p) as f:
            assert set(f.files)=={"ct","affine"}
        if not usable_lungs(p):
            excluded.setdefault("unlabelled",[]).append((row["code"],"lung field unobtainable"))
            continue
        no_masks.append(dict(row,patient=":".join(identity(row["name"])[1]),path=str(p),sha=r["processed_sha"]))
    cohort["unlabelled"] = choose_stratified(no_masks,36,"nodule-u-ready-v2")
    for p,q in combinations(cohort.values(),2):
        assert {r["patient"] for r in p}.isdisjoint({r["patient"] for r in q})
    info = {k:len(v) for k,v in cohort.items()}
    cohort["labels"] = {"20":[r["code"] for r in chosen]}
    seal(COHORT,cohort)
    seal(DEST/"lock.json",dict(cohort=sha(COHORT),patients=info,visible=7,resolution=GRID,
                              source=sha(CASE),excluded=excluded,thresholds="dev only",code=sha(Path(__file__))))
    print(info,flush=True)


def phases(stage,kind,seed,arm,shard,shards):
    assert read(DEST/"lock.json")["cohort"]==sha(COHORT)
    data = read(COHORT)
    warmup.OUT=WARM
    proposals.MANIFEST, proposals.PREVIOUS, proposals.ROI, proposals.AUDIT = COHORT,WARM,ROIS,AUDIT
    experiment.MANIFEST, experiment.PREVIOUS, experiment.ROI, experiment.OUT = COHORT,WARM,ROIS,TRAIN
    if stage=="anatomy":
        rows = [r for r in data["train"] if r["code"] in data["labels"]["20"]]+data["dev"]+data["unlabelled"]
        prepare(rows[shard::shards])
        shape.prepare(rows[shard::shards])
    elif stage=="warm-lock":
        WARM.mkdir(exist_ok=True)
        seal(WARM/"protocol.json",dict(cohort=sha(COHORT),seeds=warmup.SEEDS,steps=warmup.WARM,source=sha(Path(warmup.__file__))))
        print("Size prior",seal_size(COHORT),flush=True)
    elif stage=="warm":
        assert read(WARM/"protocol.json")["cohort"]==sha(COHORT)
        rows = [r for r in data["train"] if r["code"] in data["labels"]["20"]]
        warmup.warm("20",seed,kind,data,rows,LobeStore(limit=10))
    elif stage in ("dev", "unlabelled"):
        shape.activate()
        proposals.run(stage,kind,shard,shards)
    elif stage=="lock":
        experiment.lock("agreement","agreement","conditioned")
        metadata={r["code"]:sha(shape.LOCATIONS/(r["code"]+".json")) for k in ("dev","unlabelled") for r in data[k]}
        seal(TRAIN/"shape_protocol.json",dict(geometry=sha(Path(shape.__file__)),cases=metadata,
             size_calibration=sha(SIZE_CALIBRATION),
             scope="Same CT geometry recipe in all arms. Report adds laterality and soft lobe weighting. Proposal-conditioned transport rescales the class prior to the local support, then blends it with the calibrated report size when exactly one compatible measurement is present."))
    elif stage=="train":
        current=read(TRAIN/"shape_protocol.json")
        assert current["geometry"]==sha(Path(shape.__file__)) and current["size_calibration"]==sha(SIZE_CALIBRATION)
        shape.report_volume=get_calibrator(COHORT)
        shape.activate(experiment)
        experiment.fit(kind,arm,seed)


if __name__=="__main__":
    os.umask(0o077)
    p=ArgumentParser()
    p.add_argument("stage",choices=("build","anatomy","warm-lock","warm","dev","unlabelled","lock","train"))
    p.add_argument("--kind",choices=("A","B"),default="B")
    p.add_argument("--seed",type=int,choices=warmup.SEEDS,default=warmup.SEEDS[0])
    p.add_argument("--arm",choices=experiment.OPTIONS,default="N")
    p.add_argument("--shard",type=int,default=0)
    p.add_argument("--shards",type=int,default=1)
    a=p.parse_args()
    if a.stage=="build":build()
    else:phases(a.stage,a.kind,a.seed,a.arm,a.shard,a.shards)
