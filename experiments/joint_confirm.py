"""Lock and verify a fresh test only after the development gate is met."""
from argparse import ArgumentParser
from collections import defaultdict
from pathlib import Path
import os
import time

import numpy as np
import torch

import thesis_model as image
from atelectasis_screen import anatomy
from audit_pairs import identity
from diag_localize import seal
from finding_train import configure
from frozen_priors import read, sha, write
from joint_metrics import ARM, compare, summary as paired_interval
from joint_reserve import CANDIDATES, RESERVE, exposed_patients
from joint_roi import ROOT
from thesis_component_fast import segment
from thesis_lobe_model import LobeStore
from thesis_lobes import prepare
from thesis_ssl import SEEDS


DEFAULT = ROOT / "factorial_matched"
NODULE = ROOT / "nodule_v2" / "factorial_matched"
NODULE_GROWTH = ROOT / "nodule_v2" / "growth_matched"
NODULE_EXTERNAL = ROOT / "nodule_v2" / "external_matched"
NODULE_RESIDUAL = ROOT / "nodule_v2" / "external_residual"
EXTERNAL_STUDIES=(NODULE_EXTERNAL,NODULE_RESIDUAL)
NODULE_STUDIES = (NODULE,NODULE_GROWTH,*EXTERNAL_STUDIES)
PROTO = ROOT/"nodule_v2"/"external_proto_only"


def lock_prototype_control(root):
    if root!=NODULE_RESIDUAL:
        return None
    pid_file=root.parent/"proto_only_pid.json"
    if not pid_file.is_file():
        return None
    pid=read(pid_file)["pid"]
    process=Path(f"/proc/{pid}/cmdline")
    path=[PROTO/"B"/str(s)/"R"/"development.json" for s in SEEDS]
    while not all(f.is_file() for f in path):
        try: running=b"run_proto_only.py" in process.read_bytes()
        except FileNotFoundError: running=False
        if not running:
            raise RuntimeError("The prototype-only control ended before all three development receipts")
        time.sleep(10)
    docs=[read(f) for f in path]
    for p,v in zip(path,docs,strict=True):
        assert v["model"]==sha(p.with_name("model.pt")) and v["protocol"]==sha(PROTO/"protocol.json")
    grid={str(t):float(np.mean([v["value"][str(t)]["dice"] for d in docs for v in d["score"]])) for t in image.THRESHOLDS}
    best=max(grid,key=lambda t:(grid[t],-abs(float(t)-.50)))
    return dict(threshold=float(best),dice=grid[best],weights=[dict(seed=s,weight=d["model"],receipt=sha(f),cap_ml=d["cap_ml"])
                  for s,d,f in zip(SEEDS,docs,path,strict=True)],protocol=sha(PROTO/"ablation.json"))


def reference(root):
    if root in NODULE_STUDIES:
        p=root.parent/"cohort.private.json"
        return read(p)["test"]
    avoid=exposed_patients()
    result=[]
    for row in read(CANDIDATES):
        if row["source"]!="official_validation" or row["kind"]!="exact_linear":
            continue
        receipt=RESERVE/"receipts"/(row["code"]+".json")
        raw=read(receipt)
        if raw["status"]!="eligible" or raw["components"]!=1 or min(raw["bounding_mm"])>=40:
            continue
        assert identity(row["name"])[1] not in avoid
        disk=RESERVE/"arrays"/(row["code"]+".npz")
        assert sha(disk)==raw["sha"]
        result.append(dict(row,path=str(disk),sha=raw["sha"]))
    return result


def freeze(root):
    source=root/"development_study.json"
    diagnostic=read(source)
    assert set(diagnostic["thresholds"])==set(ARM)
    effect={k:v["dice"]["delta"] for k,v in diagnostic["contrast"].items()}
    # Guard the sealed patients until both factors show a benefit and the combination wins on development.
    common=("report_only_effect",)
    families={"UOT":common+("transport_only_effect","add_transport_to_report","add_report_to_transport","joint_transport_over_supervised"),
              "full":common+("full_framework_without_report","full_framework_with_report","full_report_beyond_framework","joint_full_over_supervised")}
    dev_gate={k:all(effect[name]>0 for name in values) for k,values in families.items()}
    cases=reference(root)
    assert len(cases)>=4
    write(root/"development_gate.json",dict(family=dev_gate,patients=len(cases),effect=effect,source=sha(source)))
    if not any(dev_gate.values()):
        print("Both families remain in development",dev_gate,flush=True)
        return
    if root in NODULE_STUDIES:
        selection=root.parent/"validation_selection.json"
        if not selection.is_file() or read(selection)["chosen"]!=root.name:
            print("Development comparison will choose one nodule model family before the reserved cases are opened.",flush=True)
            return
        chosen=read(selection)
        assert chosen["minimum"]>=.03
        recipe=next(v for v in chosen["development"] if v["root"]==root.name)
        assert recipe["development"]==sha(source) and recipe["groups"][chosen["family"]]["combination_floor"]>=.03
    geometry=(NODULE if root in NODULE_STUDIES else root)/"shape_protocol.json"
    additional=root/"growth_protocol.json"
    external_detector=None
    external_recipe=None
    external_readout=None
    if root in EXTERNAL_STUDIES:
        import joint_external_nodule as detector
        import joint_external_refine as crop
        recipe=NODULE_EXTERNAL/"warm"/"protocol.json"
        assert read(recipe)["code"]==sha(Path(crop.__file__)) and read(recipe)["model"]==detector.DIGEST
        external_detector=detector.DIGEST
        external_recipe=sha(recipe)
        if root==NODULE_RESIDUAL:
            import joint_external_residual as edge
            assert read(root/"protocol.json")["source"]==sha(Path(edge.__file__))
            external_readout=sha(Path(edge.__file__))
    seal(root/"external_lock.json",dict(development=sha(source),protocol=sha(root/"protocol.json"),families=dev_gate,
                                       geometry=sha(geometry) if geometry.is_file() else None,
                                       growth=sha(additional) if additional.is_file() else None,
                                       external_detector=external_detector,external_recipe=external_recipe,
                                       external_readout=external_readout,
                                       prototype_control=lock_prototype_control(root),
                                       reference={row["code"]:row["sha"] for row in cases},
                                       weights=diagnostic["weights"],thresholds=diagnostic["thresholds"],
                                       intended="Paired patient effects. Report-only, PRCV-only, both versus each, both versus supervision. +3pp vs stronger single condition is the practical target; confidence intervals and external scope shown explicitly."))
    print("Frozen new cases",len(cases),dev_gate,flush=True)


def evaluation(root,group,seed):
    lock=root/"external_lock.json"
    fixed=read(lock)
    cases=reference(root)
    assert fixed["reference"]=={row["code"]:row["sha"] for row in cases}
    if root in NODULE_STUDIES:
        import joint_nodule_gate as shape
        geometry=NODULE/"shape_protocol.json"
        assert fixed["geometry"]==sha(geometry) and read(geometry)["geometry"]==sha(Path(shape.__file__))
        if root==NODULE_GROWTH:
            assert fixed["growth"]==sha(root/"growth_protocol.json")
        if group=="anatomy":
            shape.prepare(cases)
            if root in EXTERNAL_STUDIES:
                import joint_external_nodule as detector
                detector.run(0,1,0,"test")
        elif root in EXTERNAL_STUDIES:
            import joint_external_refine as crop
            recipe=NODULE_EXTERNAL/"warm"/"protocol.json"
            assert fixed["external_recipe"]==sha(recipe) and read(recipe)["code"]==sha(Path(crop.__file__))
            if root==NODULE_RESIDUAL:
                import joint_external_residual as edge
                assert fixed["external_readout"]==sha(Path(edge.__file__))
                edge.install()
            else:
                crop.install()
        else:
            shape.activate()
    if group=="anatomy":
        prepare(cases)
        if fixed.get("prototype_control") is not None:
            for one in SEEDS:
                evaluation(root,"BR",one)
        return
    modality,arm=group
    extra=fixed.get("prototype_control") if group=="BR" else None
    if extra is not None:
        folder=PROTO/modality/str(seed)/arm
        assert extra["protocol"]==sha(PROTO/"ablation.json")
        recorded=next(v for v in extra["weights"] if v["seed"]==seed)
        threshold=extra["threshold"]
    else:
        folder=root/modality/str(seed)/arm
        recorded=next(v for v in fixed["weights"][group] if v["seed"]==seed)
        threshold=fixed["thresholds"][group]["threshold"]
    model=folder/"model.pt"
    assert sha(model)==recorded["weight"] and sha(folder/"development.json")==recorded["receipt"]
    out=root/"heldout"/group/str(seed)
    out.mkdir(parents=True,exist_ok=True)
    configure(seed)
    network=image.FindingNet().to(image.DEVICE).eval()
    network.load_state_dict(torch.load(model,weights_only=True,map_location=image.DEVICE))
    store=LobeStore(limit=2)
    rows=[]
    for row in cases:
        code=row["code"]
        path=out/(code+".npz")
        card=out/(code+".json")
        if card.is_file():
            existing=read(card)
            assert existing["lock"]==sha(lock) and existing["prediction"]==sha(path)
            rows.append(existing)
            continue
        case=store.get(row,True)
        field,_=image.volume_logits(network,case,row["query"] if modality=="B" else "")
        y=segment(field,threshold,recorded["cap_ml"])
        np.savez_compressed(path,shape=case.ct.shape,prediction=np.packbits(y.ravel()))
        result=dict(code=code,metrics=image.score(y,case.truth,surface=True),lock=sha(lock),prediction=sha(path))
        write(card,result)
        rows.append(result)
        print(group,seed,code,result["metrics"]["dice"],flush=True)
    write(out/"complete.json",dict(rows=rows,seed=seed,lock=sha(lock)))


def release(root):
    lock=root/"external_lock.json"
    fixed=read(lock)
    scores={}
    references={}
    for row in reference(root):
        assert sha(Path(row["path"]))==fixed["reference"][row["code"]]
        with np.load(row["path"]) as f:references[row["code"]]=f["mask"].astype(bool)
    controls=(*ARM,"BR") if fixed.get("prototype_control") is not None else ARM
    for arm in controls:
        data=defaultdict(list)
        for seed in SEEDS:
            path=root/"heldout"/arm/str(seed)
            completion=read(path/"complete.json")
            assert completion["lock"]==sha(lock)
            assert {r["code"] for r in completion["rows"]}==set(references)
            for row in completion["rows"]:
                c=row["code"]
                p=path/(c+".npz")
                assert sha(p)==row["prediction"]
                truth=references[c]
                with np.load(p) as f:
                    prediction=np.unpackbits(f["prediction"],count=truth.size).reshape(truth.shape).astype(bool)
                assert image.score(prediction,truth,surface=True)==row["metrics"]
                data[c].append(row["metrics"])
        scores[arm]={c:{m:float(np.mean([r[m] for r in scans])) for m in ("dice","recall","fp_ml","miss")}
                     for c,scans in data.items()}
    comparisons=compare(scores)
    if "BR" in scores:
        comparisons["transport_given_prototypes"]={m:paired_interval([scores["BP"][c][m]-scores["BR"][c][m] for c in sorted(references)])
                                                    for m in ("dice","recall","fp_ml","miss")}
    summary={arm:{m:float(np.mean([r[m] for r in by_patient.values()])) for m in ("dice","recall","fp_ml","miss")}
             for arm,by_patient in scores.items()}
    def superiority(name,minimum=0.):
        score=comparisons[name]["dice"]
        return score["delta"]>=minimum and score["ci95"][0]>0
    checks={}
    for name,isolated,over_text,over_method,over_supervision,interaction in (
        ("UOT","transport_only_effect","add_transport_to_report","add_report_to_transport",
         "joint_transport_over_supervised","transport_report_interaction"),
        ("full","full_framework_without_report","full_framework_with_report","full_report_beyond_framework",
         "joint_full_over_supervised","full_framework_report_interaction"),
    ):
        checks[name]=dict(report=superiority("report_only_effect"),method=superiority(isolated),
                          joint_beyond_report=superiority(over_text,.03),
                          joint_beyond_method=superiority(over_method,.03),
                          joint_beyond_supervision=superiority(over_supervision),interaction=superiority(interaction),
                          reached_dev_gate=fixed["families"][name])
    write(root/"confirmed.json",dict(summary=summary,comparisons=comparisons,patients=scores,
          criteria=checks,lock=sha(lock)))
    print({k:round(v["dice"]*100,3) for k,v in summary.items()},flush=True)


if __name__=="__main__":
    os.umask(0o077)
    p=ArgumentParser()
    p.add_argument("stage",choices=("lock","eval","release"))
    p.add_argument("--root",type=Path,default=DEFAULT,choices=(DEFAULT,*NODULE_STUDIES))
    p.add_argument("--arm",choices=(*ARM,"anatomy"),default="anatomy")
    p.add_argument("--seed",type=int,choices=SEEDS,default=SEEDS[0])
    a=p.parse_args()
    if a.stage=="lock":freeze(a.root)
    elif a.stage=="eval":evaluation(a.root,a.arm,a.seed)
    else:release(a.root)
