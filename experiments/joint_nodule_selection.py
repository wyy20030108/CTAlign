"""Select from development comparisons and spend the fresh nodule evaluation once."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import os
import sys
import time

from diag_localize import seal
from frozen_priors import read,sha,write
from joint_metrics import ARM
from joint_nodule_pipeline import alive,batch,command
from joint_nodule_cohort import DEST
from joint_nodule_growth_refine import ROOT as GROWTH
from joint_nodule_run import TRAIN
from thesis_ssl import SEEDS


MIN_COMBINATION=.03


def compare_development(root):
    path=root/"development_study.json"
    evidence=read(path)
    contrasts=evidence["contrast"]
    required={
        "UOT":("transport_only_effect","add_transport_to_report","add_report_to_transport","joint_transport_over_supervised"),
        "full":("full_framework_without_report","full_framework_with_report","full_report_beyond_framework","joint_full_over_supervised"),
    }
    groups={}
    for name,keys in required.items():
        common=("report_only_effect",*keys)
        effects={k:contrasts[k]["dice"]["delta"] for k in common}
        floor=min(effects[k] for k in keys[1:3])
        groups[name]=dict(eligible=min(effects.values())>0 and floor>=MIN_COMBINATION,effects=effects,combination_floor=floor)
    return dict(root=root.name,development=sha(path),groups=groups)


def select(roots):
    proposals=[compare_development(r) for r in roots]
    eligible=[(g["combination_floor"],p["root"],family) for p in proposals for family,g in p["groups"].items() if g["eligible"]]
    if not eligible:
        write(DEST/"no_confirmation_yet.json",dict(development=proposals,minimum=MIN_COMBINATION,
              reason="no development family met the positive-direction and practical combination targets"))
        write(DEST/"selection_progress.json",dict(stage="development_no_gate"))
        print("The fresh nodule references remain unopened",flush=True)
        return
    _,name,family=max(eligible)
    root=next(r for r in roots if r.name==name)
    seal(DEST/"validation_selection.json",dict(chosen=name,family=family,development=proposals,
              minimum=MIN_COMBINATION,rule="Among groups with all positive development directions and >=3 points over both individual approaches, maximize the smaller of those two gains."))
    command("joint_confirm.py",["lock","--root",root],0,"selected_validation_lock")
    assert (root/"external_lock.json").is_file()
    command("joint_confirm.py",["eval","--root",root,"--arm","anatomy"],0,"selected_validation_anatomy")
    batch([("joint_confirm.py",["eval","--root",root,"--arm",a,"--seed",s],f"selected_validation_{a}_{s}")
           for a in ARM for s in SEEDS],slots=(0,1,0,1))
    command("joint_confirm.py",["release","--root",root],0,"selected_validation_release")
    write(DEST/"selection_progress.json",dict(stage="fresh_verified",root=name))


def run():
    source="joint_nodule_growth_refine.py"
    write(DEST/"selection_progress.json",dict(stage="alternative_candidate_audit"))
    batch([(source,["dev","--kind",k],f"growth_new_dev_{k}") for k in "AB"])
    audits={}
    for kind in "AB":
        paths=list((GROWTH/"audit").glob(kind+"_*.json"))
        for p in paths:
            assert read(p)["role"]=="dev"
        audits[kind]=sum(read(p)["metrics"]["dice"] for p in paths)/len(paths)
    audit=read(DEST/"growth_summary.json")
    useful=sum(audits.values())>sum(audit[k]["teacher_shape"]["dice"] for k in "AB")
    seal(GROWTH/"candidate_decision.json",dict(mean_dice=audits,go=useful,
             guideline="shared growth ranking should improve average development overlap over the starting shaped teacher"))
    roots=[TRAIN]
    if useful:
        write(DEST/"selection_progress.json",dict(stage="alternative_unlabelled_regions",audit=audits))
        batch([(source,["unlabelled","--kind",k],f"growth_new_unlabelled_{k}") for k in "AB"])
        command(source,["lock"],0,"growth_protocol")
        write(DEST/"selection_progress.json",dict(stage="alternative_development_training",audit=audits))
        batch([(source,["train","--kind",k,"--seed",s,"--arm",a],f"growth_fit_{k}_{s}_{a}")
               for s in SEEDS for a in "SNUP" for k in "AB"])
        command("joint_metrics.py",[GROWTH],0,"growth_development_analysis")
        roots.append(GROWTH)
    main=read(DEST/"pipeline_pid.json")["pid"]
    while alive(main,"joint_nodule_pipeline.py") and not (TRAIN/"development_study.json").is_file():
        time.sleep(15)
    assert (TRAIN/"development_study.json").is_file()
    write(DEST/"selection_progress.json",dict(stage="comparing_development",variants=[r.name for r in roots]))
    select(roots)


if __name__=="__main__":
    os.umask(0o077)
    run()
