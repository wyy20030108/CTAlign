"""Run one sealed evaluation and keep optional dual-query checks in the same lock."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from queue import Queue,Empty
import os,time,sys,subprocess
from thesis_pipeline import call
from thesis_evaluate import EVAL,RESULTS
from thesis_dual import OUT as DUAL,LOCK as DUAL_SOURCE
from thesis_dual_eval import PAIR_LOCK
from thesis_cohort import ROOT,MANIFEST,PRIMARY
from thesis_ssl import SEEDS
from frozen_priors import read,sha,write

def batch(commands,tag,slots=(0,1,0,1)):
    q=Queue()
    for x in commands:q.put(x)
    def worker(index,gpu):
        while True:
            try:(script,args)=q.get_nowait()
            except Empty:return
            name=tag+"_"+str(index)+".log"
            print(tag,gpu,script,args,flush=True)
            with (ROOT/name).open("ab",buffering=0) as stream:
                call(stream,script,args,gpu)
    with ThreadPoolExecutor(len(slots)) as pool:
        futures=[pool.submit(worker,index,gpu) for index,gpu in enumerate(slots)]
        for f in futures:f.result()

def main():
    os.umask(0o077)
    f=read(EVAL)
    assert f["manifest"]==sha(MANIFEST)
    group=[("thesis_lobes.py",["test","--shards",2,"--shard",i]) for i in (0,1)]
    batch(group,"test_anatomy",(0,1))
    procedures=[]
    for budget,arms in f["arms"].items():
        for seed in SEEDS:
            procedures.append(("thesis_evaluate.py",["run","--budget",budget,"--seed",seed,"--arms","".join(arms)]))
    softened=ROOT/"controlled_soft_transport/decision.json"
    with (ROOT/"soft_transport_freeze.log").open("ab",buffering=0) as stream:
        call(stream,"thesis_soft_eval.py",["lock"])
    if read(softened)["auxiliary_independent"]:
        procedures.extend(("thesis_soft_eval.py",["run","--seed",seed]) for seed in SEEDS)
    write(ROOT/"pipeline_progress.json",dict(stage="independent_evaluation",lock=sha(EVAL)))
    batch(procedures,"independent")
    receipts={r["code"] for r in read(DUAL_SOURCE)}
    assert all((DUAL/(c+".json")).exists() for c in receipts)
    with (ROOT/"freeze_dual.log").open("ab",buffering=0) as f:call(f,"thesis_dual_eval.py",["lock"])
    batch([("thesis_dual_eval.py",["run","--seed",s]) for s in SEEDS],"same_ct_exchange")
    with (ROOT/"final_verification.log").open("ab",buffering=0) as f:call(f,"thesis_release.py")
    write(ROOT/"pipeline_progress.json",dict(stage="evidence_ready",lock=sha(EVAL),result=sha(ROOT/"completed_study.json")))
if __name__=="__main__":main()

