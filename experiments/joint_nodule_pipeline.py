"""Run the prepared second phenotype with the prespecified factorial comparison."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from queue import Empty, Queue
import os
import subprocess
import sys
import time

from frozen_priors import read, sha, write
from joint_nodule_cohort import DEST
from joint_nodule_run import COHORT, TRAIN
from joint_metrics import ARM
from thesis_ssl import SEEDS


def alive(pid, command):
    try:
        return command.encode() in Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0")
    except FileNotFoundError:
        return False


def progress(stage):
    write(DEST/"progress.json",dict(stage=stage, cohort=sha(COHORT) if COHORT.exists() else None))


def command(script,args,gpu,label):
    env=dict(os.environ,OMP_NUM_THREADS="1",OPENBLAS_NUM_THREADS="1",CUDA_VISIBLE_DEVICES=str(gpu))
    with (DEST/(label+".log")).open("ab") as f:
        subprocess.run([sys.executable,"-u",script,*map(str,args)],stdin=subprocess.DEVNULL,
                       stdout=f,stderr=f,env=env,check=True)


def batch(procedures,slots=(0,1)):
    jobs=Queue()
    for item in procedures:
        jobs.put(item)
    def process(gpu):
        while True:
            try:
                script,args,label=jobs.get_nowait()
            except Empty:
                return
            command(script,args,gpu,label)
    with ThreadPoolExecutor(len(slots)) as pool:
        handles=[pool.submit(process,gpu) for gpu in slots]
        for h in handles:
            h.result()


def run():
    p=DEST/"download_pids.json"
    progress("downloads")
    while any(alive(pid,"joint_nodule_cohort.py") for pid in read(p).values()):
        counts=[len(list(path.glob("X????.json"))) for path in (DEST/"receipts",DEST/"report_only/receipts")]
        print("Acquisition",counts,flush=True)
        time.sleep(20)
    src="joint_nodule_run.py"
    command(src,["build"],0,"build_roles")
    progress("anatomy")
    batch([(src,["anatomy","--shard",i,"--shards",2],f"anatomy_{i}") for i in range(2)])
    progress("warmup")
    command(src,["warm-lock"],0,"warm_protocol")
    batch([(src,["warm","--kind",k,"--seed",s],f"warm_{k}_{s}") for s in SEEDS for k in "AB"])
    progress("candidate_regions")
    batch([(src,[role,"--kind",k],f"candidate_{role}_{k}") for role in ("unlabelled","dev") for k in "AB"])
    command(src,["lock"],0,"factorial_protocol")
    progress("development_training")
    batch([(src,["train","--kind",k,"--seed",s,"--arm",a],f"fit_{k}_{s}_{a}")
           for s in SEEDS for a in "SNUP" for k in "AB"])
    command("joint_metrics.py",[TRAIN],0,"development_analysis")
    command("joint_confirm.py",["lock","--root",TRAIN],0,"confirm_gate")
    if not (TRAIN/"external_lock.json").exists():
        progress("tested_development_variants_no_gate")
        return
    progress("fresh_test")
    command("joint_confirm.py",["eval","--root",TRAIN,"--arm","anatomy"],0,"fresh_anatomy")
    batch([("joint_confirm.py",["eval","--root",TRAIN,"--arm",a,"--seed",s],f"fresh_{a}_{s}")
           for a in ARM for s in SEEDS],slots=(0,1,0,1))
    command("joint_confirm.py",["release","--root",TRAIN],0,"fresh_verification")
    progress("fresh_verified")


if __name__=="__main__":
    os.umask(0o077)
    run()
