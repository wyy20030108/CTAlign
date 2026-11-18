"""Run all development controls and budget sensitivities, leaving the holdout untouched."""
from argparse import ArgumentParser
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from queue import Queue,Empty
import json,os,subprocess,sys,time
from thesis_expand_screen import ROOT,SOURCES
from thesis_report_only import U
from thesis_cohort import MANIFEST,LOCK,PRIMARY
from frozen_priors import read,write,sha
from thesis_ssl import SEEDS,ARMS

def call(stream,script,args=(),gpu=None):
    env=dict(os.environ,OMP_NUM_THREADS="1",OPENBLAS_NUM_THREADS="1")
    if gpu is not None:env["CUDA_VISIBLE_DEVICES"]=str(gpu)
    accelerated=(ROOT/"component_equivalence.json").exists() and script in {
        "thesis_ssl.py","thesis_evaluate.py","thesis_dual_eval.py"}
    prefix=["thesis_accelerated.py"] if accelerated else []
    return subprocess.run([sys.executable,"-u",*prefix,script,*map(str,args)],env=env,stdout=stream,stderr=stream,check=True)
def wait_for_preparation():
    required=(("anatomy_candidates",len(read(SOURCES)),ROOT/"receipts","N???.json","prepare_workers.json"),
              ("unlabelled_scans",len(read(U/"source.private.json")),U/"receipts","U???.json","u_workers.json"))
    last=None
    while True:
        now={key:len(list(path.glob(pattern))) for key,target,path,pattern,workers in required}
        if now!=last:print("Data status",now,flush=True);last=now;write(ROOT/"pipeline_progress.json",dict(stage="preparation",now=now))
        if all(now[key]==target for key,target,*_ in required):break
        for key,target,path,pattern,workers in required:
            if now[key]==target:continue
            pids=read(ROOT/workers)
            live=[x for x in pids if Path(f"/proc/{x}/cmdline").exists() and Path(f"/proc/{x}/cmdline").stat().st_size>=0 and
                  b"thesis_" in Path(f"/proc/{x}/cmdline").read_bytes()]
            if not live:raise RuntimeError(f"{key} transfer workers exited with unfinished cases")
        time.sleep(20)
def stage_health():
    print(json.dumps({str(p):len(list(p.glob("*.json"))) for p in (ROOT/"receipts",U/"receipts")}),flush=True)
def run():
    wait_for_preparation()
    precache=ROOT/"precache_workers.json"
    if precache.exists():
        live=read(precache)
        while any((Path(f"/proc/{pid}/cmdline").exists()
                   and b"thesis_lobes.py" in Path(f"/proc/{pid}/cmdline").read_bytes()) for pid in live):
            time.sleep(10)
    os.umask(0o077)
    with (ROOT/"cohort_build.log").open("ab",buffering=0) as f:call(f,"thesis_cohort.py")
    with (ROOT/"seal_controls.log").open("ab",buffering=0) as f:call(f,"thesis_ssl.py",["seal"])
    write(ROOT/"pipeline_progress.json",dict(stage="anatomy",manifest_sha=sha(MANIFEST)))
    def anatomy(gpu):
        with (ROOT/f"expanded_anatomy_gpu{gpu}.log").open("ab",buffering=0) as f:
            call(f,"thesis_lobes.py",["expanded","--shards",2,"--shard",gpu],gpu)
    with ThreadPoolExecutor(2) as pool:
        jobs=[pool.submit(anatomy,g) for g in (0,1)]
        for f in jobs:f.result()
    queue=Queue()
    for budget,arms in ((PRIMARY,ARMS),("10","BF"),("5","BF")):
        for seed in SEEDS:queue.put((budget,seed,arms))
    def experiments(gpu):
        while True:
            try:budget,seed,arms=queue.get_nowait()
            except Empty:return
            print("Assigned",gpu,budget,seed,arms,flush=True)
            with (ROOT/f"development_{budget}_{seed}.log").open("ab",buffering=0) as f:
                call(f,"thesis_ssl.py",["run","--budget",budget,"--seed",seed,"--arms",arms],gpu)
    write(ROOT/"pipeline_progress.json",dict(stage="training",manifest_sha=sha(MANIFEST)))
    with ThreadPoolExecutor(2) as pool:
        tasks=[pool.submit(experiments,g) for g in (0,1)]
        for t in tasks:t.result()
    write(ROOT/"pipeline_progress.json",dict(stage="development_complete",manifest_sha=sha(MANIFEST)))
    print("Development controls complete; independent labels remain held.",flush=True)
if __name__=="__main__":
    p=ArgumentParser();p.add_argument("--status",action="store_true");a=p.parse_args()
    if a.status:stage_health()
    else:run()

