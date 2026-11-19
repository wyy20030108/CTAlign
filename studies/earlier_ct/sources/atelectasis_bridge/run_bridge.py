from concurrent.futures import ThreadPoolExecutor
import os, subprocess, sys
from pathlib import Path
p=Path("phase8_thesis/original_budget_bridge")
def run(seed,gpu):
    command=[sys.executable,"-u","thesis_accelerated.py","thesis_supervised_bridge.py","run","--seed",str(seed)]
    with (p/(str(seed)+".log")).open("ab") as stream:
        subprocess.run(command,check=True,stdout=stream,stderr=stream,env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS="1"))
with ThreadPoolExecutor(2) as pool:
    futures=[pool.submit(run,seed,i%2) for i,seed in enumerate((891,1901,2707))]
    for f in futures:
        f.result()
subprocess.run([sys.executable,"-u","thesis_accelerated.py","thesis_supervised_bridge.py","release"],check=True)
