"""Run the predeclared B0 pilots and validate the outputs on completion."""
from concurrent.futures import ThreadPoolExecutor
import argparse
import os,subprocess,sys
from atelectasis_screen import ROOT
from frozen_priors import read,write

def evaluate(gpu, runs):
    with (ROOT/f"baseline-gpu{gpu}.log").open("ab",buffering=0) as log:
        for seed,budget in runs:
            subprocess.run([sys.executable,"-u","atelectasis_b0.py","train","--seed",str(seed),"--budget",budget],
                check=True,stdout=log,stderr=log,env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS="1"))

def main():
    p=argparse.ArgumentParser();p.add_argument("--allow-unresolved",action="store_true")
    args=p.parse_args()
    rows=read(ROOT/"screen.private.json")
    processed={f.stem for f in (ROOT/"receipts").glob("L???.json")}
    if not args.allow_unresolved and len(processed)!=len(rows):
        raise RuntimeError("Resolve the cohort downloads prior to training")
    subprocess.run([sys.executable,"-u","atelectasis_b0.py","freeze"],check=True)
    write(ROOT/"baseline_runner_started.json",{"pid":os.getpid(),"budget_pilots":[891,1901,2707],"all_labeled_training_pool_seed":891})
    with ThreadPoolExecutor(2) as pool:
        jobs=[pool.submit(evaluate,0,[(891,"b10"),(2707,"b10")]),pool.submit(evaluate,1,[(1901,"b10"),(891,"all_labeled")])]
        for j in jobs:j.result()
    subprocess.run([sys.executable,"-u","atelectasis_release.py"],check=True)
if __name__=="__main__":main()

