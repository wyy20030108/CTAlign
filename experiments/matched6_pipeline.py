"""Schedule the predeclared development sequence and seal the first cohort."""
import concurrent.futures,os,subprocess,sys,time
from pathlib import Path
from frozen_priors import read,write,sha

out=Path("phase6_matched")
def call(gpu,script,args):
    subprocess.run([sys.executable,script,*map(str,args)],check=True,env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu)))
def next_train(gpu,first,new,sensitivity):
    # Prior seed workers were launched by the parent process through SSH.
    for attempt in range(1800):
        if (out/str(first)/"training.json").exists():break
        time.sleep(2)
    else:raise TimeoutError(str(first))
    call(gpu,"matched6_sensitivity.py" if sensitivity else "matched6.py",["train"] if sensitivity else [new,"train"])
def score_batch(gpu,seeds,stage):
    for seed in seeds:
        if seed==0:call(gpu,"matched6_sensitivity.py",[stage])
        else:call(gpu,"matched6.py",[seed,stage])
with concurrent.futures.ThreadPoolExecutor(2) as pool:
    futures=[pool.submit(next_train,0,891,2707,False),pool.submit(next_train,1,1901,0,True)]
    for f in futures:f.result()
    for stage in ("fallback_train","fallback_val","swap_dev"):
        futures=[pool.submit(score_batch,0,[891,2707],stage),pool.submit(score_batch,1,[1901,0],stage)]
        for f in futures:f.result()
        if stage=="fallback_val":
            files=list(out.rglob("fallback_val/scores.json"))
            assert len(files)==4
            write(out/"development_lock.json",dict(protocol=sha(out/"protocol.json"),first_scores={str(p):sha(p) for p in files},models={str(p):sha(p) for p in out.rglob("model.pt")}))
    print("Completed four models, three cohorts, first development scores sealed before replication",flush=True)

