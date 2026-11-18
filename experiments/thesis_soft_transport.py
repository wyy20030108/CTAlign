"""One prospectively specified, confidence-preserving transport sensitivity."""
from argparse import ArgumentParser
from pathlib import Path
from statistics import mean
import shutil,time,os
from frozen_priors import read,write,sha
from diag_localize import seal
import thesis_ssl as ssl
import thesis_model
from thesis_component_fast import segment
from thesis_lobe_model import LobeStore
from thesis_cohort import ROOT,PRIMARY

SOURCE=ssl.OUT
OUT=ROOT/"controlled_soft_transport"
DESIGN=dict(method="Equal-weight mixture of original teacher probability and binary Adaptive UOT probability",
    unchanged="Same accepted voxels, augmentation, labelled positive prior, seed, warmup checkpoints, report pairing, patient splits and updates as original F.",
    choice="Eligible for a prespecified auxiliary independent comparison only if all 3 dev seeds collectively beat original F and the stronger of B and D by at least 1 Dice percentage point after the standard calibration rule. The original F remains reported.")

def run(seed):
    os.umask(0o077)
    OUT.mkdir(exist_ok=True)
    original=ssl.adaptive_uot
    def softened(p,valid,prior):
        q,lam,pre,post=original(p,valid,prior)
        mixed=.5*p+.5*q
        return mixed,lam,pre,float(mixed[valid].mean())
    ssl.adaptive_uot=softened
    thesis_model.segment=segment
    ssl.OUT=OUT
    protocol=OUT/"protocol.json"
    seal(protocol,dict(sha=sha(Path(ssl.__file__)),wrapper=sha(Path(__file__)),description=DESIGN,
                       main_protocol=sha(SOURCE/"protocol.json"),seed_list=list(ssl.SEEDS)))
    initialized=SOURCE/PRIMARY/str(seed)/"initial"/"B.pt"
    for _ in range(180):
        if initialized.is_file() and initialized.with_suffix(".json").is_file():break
        time.sleep(10)
    else:raise TimeoutError("Original supervised starting point did not appear")
    origin=read(initialized.with_suffix(".json"))
    assert origin["sha"]==sha(initialized) and origin["plan"]==sha(SOURCE/"protocol.json")
    book=OUT/PRIMARY/str(seed)/"initial";book.mkdir(exist_ok=True,parents=True)
    dst=book/"B.pt"
    if not dst.exists():shutil.copyfile(initialized,dst)
    assert sha(dst)==origin["sha"]
    seal(dst.with_suffix(".json"),dict(sha=origin["sha"],plan=sha(protocol),original_plan=origin["plan"],
                               key="original B transferred unchanged"))
    ssl.fit_arm(PRIMARY,seed,"F")

def aggregate():
    settings={}
    for root,arm,label in ((SOURCE,"B","B"),(SOURCE,"D","D"),(SOURCE,"F","F"),(OUT,"F","soft")):
        byseed=[read(root/PRIMARY/str(seed)/arm/"development.json") for seed in ssl.SEEDS]
        metrics={str(t):mean(row["values"][str(t)]["dice"] for s in byseed for row in s["rows"])
                 for t in thesis_model.THRESHOLDS}
        key=max(metrics,key=lambda t:(metrics[t],-abs(float(t)-.5),float(t)))
        settings[label]=dict(threshold=key,dev_dice=metrics[key],all=metrics)
    cutoff=max(settings[a]["dev_dice"] for a in ("B","D","F"))+.01
    proposed=settings["soft"]["dev_dice"]>=cutoff
    seal(OUT/"decision.json",dict(results=settings,declared_threshold=cutoff,auxiliary_independent=proposed,plan=sha(OUT/"protocol.json")))
    print(read(OUT/"decision.json"),flush=True)

if __name__=="__main__":
    p=ArgumentParser();p.add_argument("stage",choices=("run","compare"));p.add_argument("--seed",type=int,default=891)
    a=p.parse_args()
    if a.stage=="run":run(a.seed)
    else:aggregate()

