"""Score the qualified softened transport arm in a separately frozen evaluation."""
from argparse import ArgumentParser
from pathlib import Path
import os
import thesis_evaluate as evaluator
import thesis_model
from thesis_component_fast import segment
from thesis_soft_transport import OUT,DESIGN
from thesis_cohort import ROOT,PRIMARY,MANIFEST
from thesis_ssl import SEEDS
from frozen_priors import read,sha,write
from diag_localize import seal

LOCK=OUT/"independent_auxiliary_lock.json"
RESULTS=OUT/"independent_evaluation"
def freeze():
    trial=read(OUT/"decision.json")
    if not trial["auxiliary_independent"]:
        print("Secondary transport comparison is confined to development.")
        return
    main=read(evaluator.EVAL)
    records=[]
    for seed in SEEDS:
        r=OUT/PRIMARY/str(seed)/"F";fit=read(r/"training.json")
        assert fit["model_sha"]==sha(r/"model.pt") and fit["protocol"]==sha(OUT/"protocol.json")
        records.append(dict(seed=seed,model=fit["model_sha"],development=sha(r/"development.json"),
                            training=sha(r/"training.json")))
    selection=dict(chosen=trial["results"]["soft"]["threshold"],checkpoints=records)
    seal(LOCK,dict(manifest=sha(MANIFEST),evaluator=sha(Path(evaluator.__file__)),main=sha(evaluator.EVAL),
        training_protocol=sha(OUT/"protocol.json"),arms={PRIMARY:{"F":selection}},
        design=DESIGN,comparator=main["comparator"]))
    print("Auxiliary model independently frozen",flush=True)

def run(seed):
    assert seed in SEEDS and read(LOCK)["main"]==sha(evaluator.EVAL)
    thesis_model.segment=segment
    evaluator.segment=segment
    evaluator.OUT=OUT;evaluator.EVAL=LOCK;evaluator.RESULTS=RESULTS
    evaluator.evaluate(PRIMARY,seed,"F")

if __name__=="__main__":
    os.umask(0o077)
    p=ArgumentParser();p.add_argument("stage",choices=("lock","run"));p.add_argument("--seed",type=int,default=891)
    a=p.parse_args()
    if a.stage=="lock":freeze()
    else:run(a.seed)

