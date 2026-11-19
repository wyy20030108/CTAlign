"""Permute only U image identities; replay all training sentences and augmented-view seeds in the original order."""
import argparse,os
from pathlib import Path
import numpy as np
import finding_train as tr
import finding_eval as ev
from audit_pairs import identity

def main(seed,action):
    dest=tr.OUT/"unpaired_U"
    if action=="evaluate":
        ev.OUT=dest
        ev.evals(seed,["semi"])
        return
    normal=tr.load_split
    def permuted(role):
        rows=normal(role)
        if role!="unlabelled_train":return rows
        # All 32 original CTs and all sentence draws retained; same-patient links all removed.
        rng=np.random.default_rng(55621)
        cycle=rng.permutation(len(rows));donor=np.empty(len(rows),dtype=int)
        donor[cycle]=np.roll(cycle,1)
        assert len(set(donor))==32 and not np.any(donor==np.arange(32))
        out=[]
        for i,r in enumerate(rows):
            other=rows[donor[i]]
            assert identity(other["name"])[1]!=identity(r["name"])[1]
            out.append(dict(r,x=other["x"],shape=other["shape"],name=other["name"]))
        base=dest/str(seed);base.mkdir(parents=True,exist_ok=True)
        tr.write(base/"intervention.json",dict(seed=seed,unlabelled_CT_permutation=donor.tolist(),
                 fixed_points=0,script_sha256=tr.sha(Path(__file__)),description="Same sequence of sentences, swaps U image assignments only; unrestricted cross-patient control."))
        return out
    tr.OUT=dest;tr.load_split=permuted;tr.train(seed,["semi"])
if __name__=="__main__":
    os.umask(0o077);p=argparse.ArgumentParser();p.add_argument("seed",type=int,choices=tr.SEEDS)
    p.add_argument("action",choices=["train","evaluate"]);a=p.parse_args();main(a.seed,a.action)
