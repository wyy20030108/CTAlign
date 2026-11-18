"""Check learned foreground on the two masks exposed to the low-label runs."""
import sys
import numpy as np
import torch
from atelectasis_sliding import OUT, ROOT, DualScale, load_row, sliding, score
from finding_train import configure
from frozen_priors import read, sha, write

seed=int(sys.argv[1])
configure(seed)
d=OUT/"b10"/str(seed)
state=read(d/"results.json")
assert sha(d/"B1.pt")==state["model_sha"]
net=DualScale().cuda().eval()
net.load_state_dict(torch.load(d/"B1.pt",map_location="cuda",weights_only=True))
rows=[]
for case in read(ROOT/"pilot_private.json")["selected"]:
    ct,reference=load_row(case)
    fine,coarse,tiles=sliding(net,ct,case["query"],torch.device("cuda"))
    scores={k:score(x>=0,reference) for k,x in (("B1",fine),("B1_coarse",coarse))}
    entry={"code":case["code"],"tiles":tiles,"scores":scores,"fine_logits_in_target_median":float(np.median(fine[reference])),
           "fine_max_logit":float(fine.max())}
    rows.append(entry)
write(d/"training_field_check.json",dict(seed=seed,results=rows,checkpoint=state["model_sha"]))
print(seed,[(r["code"],r["scores"]["B1"]["dice"],r["scores"]["B1"]["pred"]) for r in rows],flush=True)

