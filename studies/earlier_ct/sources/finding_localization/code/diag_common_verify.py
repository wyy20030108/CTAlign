"""Verify released probabilities and visualize the same A/B/empty field without exporting clinical text."""
import os,sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from diag_common import OUT,ROOT,RULE,load_selected,score,QueryNet,configure,read,write,sha,fingerprint

def plot(row,probs,mode,dest):
    m=row["m"].cpu().numpy();ct=row["ct"]
    colors=["#ffad36","#52e0ea"];fig,axes=plt.subplots(2,4,figsize=(11.5,6.3))
    for j in range(2):
        z=int(np.argmax(m[j].sum(axis=(0,1))))
        for col,ax in enumerate(axes[j]):
            if col==0:ax.imshow(ct[:,:,z].T,vmin=-1000,vmax=200,cmap="gray",origin="lower")
            else:display=ax.imshow(probs[col-1,:,:,z].T,vmin=0,vmax=1,cmap="magma",origin="lower")
            for k in (0,1):
                plane=m[k,:,:,z].T
                if plane.any() and not plane.all():ax.contour(plane.astype(float),levels=[.5],colors=[colors[k]],linewidths=.85)
            ax.set_xticks([]);ax.set_yticks([])
            if j==0:ax.set_title(("CT and reference contours","P(target | query A)","P(target | query B)","P(target | empty)")[col],fontsize=10)
            if col==0:ax.set_ylabel(f'{"AB"[j]}: z={z+row["start"][2]}')
    fig.suptitle(f'{row["code"]}  |  identical CT field for all queries   |   {mode.replace("_"," ")}',fontsize=13)
    fig.subplots_adjust(left=.06,right=.89,bottom=.12,top=.86,wspace=.06,hspace=.08)
    fig.text(.06,.04,"Reference contours: orange A, cyan B. Rows show the largest axial section of each target. 2 mm RAS+.",fontsize=9)
    cax=fig.add_axes([.91,.26,.012,.45]);fig.colorbar(display,cax=cax,ticks=[0,.5,1])
    fig.savefig(dest/(row["code"]+"_probabilities.png"),dpi=150,bbox_inches="tight");plt.close(fig)

@torch.inference_mode()
def main(mode):
    os.umask(0o077);configure(891);dest=OUT/mode;items=load_selected()
    summ=read(dest/"summary.json");model=QueryNet().cuda().eval()
    assert sha(dest/"model.pt")==summ["weights_sha256"]
    model.load_state_dict(torch.load(dest/"model.pt",map_location="cpu",weights_only=True))
    results=[]
    for row,rec in zip(items,summ["cases"],strict=True):
        code=row["code"];f=dest/(code+".npz")
        assert rec["code"]==code and sha(f)==rec["map_sha256"]
        with np.load(f) as v:
            probs=v["probability"];assert np.array_equal(v["affine"],row["aff"]) and np.array_equal(v["start"],row["start"])
        m=row["m"].cpu().numpy();assert score(probs>=.5,m)==rec["matrix"]
        qs=row["queries"]
        fresh=model(row["x"],[*qs,""]).sigmoid().cpu().numpy()
        reversed_=model(row["x"],[qs[1],qs[0],""]).sigmoid().cpu().numpy()[[1,0,2]]
        gaps=[float(np.abs(p-q).max()) for p,q in [(probs,fresh),(fresh,reversed_)]]
        assert max(gaps)<1e-3
        plot(row,probs,mode,dest)
        results.append(dict(code=code,reference_pixels_recounted=True,probability_max_abs_reload_diff=gaps[0],
                  max_abs_query_permutation_diff=gaps[1],figure_sha=sha(dest/(code+"_probabilities.png"))))
    write(dest/"verification.json",dict(cases=results,verified=True,validator=sha(Path(__file__))))
if __name__=="__main__":main(sys.argv[1])
