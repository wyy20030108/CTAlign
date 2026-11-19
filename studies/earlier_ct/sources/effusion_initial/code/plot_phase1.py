"""Paired per-patient developmental results: extent of predictions and coverage of partial reference."""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    base=Path("phase1/full_report_trainonly/auxiliary_AI/development_7")
    groups=[json.loads((base/(n+"_128.json")).read_text())["cases"] for n in ("no_aux","with_aux")]
    a,b=groups
    if [r["code"] for r in a]!=[r["code"] for r in b] or len(a)!=7: raise ValueError("Unpaired patients")
    fig,axes=plt.subplots(1,2,figsize=(9.0,4.5),layout="constrained")
    colors=plt.get_cmap("tab10").colors
    for ax,field,title,ylabel,multiple in zip(axes,
                    ["predicted_volume_mL","known_positive_coverage_fraction"],
                    ["Predicted effusion volume","Coverage of annotated foreground"],
                    ["Predicted volume (L)","Covered annotated voxels (%)"],[.001,100],strict=True):
        vals=np.array([[r[field] for r in lst] for lst in groups]).T*multiple
        for idx,row in enumerate(vals):
            ax.plot([0,1],row,"o-",color=colors[idx],ms=5,lw=1.4,alpha=.88,label=f"Case {idx+1}")
        ax.set(xticks=[0,1],xticklabels=["Without AI\nauxiliary labels","With AI\nauxiliary labels"],xlim=(-.14,1.14),ylabel=ylabel,title=title)
        ax.grid(axis="y",color=".88",lw=.7)
        ax.spines["top"].set_visible(False);ax.spines["right"].set_visible(False)
        if field=="predicted_volume_mL": ax.set_ylim(0,max(vals.max()*1.08,1))
        else:
            ax.set_ylim(max(0, np.floor(vals.min()-1)),100.6)
    axes[1].legend(loc="lower left",ncol=2,fontsize=7,frameon=False)
    fig.suptitle("Exploratory paired comparison on seven CT scans\n128 updates in both arms; CT-only student inference",fontsize=13)
    out=Path("phase1/reports");out.mkdir(exist_ok=True)
    fig.savefig(out/"development_pairs.png",dpi=200,bbox_inches="tight",facecolor="white")
    plt.close(fig)


if __name__=="__main__":
    main()
