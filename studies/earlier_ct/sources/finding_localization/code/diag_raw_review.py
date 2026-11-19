"""S002 small-channel through-plane check and a record of the visual source audit."""
import os
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from scipy import ndimage
from diag_raw import OUT,read,sha256,write_json,clip_region,pane

def main():
    data=read(OUT/"raw_audit.json")
    paths={v["publisher_path"]:v for v in read(OUT/"raw_receipts.private.json") if v["code"]=="S002"}
    src=next(v for k,v in paths.items() if "_fixed/" in k)
    target=next(v for k,v in paths.items() if k.startswith("segmentations/"))
    ctimg=nib.load(src["local"]);seg=nib.load(target["local"])
    raw=np.asarray(seg.dataobj[0])>0
    hu=np.asarray(nib.as_closest_canonical(ctimg).dataobj)
    mask=np.asarray(nib.as_closest_canonical(nib.Nifti1Image(raw.astype(np.uint8),ctimg.affine)).dataobj).astype(bool)
    zs=np.flatnonzero(mask.any((0,1)));px=np.argwhere(mask)
    center=px.mean(0)[:2]
    region=clip_region(center,mask.shape[:2],25)
    fig,axs=plt.subplots((len(zs)+2)//3,3,figsize=(9,3*((len(zs)+2)//3)),squeeze=False)
    for ax in axs.flat:ax.set_visible(False)
    for ax,z in zip(axs.flat,zs):
        ax.set_visible(True)
        pane(ax,hu[:,:,z],mask[:,:,z],region,True,f"Native axial k={int(z)}")
    fig.suptitle("S002 A: every native CT slice intersecting ReX channel 0",fontsize=12)
    fig.subplots_adjust(left=.02,right=.98,bottom=.025,top=.925,hspace=.18,wspace=.10)
    p=OUT/"raw_visual/S002_A_stack.png";fig.savefig(p,dpi=125);plt.close(fig)
    vals=hu[mask]
    geometry=data["cases"]["S002"]["channels"][0]
    info=dict(all_native_slices=len(zs),native_ROI_voxels=len(vals),HU_summary=np.percentile(vals,[0,10,50,90,95,99,100]).tolist(),
              fraction_native_HU_above_minus600=float(np.mean(vals>-600)),fraction_native_HU_above_minus300=float(np.mean(vals>-300)),
              figure_sha256=sha256(p),raw_channel_shape=geometry["raw"]["bbox_extent_native_axes_mm"])
    write_json(OUT/"raw_s002_planes.json",info);print(info)

if __name__=="__main__":
    os.umask(0o077);main()
