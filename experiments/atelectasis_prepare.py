"""Verify released image/mask grids, then prepare coded full-volume data on a 1.5 mm grid."""
import argparse, concurrent.futures, os, shutil, tempfile, time
from collections import Counter
from pathlib import Path
import nibabel as nib
import numpy as np
from scipy.ndimage import label
from matplotlib import use
use("Agg")
import matplotlib.pyplot as plt
from atelectasis_screen import ROOT
from fetch_manifest import FILES
from frozen_priors import read, sha, write
from prepare_unlabeled import obtain_file, release_path
from swap_data import reduce_ct, reduce_mask

GRID=1.5
DATA=ROOT/"arrays"
RECEIPTS=ROOT/"receipts"
PANELS=ROOT/"technical_panels"

def panel(code,ct,mask,coords):
    fig,axes=plt.subplots(1,3,figsize=(8,2.7))
    for axis,ax in enumerate(axes):
        k=int(np.argmax(mask.sum(tuple(j for j in range(3) if j!=axis))))
        image=np.take(ct,k,axis=axis);region=np.take(mask,k,axis=axis)
        other=np.delete(coords.mean(0),axis).astype(int)
        crop=tuple(slice(max(0,p-42),min(n,p+43)) for p,n in zip(other,image.shape))
        ax.imshow(image[crop].T,cmap="gray",vmin=-1000,vmax=200,origin="lower")
        ax.contour(region[crop].T,levels=[.5],colors=["#ffb444"],linewidths=.65,origin="lower",extent=None)
        ax.set_title(("Sagittal","Coronal","Axial")[axis],fontsize=9);ax.axis("off")
    fig.suptitle(code,fontsize=10);fig.tight_layout();fig.savefig(PANELS/(code+".png"),dpi=90);plt.close(fig)

def convert(row,render):
    code=row["code"];rec=RECEIPTS/(code+".json");out=DATA/(code+".npz")
    if rec.exists():
        result=read(rec)
        if result["status"]=="eligible":assert sha(out)==result["sha"]
        return result
    if shutil.disk_usage(ROOT).free < 2*1024**3:
        raise OSError("data disk has under 2 GiB free")
    with tempfile.TemporaryDirectory(prefix=code+"-",dir=ROOT) as tmp:
        repo,rev,_=FILES["ct_rate"];name=row["name"]
        ct_file,ct_hash=obtain_file(repo,rev,release_path(name),tmp)
        repo,rev,_=FILES["rex"]
        mask_file,mask_hash=obtain_file(repo,rev,"segmentations/"+name,tmp)
        image=nib.load(ct_file);seg=nib.load(mask_file)
        result=dict(code=code,ct_sha=ct_hash,rex_sha=mask_hash,status="excluded",native_shape=list(image.shape),
                    source_spacing=[float(x) for x in image.header.get_zooms()[:3]])
        def exclude(reason):
            result["reason"]=reason;write(rec,result);return result
        if len(image.shape)!=3 or len(seg.shape)!=4 or image.shape!=seg.shape[1:] or row["channel"]>=seg.shape[0]:
            return exclude("native_channel_grid")
        if nib.aff2axcodes(image.affine)!=("L","P","S") or not np.isfinite(image.affine).all():
            return exclude("raw_orientation")
        raw=np.asarray(seg.dataobj[row["channel"]])>0
        native_count=int(raw.sum())
        result["native_count"]=native_count
        if native_count!=row["native_count"] or not native_count:return exclude("native_count")
        ct,aff=reduce_ct(image,GRID)
        if not np.isfinite(ct).all() or not (ct.min() < -400 and ct.max()>50):
            return exclude("invalid_hu")
        mask=reduce_mask(raw,image.affine,(ct.shape,aff)).astype(bool)
        coords=np.argwhere(mask)
        n=len(coords)
        volume_ratio=n*GRID**3/(native_count*abs(np.linalg.det(image.affine[:3,:3])))
        result.update(voxels=n,shape=list(ct.shape),volume_ratio=float(volume_ratio),
                      original_volume_ml=float(native_count*abs(np.linalg.det(image.affine[:3,:3]))/1000))
        if n<8 or not .60<=volume_ratio<=1.45:return exclude("resampled_support")
        centre=(ct.shape[0]-1)/2
        reported=coords[:,0]>=centre if row["side"]=="right" else coords[:,0]<=centre
        result["reported_side_fraction"]=float(reported.mean())
        if reported.mean()<.80:return exclude("case_midline_and_label_disagree")
        _,components=label(mask,np.ones((3,3,3), dtype=bool))
        result.update(components=components,bounding_mm=((coords.max(0)-coords.min(0)+1)*GRID).tolist(),
                      mask_hu_median=float(np.median(ct[mask])))
        tmpfile=out.with_suffix(".part.npz")
        np.savez_compressed(tmpfile,ct=ct.astype(np.int16),mask=mask,affine=aff)
        tmpfile.replace(out);result.update(status="eligible",sha=sha(out),bytes=out.stat().st_size)
        if render:
            panel(code,ct,mask,coords)
        write(rec,result)
        return result

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--shard",type=int,default=0)
    parser.add_argument("--n-shards",type=int,default=3)
    args=parser.parse_args()
    os.umask(0o077)
    rows=read(ROOT/"screen.private.json")[args.shard::args.n_shards]
    for p in (DATA,RECEIPTS,PANELS):p.mkdir(exist_ok=True)
    for row in rows:
        for attempt in range(3):
            try:
                r=convert(row,True)
                print(row["code"],r["status"],r.get("reason",""),r.get("voxels",""),flush=True)
                break
            except Exception as exc:
                print(row["code"],"retry",attempt+1,type(exc).__name__,str(exc).split("http",1)[0][:70],flush=True)
                if attempt==2:
                    write(RECEIPTS/(row["code"]+".pending.json"),dict(code=row["code"],error=type(exc).__name__))
                else:time.sleep(3)
if __name__=="__main__":main()

