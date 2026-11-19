"""Independent training-case provenance, geometry and paired-phrase review for the phase five diagnostic."""
import csv
import hashlib
import json
import os
import re
import sys
from pathlib import Path
import numpy as np
import nibabel as nib
from scipy import ndimage
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from fetch_four import write_json, sha256
from swap_data import norm
from replay_controls import read_reports
sys.path.insert(0, str(Path("phase1/geometry").resolve()))
from geometry import check_ct, reduce_ct


OUT=Path("phase5_diag")
def read(p):return json.loads(Path(p).read_text())

# The two source anchors for each patient are drawn directly from the released CT-RATE text.
ANCHORS={
    "S002":("A nonspecific parenchymal nodule with a diameter of 3 mm","Minimal contour irregularities were observed in the pleura"),
    "S010":("There are linear atelectasis in the upper, middle and lower lobes of the right lung",
            "A slightly irregular nodule measuring approximately 12 mm in diameter"),
    "S030":("Peripheral consolidations are observed in both lungs","Although the described views are not specific"),
    "S046":("Apart from these, local thickening is observed in the bronchovascular structures",
            "The thickening of the bronchovascular structures is most evident")}

def anatomy_stats(m,aff,ct,s):
    pos=np.argwhere(m)
    coords=nib.affines.apply_affine(aff,pos)
    body_center=nib.affines.apply_affine(aff,(np.array(m.shape)-1)/2)
    length_vox=pos.max(0)-pos.min(0)+1
    comps,n=ndimage.label(m,structure=np.ones((3,3,3),np.uint8))
    sizes=np.sort(np.bincount(comps[m]))[::-1]
    sizes=sizes[sizes>0]
    assert n==len(sizes) and int(sizes.sum())==len(pos)
    vol=float(abs(np.linalg.det(aff[:3,:3]))/1000*len(pos))
    return dict(voxels=len(pos),voxel_centre_RAS_mm=coords.mean(0).tolist(),
                bbox_native_ijk_min=pos.min(0).tolist(),bbox_native_ijk_max=pos.max(0).tolist(),
                bbox_extent_native_axes_mm=(length_vox*nib.affines.voxel_sizes(aff)).tolist(),
                volume_ml=vol,equivalent_sphere_diameter_mm=float((6000*vol/np.pi)**(1/3)),
                components_26=int(n),largest_component_counts=sizes[:10].tolist(),
                fraction_on_reported_side=float(np.mean((coords[:,0]-body_center[0])*(1 if s=="right" else -1)>0)) if s else None,
                CT_HU_quantiles_p10_p50_p90=np.percentile(ct[m],[10,50,90]).tolist())

def pane(ax,arr,mask,bounds,outline=False,title=""):
    x0,x1,y0,y1=bounds;view=arr[x0:x1,y0:y1].T
    ax.imshow(view,vmin=-950,vmax=200,cmap="gray",origin="lower",interpolation="nearest",
              extent=(x0-.5,x1-.5,y0-.5,y1-.5),aspect="equal")
    if outline:
        cropped=mask[x0:x1,y0:y1]
        if np.any(cropped) and not np.all(cropped):
            ax.contour(np.arange(x0,x1),np.arange(y0,y1),cropped.T,[.5],colors=["#ffb141"],linewidths=.9)
    ax.set_xlim(x0-.5,x1-.5);ax.set_ylim(y0-.5,y1-.5);ax.set_title(title,fontsize=9);ax.set_xticks([]);ax.set_yticks([])

def clip_region(center,shape,side):
    hw=min(int(side),min(shape)//2)
    ctr=np.clip(np.rint(center).astype(int),hw,np.asarray(shape)-hw)
    return [ctr[0]-hw,ctr[0]+hw,ctr[1]-hw,ctr[1]+hw]

def make_figure(code,fixed,raw_arrays,cached_ct,cached_masks,cache_aff):
    # Fixed raw input grid is axis-aligned LPS; after canonical orientation all views show RAS+.
    obj=nib.as_closest_canonical(fixed)
    oriented=np.asarray(obj.dataobj)
    fig,axes=plt.subplots(2,4,figsize=(13.5,7))
    raw_info=[]
    for j,m in enumerate(raw_arrays):
        label=nib.as_closest_canonical(nib.Nifti1Image(m.astype("uint8"),fixed.affine))
        u=np.asarray(label.dataobj)>0
        assert np.array_equal(label.affine,obj.affine)
        z=int(np.argmax(u.sum((0,1))));cent=np.argwhere(u[:,:,z]).mean(0)
        spread=np.ptp(np.argwhere(u[:,:,z]),axis=0).max()
        native_zoom=clip_region(cent,u.shape[:2],max(24.,spread*.61))
        raster=oriented[:,:,z];m2=u[:,:,z]
        pane(axes[j,0],raster,m2,[0,raster.shape[0],0,raster.shape[1]],True,("A","B")[j]+"  Native CT, axial")
        x0,x1,y0,y1=native_zoom;axes[j,0].add_patch(Rectangle((x0,y0),x1-x0,y1-y0,fill=False,edgecolor="#50d4f3",linewidth=.75))
        pane(axes[j,1],raster,m2,native_zoom,False,"Native close-up")
        pane(axes[j,2],raster,m2,native_zoom,True,"Original ReX outline")
        wz=nib.affines.apply_affine(obj.affine,[*cent,z])
        ijk=nib.affines.apply_affine(np.linalg.inv(cache_aff),wz)
        kz=int(np.clip(np.rint(ijk[2]),0,cached_ct.shape[2]-1))
        half=max(12,int(round((x1-x0)/2*nib.affines.voxel_sizes(obj.affine)[0]/2)))
        zoom=clip_region(ijk[:2],cached_ct.shape[:2],half)
        pane(axes[j,3],cached_ct[:,:,kz],cached_masks[j,:,:,kz],zoom,True,"2-mm nearest plane (cached)")
        raw_info.append(dict(raw_canonical_z=z,cached_z=kz))
    fig.suptitle(code+"  |  Paired training findings on release CT and 2-mm reconstruction",fontsize=13)
    fig.text(.5,.02,"Display convention: patient left on the left; anterior at the top. Axial plane maximises native mask area. Outlines are dataset annotations.",ha="center",fontsize=9)
    fig.tight_layout(rect=(0,.045,1,.95));p=OUT/"raw_visual"/(code+".png");fig.savefig(p,dpi=130);plt.close(fig)
    return dict(path=str(p),sha256=sha256(p),slices=raw_info)

def main():
    os.umask(0o077);(OUT/"raw_visual").mkdir(exist_ok=True)
    sel=read(OUT/"selection_clean.json"); codes=sel["source_audit_codes"]
    rows={r["code"]:r for r in read("phase3_swap/fallback/fallback_train.private.json") if r["code"] in codes}
    names={r["name"] for r in rows.values()};docs=read_reports(names)
    rex={r["name"]:r for r in read("metadata/rex/dataset.json")["train"] if r["name"] in names}
    source=read(OUT/"raw_receipts.private.json"); meta={}
    for fp in Path("data/four_train/ct_rate/dataset/metadata").glob("*metadata.csv"):
        with fp.open(encoding="utf-8-sig",newline="") as f:
            for v in csv.DictReader(f):
                if v["VolumeName"] in names:meta[v["VolumeName"]]=v
    result={};private={}
    for c in codes:
        r=rows[c];n=r["name"];rr=rex[n];record=docs[n];assert r["report"]==record
        paths={("rex" if t["publisher_path"].startswith("segmentations/") else "fixed" if "_fixed/" in t["publisher_path"] else "v1"):t for t in source if t["code"]==c}
        assert set(paths)=={"v1","fixed","rex"} and all(Path(t["local"]).name==n and sha256(Path(t["local"]))==t["sha256"] for t in paths.values())
        assert paths["fixed"]["sha256"]==r["source_ct"] and paths["rex"]["sha256"]==r["source_rex"] and sha256(Path(r["path"]))==r["sha256"]
        v1,fix,seg=[nib.load(paths[k]["local"]) for k in ("v1","fixed","rex")]
        hu,cal=check_ct(v1,fix,meta[n])
        assert seg.shape==(len(rr["findings"]),*fix.shape) and tuple(rr["shape"])==fix.shape and nib.aff2axcodes(fix.affine)==("L","P","S")
        with np.load(r["path"]) as z: cached,masks,aff=z["ct"],z["masks"],z["affine"]
        ct2,aff2=reduce_ct(fix,2);assert np.array_equal(ct2,cached) and np.array_equal(aff2,aff)
        grid=np.linalg.inv(fix.affine)@aff;raw_arrays=[];items=[]
        sentences=[s.strip() for s in re.split(r'(?<=[.!?])\s+',record["Findings_EN"]+" "+record["Impressions_EN"])]
        related=[]
        for j,k in enumerate(r["channels"]):
            q=rr["findings"][str(k)];assert r["queries"][j]==q and rr["categories"][str(k)]==r["categories"][j]
            near=[s for s in sentences if ANCHORS[c][j] in s]
            assert near
            related.append(dict(query=q,source_sentence=near[0],channel=k))
            raw=np.asarray(seg.dataobj[k])>0;raw_arrays.append(raw)
            assert int(raw.sum())==r["native_counts"][j]==rr["pixels"][str(k)]
            rea=ndimage.affine_transform(raw.astype(np.uint8),grid[:3,:3],offset=grid[:3,3],output_shape=cached.shape,
                                         order=0,prefilter=False,mode="constant",cval=0).astype(bool)
            assert np.array_equal(rea,masks[j])
            src_metrics=anatomy_stats(raw,fix.affine,hu,r["sides"][j])
            res_metrics=anatomy_stats(rea,aff,cached,r["sides"][j])
            items.append(dict(query="AB"[j],channel=k,category=r["categories"][j],published_side=r["sides"][j],
                              public_query_equals_manifest=True,source_sentence_verified=True,raw=src_metrics,
                              two_mm=res_metrics,volume_change_percent=100*(res_metrics["volume_ml"]/src_metrics["volume_ml"]-1),
                              centroid_displacement_mm=float(np.linalg.norm(np.array(src_metrics["voxel_centre_RAS_mm"])-res_metrics["voxel_centre_RAS_mm"]))))
        private[c]=related
        result[c]=dict(code=c,channels=items,revisions={k:t["rev"] for k,t in paths.items()},source_hashes={k:t["sha256"] for k,t in paths.items()},
                       CT_shape=list(fix.shape),raw_mask_shape=list(seg.shape),CT_grid=cal,native_ct_and_2mm_cached_equal=True,
                       both_independently_resampled_masks_identical=True,raw_overlap_voxels=int((raw_arrays[0]&raw_arrays[1]).sum()),
                       two_mm_overlap_voxels=int((masks[0]&masks[1]).sum()),figure=make_figure(c,fix,raw_arrays,cached,masks,aff))
        print("verified official data and geometry",c,flush=True)
    write_json(OUT/"raw_source_text.private.json",private)
    write_json(OUT/"raw_audit.json",dict(cases=result,clean_codes=sel["codes"],
               selection_sha256=sha256(OUT/"selection_clean.json"),script_sha256=sha256(Path(__file__)),
               scope="Native ReX channel indices aligned to corresponding corrected CT; voxelwise identity verifies pipeline consistency. Laterality uses FOV midpoint. Notes and figures are engineering reviews, not independent clinical readings."))
if __name__=="__main__":main()
