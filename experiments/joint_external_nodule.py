"""Audit an externally trained, frozen 3D nodule detector as a shared localization prior."""
from argparse import ArgumentParser
from pathlib import Path
import os
import sys

import nibabel as nib
import numpy as np
from scipy.spatial import cKDTree
import torch

from frozen_priors import read,sha,write
from joint_nodule_run import COHORT,DEST


BUNDLE=Path("/tmp/joint_external_nodule")
CHECKPOINT=BUNDLE/"model.pt"
DIGEST="b5e79231466adae93a6fe8e8594029e9add142914e223b879aa0343bb2402d01"
OUT=DEST/"external_luna16"
SPACING=(.703125,.703125,1.25)
PATCH=(256,256,96)


def detector():
    sys.path.insert(0,str(BUNDLE/"deps"))
    from monai.apps.detection.networks.retinanet_detector import RetinaNetDetector
    from monai.apps.detection.networks.retinanet_network import RetinaNet,resnet_fpn_feature_extractor
    from monai.apps.detection.utils.anchor_utils import AnchorGeneratorWithAnchorShape
    from monai.networks.nets.resnet import resnet50

    assert sha(CHECKPOINT)==DIGEST
    backbone=resnet50(spatial_dims=3,n_input_channels=1,conv1_t_stride=[2,2,1],conv1_t_size=[7,7,7])
    extractor=resnet_fpn_feature_extractor(backbone,3,False,[1,2],None)
    classifier=RetinaNet(3,1,3,extractor,size_divisible=[16,16,8],use_list_output=False)
    classifier.load_state_dict(torch.load(CHECKPOINT,map_location="cpu",weights_only=True),strict=True)
    anchor=AnchorGeneratorWithAnchorShape(feature_map_scales=[1,2,4],base_anchor_shapes=[[6,8,4],[8,6,5],[10,10,6]])
    net=RetinaNetDetector(network=classifier,anchor_generator=anchor,spatial_dims=3,num_classes=1,
                         size_divisible=[16,16,8],debug=False).cuda().eval()
    net.set_target_keys(box_key="box",label_key="label")
    net.set_box_selector_parameters(score_thresh=.02,topk_candidates_per_level=1000,nms_thresh=.22,detections_per_img=300)
    net.set_sliding_window_inferer(roi_size=PATCH,overlap=.25,sw_batch_size=1,mode="constant",sw_device="cuda",device="cpu")
    return net


@torch.inference_mode()
def infer(net,ct,affine):
    from monai.data import MetaTensor
    from monai.transforms import Spacing

    vol=MetaTensor(torch.from_numpy(ct[None].astype(np.float32)),affine=affine)
    pix=Spacing(pixdim=SPACING,mode="bilinear",padding_mode="border")(vol)
    image=torch.clamp((pix.as_tensor()+1024.)/1324.,0.,1.)
    with torch.autocast("cuda",dtype=torch.float16):
        pred=net([image],use_inferer=True)[0]
    edges=pred["box"].detach().cpu().numpy()
    confidence=pred["label_scores"].detach().cpu().numpy()
    mapping=np.linalg.inv(affine)@pix.affine.detach().cpu().numpy()
    centers=nib.affines.apply_affine(mapping,.5*(edges[:,:3]+edges[:,3:]))
    rank=np.argsort(confidence)[::-1][:50]
    return [(centers[i].tolist(),float(confidence[i])) for i in rank]


def run(shard,shards,limit,role):
    OUT.mkdir(exist_ok=True,parents=True)
    if role=="test":
        choice=read(DEST/"validation_selection.json")["chosen"]
        assert choice in ("external_matched","external_residual")
        locked=read(DEST/choice/"external_lock.json")
        wanted={r["code"]:r["sha"] for r in read(COHORT)["test"]}
        assert locked["reference"]==wanted and locked["external_detector"]==DIGEST
    net=detector()
    cases=read(COHORT)[role][shard::shards]
    if limit:cases=cases[:limit]
    for row in cases:
        code=row["code"]
        path=OUT/(code+".json")
        if path.exists():
            assert read(path)["source"]==row["sha"]
            continue
        assert sha(Path(row["path"]))==row["sha"]
        with np.load(row["path"]) as f:
            ct=f["ct"];aff=f["affine"]
        pred=infer(net,ct,aff)
        score={}
        local=[]
        if role=="dev":
            with np.load(row["path"]) as f:
                labels=f["mask"]
            tree=cKDTree(np.argwhere(labels)*1.5)
            local=[float(tree.query(np.asarray(pt)*1.5)[0]) for pt,p in pred]
            score={str(k):any(x<=6 for x in local[:k]) for k in (1,4,8,16,50)}
        write(path,dict(code=code,source=row["sha"],model=DIGEST,predictions=pred,distances=local,near=score,
                        disclosure="LUNA16 detection-box supervision external to study; use only a frozen shared localization prior."))
        print(code,len(pred),score,flush=True)


if __name__=="__main__":
    os.umask(0o077)
    a=ArgumentParser()
    a.add_argument("--shard",type=int,default=0)
    a.add_argument("--shards",type=int,default=1)
    a.add_argument("--limit",type=int,default=0)
    a.add_argument("--role",choices=("dev","train","unlabelled","test"),default="dev")
    p=a.parse_args()
    run(p.shard,p.shards,p.limit,p.role)
