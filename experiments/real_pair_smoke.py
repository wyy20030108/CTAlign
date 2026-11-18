"""Four real reserved training patients: CT/report consistency-path smoke check, no labels."""
import argparse
import copy
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import re
import time

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import nibabel as nib
import numpy as np
import torch
from torch.nn import functional as F

from audit_pairs import identity
from fetch_four import fetch, sha256, write_json
from fetch_manifest import FILES
from volume_benchmark import ReportUNet


def text_ids(sections):
    ids = []
    words_total = 0
    for marker, f in enumerate(('Findings_EN','Impressions_EN'), 1):
        words = re.findall(r"[A-Za-z]+|[0-9]+(?:\.[0-9]+)?|[^\w\s]", sections[f].lower())[:31]
        payload = [4 + int.from_bytes(hashlib.blake2b(w.encode(),digest_size=4).digest(),'big') % 508
                   for w in words]
        ids.extend([marker]+payload+[0]*(31-len(payload)))
        words_total += len(words)
    if not words_total:
        raise ValueError('An audited joined report has no usable tokens')
    return ids, words_total


def metadata_for(names,root):
    values = {}
    excluded = set()
    for tag, split in (('train','train'),('validation','valid')):
        path = root/f'{tag}_metadata.csv'
        with path.open(encoding='utf-8-sig',newline='') as s:
            for row in csv.DictReader(s):
                key = row['VolumeName'].strip()
                if key in names:
                    pair = float(row['RescaleSlope']), float(row['RescaleIntercept'])
                    if (not all(map(math.isfinite,pair)) or pair[0]<=0
                            or (key in values and values[key]!=pair)):
                        raise ValueError('Conflicting or invalid original-series intensity metadata')
                    values[key]=pair
        excluded.update(Path(line.strip()).name for line in (root/f'no_chest_{split}.txt').read_text().splitlines())
    if names != set(values) or names & excluded:
        raise ValueError('Selected patients lack intensity metadata or include a listed non-chest acquisition')
    return values


def read_cube(path,record,size,slope,intercept):
    if sha256(path)!=record['ct_sha256']:
        raise ValueError('CT file hash does not match published pinned file recorded at download')
    img=nib.load(path)
    if len(img.shape)!=3 or min(img.shape)<size:
        raise ValueError('Inadequate dimensions for the specified native-grid crop')
    if img.dataobj.slope != 1 or img.dataobj.inter != 0:
        raise ValueError('Original v1 identity-header scaling expected. Avoid double intensity correction.')
    start=[(n-size)//2 for n in img.shape]
    sl=tuple(slice(i,i+size) for i in start)
    arr=np.asarray(img.dataobj[sl],dtype=np.float32)*slope+intercept
    if not np.isfinite(arr).all() or float(arr.max()-arr.min())==0:
        raise ValueError('Degenerate native-grid crop')
    # Official v1 raw * RescaleSlope + RescaleIntercept, followed by fixed engineering clipping.
    arr=np.clip((arr+1024.)/2048.,0,1)
    if arr.max()==arr.min():
        raise ValueError('All values saturated after fixed clipping')
    return torch.from_numpy(np.ascontiguousarray(arr[None,None]))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',type=Path,default=Path('data/four_train'))
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--seed',type=int,default=891)
    parser.add_argument('--patch',type=int,default=96)
    parser.add_argument('--fetch-metadata',action='store_true',
                        help='Fetch the four pinned official v1 calibration and non-chest index files first.')
    a=parser.parse_args()
    if a.patch%8 or a.patch<16:parser.error('patch must be a multiple of 8 and at least 16')
    os.umask(0o077)
    audit=json.loads((a.data/'four_header_audit.json').read_text())
    lock=json.loads((a.data/'pilot_T_reservation.json').read_text())
    reports_path=a.data/'four_reports.private.json'
    reports=json.loads(reports_path.read_text())
    rows={r['name']:r for r in audit['cases']}
    names=lock['volumes']
    if (lock['role']!='T_paid_engineering_pilot' or len(set(names))!=4 or len(names)!=4
            or set(names)!=set(rows) or set(rows)!=set(reports)
            or len({identity(n)[1] for n in names})!=4
            or sha256(reports_path)!=audit['report_subset_sha256']
            or audit['repository_revisions']!={repo:rev for repo,rev,_ in FILES.values()}):
        raise ValueError('Input is not the pinned four-patient training reservation')
    if a.fetch_metadata:
        for filename in ('train_metadata.csv','validation_metadata.csv','no_chest_train.txt','no_chest_valid.txt'):
            fetch('ct_rate','dataset/metadata/'+filename,a.data)
    params=metadata_for(set(names),a.data/'ct_rate/dataset/metadata')
    note=a.data/'ct_rate/dataset/data_correction_note.md'
    if sha256(note)!=audit['correction_note_sha256']:
        raise ValueError('Missing pinned correction note')
    torch.manual_seed(a.seed)
    # PyTorch 2.8 CUDA does not certify deterministic max-pool3d backward.
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cuda.matmul.allow_tf32=False
    net=ReportUNet().cuda().train()
    teacher=copy.deepcopy(net).eval().requires_grad_(False)
    opt=torch.optim.Adam(net.parameters(),lr=1e-4)
    measures=[]
    # Use the same trained graph as the GPU benchmark, now with deterministic real text and native image crops.
    for name in names:
        _,(split,num),(_,_,exam)=identity(name)
        expected=f'dataset/{split}/{split}_{num}/{split}_{num}_{exam}/{name}'
        record=rows[name]
        if record['ct_release_path']!=expected or record['report_key']!=name:
            raise ValueError('Volume/report join or pinned v1 path changed')
        x=read_cube(a.data/'ct_rate'/expected,record,a.patch,*params[name]).cuda()
        ids,n_words=text_ids(reports[name]); text=torch.tensor([ids],device='cuda')
        empty=torch.zeros_like(text); no_words=torch.where(text>=4,3,text)
        torch.cuda.synchronize(); t=time.perf_counter()
        with torch.no_grad():
            q=teacher(x,text).sigmoid()
            q0=teacher(x,empty).sigmoid()
            qlex=teacher(x,no_words).sigmoid()
        strong=(x+0.02*torch.randn_like(x)).clamp(0,1)
        old=net.output.weight.detach().clone()
        opt.zero_grad(set_to_none=True)
        logits=net(strong,empty)
        loss=F.binary_cross_entropy_with_logits(logits,q)
        loss.backward()
        grad=net.output.weight.grad.abs().max().item()
        opt.step()
        with torch.no_grad():
            for ema,online in zip(teacher.parameters(),net.parameters()):
                ema.lerp_(online,0.01)
        torch.cuda.synchronize()
        row={'tokens_used':n_words,'loss':loss.item(),'teacher_real_vs_empty_mae':(q-q0).abs().mean().item(),
             'teacher_real_vs_same_layout_lexical_ablation_mae':(q-qlex).abs().mean().item(),
             'output_weight_gradient_max_abs':grad,'output_weight_update_max_abs':(net.output.weight-old).abs().max().item(),
             'compute_seconds':time.perf_counter()-t}
        if not all(math.isfinite(v) and v>0 for v in row.values()):
            raise RuntimeError('One of the real update checks failed')
        measures.append(row)
    summary={'status':'pass','scope':'Engineering check using four reserved training CT/report pairs; randomly initialized teacher/student. No mask values, effusion supervision, performance claims, matched-donor intervention or holdout usage.',
             'seed':a.seed,'gpu':torch.cuda.get_device_name(0),'torch':str(torch.__version__),'patch':[a.patch]*3,
             'input':'native-grid center crop, pinned CT-RATE v1 raw * per-volume slope + intercept, scale [-1024,1024] to [0,1]; no reorientation or mask geometry inference',
             'text':'English Findings_EN and Impressions_EN, first 31 regex tokens per section, BLAKE2b to 508 buckets, two section markers and padding. Real, all-zero and lexical-content-neutralized teacher conditions; student always all-zero.',
             'optimizer':'Adam, lr 0.0001, four sequential FP32 consistency updates, EMA decay 0.99',
             'determinism':'Fixed crop, seed and token mapping; deterministic algorithms in warn-only mode for CUDA max-pool3d backward.',
             'reservation_sha256':sha256(a.data/'pilot_T_reservation.json'),
             'unique_train_patients':4,'real_report_pairs':4,'completed_optimizer_steps':len(measures),
             'peak_allocated_mib':torch.cuda.max_memory_allocated()/2**20,
             'range_across_four_steps':{k:[min(r[k] for r in measures),max(r[k] for r in measures)] for k in measures[0]}}
    write_json(a.output,summary)
    print(json.dumps(summary,indent=2))


if __name__=='__main__':
    main()
