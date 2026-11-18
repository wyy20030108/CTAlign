"""Three matched training pilots on engineering-reviewed T foreground and image-only U derivatives."""
import argparse
import copy
import csv
import hashlib
import json
import os
from pathlib import Path
import time

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import numpy as np
import torch
from torch.nn import functional as F
from audit_pairs import identity
from fetch_four import sha256, write_json
from real_pair_smoke import text_ids
from replay_controls import fingerprint, read_reports
from volume_benchmark import ReportUNet


ARMS = ('supervised', 'image_ema', 'report_ema')
BASE = Path('phase1/data')
SPLIT = Path('phase1/cohort/split_v1/b10')


def digest(obj):
    data = json.dumps(obj, sort_keys=True, separators=(',', ':')).encode()
    return hashlib.sha256(data).hexdigest()


def read(path):
    return json.loads(path.read_text(encoding='utf8'))


def allowed(split):
    with split.open(newline='') as handle:
        return {row['volume'] for row in csv.DictReader(handle)}


def load_cases(qc, size):
    release = read(qc)
    tnames = sorted(release['usable'])
    excluded = set(release['excluded'])
    t_dir = BASE/'T_partial'
    if (not tnames or set(tnames) & excluded or len(set(tnames)) != len(tnames)
            or not set(tnames) <= allowed(SPLIT/'T.csv')):
        raise ValueError('QC release is invalid for this T whitelist')
    t = [read(t_dir/(n.removesuffix('.nii.gz')+'.json')) for n in tnames]
    u_manifest = BASE/'U/manifest.private.json'
    u = sorted(read(u_manifest)['rows'], key=lambda row: row['name'])
    unames = [row['name'] for row in u]
    if len(u) != 8 or not set(unames) <= allowed(SPLIT/'U_train.csv'):
        raise ValueError('Eight U patients with prespecified membership expected')
    if len({identity(n)[1] for n in tnames+unames}) != len(t)+len(u):
        raise ValueError('Inconsistent patient-level separation')
    tokens = {n: torch.tensor([text_ids(report)[0]], device='cuda')
              for n, report in read_reports(set(tnames+unames)).items()}
    roles, provenance = {}, {}
    for role, rows, names, directory in [('T',t,tnames,t_dir), ('U',u,unames,BASE/'U')]:
        cases = []
        hashes = []
        for name, row in zip(names, rows, strict=True):
            file = directory/(name.removesuffix('.nii.gz')+'.npz')
            if row['name'] != name or file.resolve() != Path(row['file']).resolve() or sha256(file) != row['npz_sha256']:
                raise ValueError('Input provenance mismatch')
            if role == 'T' and release['cases'][name]['npz_sha256'] != row['npz_sha256']:
                raise ValueError('QC release does not match derivative')
            with np.load(file, allow_pickle=False) as pack:
                schema = {'ct_hu','affine'} | ({'positive_mask'} if role == 'T' else set())
                if set(pack.files) != schema:
                    raise ValueError('Unexpected arrays for assigned role')
                affine, ct = pack['affine'], pack['ct_hu']
                if (ct.dtype != np.int16 or ct.ndim != 3 or min(ct.shape) < size
                        or not np.isfinite(affine).all() or not np.allclose(affine[:3,:3], np.eye(3)*2, atol=1e-5)):
                    raise ValueError('Expected fixed 2-mm RAS HU CT')
                mask = pack['positive_mask'] if role=='T' else None
                anchors = None
                if mask is not None:
                    if mask.shape != ct.shape or mask.max() != 1 or int(mask.sum()) != row['positive_voxels']:
                        raise ValueError('Invalid QC-reviewed partial foreground')
                    anchors = np.flatnonzero(mask)
            cases.append((ct,mask,anchors,tokens[name]))
            hashes.append({'volume':name,'npz_sha256':row['npz_sha256'],
                           'token_sha256':digest(tokens[name].cpu().tolist())})
        roles[role], provenance[role] = cases, hashes
    return roles, {'roles':provenance, 'qc_path':str(qc),'qc_sha256':sha256(qc),
                   'T_split_sha256':sha256(SPLIT/'T.csv'),'U_split_sha256':sha256(SPLIT/'U_train.csv'),
                   'U_manifest_sha256':sha256(u_manifest)}


def schedule(cases, steps, seed):
    rng = np.random.default_rng(seed)
    def indices(n):
        return np.concatenate([rng.permutation(n) for _ in range((steps+n-1)//n)])[:steps]
    ti, ui = indices(len(cases['T'])), indices(len(cases['U']))
    # Fix exactly half of the labelled steps in the report arm to the null report.
    drop = rng.permutation(steps) < steps//2
    return [{'T':int(ti[i]),'U':int(ui[i]),'crop_T':int(rng.integers(2**31)),
             'crop_U':int(rng.integers(2**31)), 'drop_T_text':bool(drop[i])} for i in range(steps)]


def tensors(case, seed, size, augment):
    ct, mask, anchors, tokens = case
    rng = np.random.default_rng(seed)
    shape = np.array(ct.shape)
    if mask is None:
        # Geometry only. No disease labels, model outputs or report used to place U crops.
        center = np.rint((shape-1)*rng.uniform(.3,.7,3)).astype(int)
    else:
        center = np.unravel_index(rng.choice(anchors), ct.shape)
    start = np.clip(np.asarray(center)-size//2, 0, shape-size)
    region = tuple(slice(x,x+size) for x in start)
    data = np.clip((ct[region].astype(np.float32)+1000)/1200,0,1)
    x = torch.from_numpy(data).cuda()[None,None]
    fg = torch.from_numpy(mask[region].astype(bool)).cuda()[None,None] if mask is not None else None
    if fg is not None and not bool(fg.any()):
        raise ValueError('Empty positive-anchored crop')
    if not augment:
        return x, fg, tokens, x
    offset = float(rng.uniform(-.04,.04))
    gain = float(rng.uniform(.90,1.10))
    # Shared voxel coordinates between weak and strong views; no left-right flips of reported side.
    noise = torch.from_numpy(rng.normal(0,.025,data.shape).astype(np.float32)).cuda()[None,None]
    strong = (x*gain+offset+noise).clamp(0,1)
    return strong, fg, tokens, x


@torch.inference_mode()
def probe(model, cases, seed, size, empty):
    model.eval()
    out = {}
    for role, rows in cases.items():
        probs, fracs, known_probs, known_fracs = [], [], [], []
        for i, row in enumerate(rows):
            x, fg, _, _ = tensors(row, seed+10000+i, size, False)
            p = model(x,empty).sigmoid()
            probs.append(p.mean().item())
            fracs.append((p>=.5).float().mean().item())
            if fg is not None:
                known_probs.append(p[fg].mean().item())
                known_fracs.append((p[fg]>=.5).float().mean().item())
        out[role] = {'crops':len(fracs),'mean_p':float(np.mean(probs)),
                     'mean_fraction_predicted_fg_at_0.5':float(np.mean(fracs)),
                     'crops_at_least_95pct_predicted_fg':sum(f>=.95 for f in fracs),
                     'crops_at_most_5pct_predicted_fg':sum(f<=.05 for f in fracs)}
        if known_probs:
            out[role].update(known_positive_mean_p=float(np.mean(known_probs)),
                             known_positive_fraction_at_0_5=float(np.mean(known_fracs)))
    model.train()
    return out


def train_arm(arm, init, cases, steps, args, common):
    student = copy.deepcopy(init).cuda().train()
    teacher = copy.deepcopy(student).eval().requires_grad_(False) if arm!='supervised' else None
    optimizer = torch.optim.Adam(student.parameters(),lr=1e-4)
    empty = torch.zeros_like(cases['T'][0][3])
    begin = probe(student,cases,args.seed,args.patch,empty)
    rows = []
    torch.cuda.reset_peak_memory_stats()
    for item in steps:
        torch.cuda.synchronize(); tick = time.perf_counter()
        x, fg, real_T, _ = tensors(cases['T'][item['T']],item['crop_T'],args.patch,True)
        t_tokens = real_T if arm=='report_ema' and not item['drop_T_text'] else empty
        optimizer.zero_grad(set_to_none=True)
        pos = F.softplus(-student(x,t_tokens)[fg]).mean()
        cons = None
        if teacher is not None:
            strong, _, real_U, weak = tensors(cases['U'][item['U']],item['crop_U'],args.patch,True)
            with torch.no_grad():
                target = teacher(weak,real_U if arm=='report_ema' else empty).sigmoid()
            cons = F.mse_loss(student(strong,empty).sigmoid(),target)
        loss = pos if cons is None else pos+cons
        if not torch.isfinite(loss):
            raise ValueError('Nonfinite training objective')
        loss.backward()
        grad = torch.nn.utils.clip_grad_norm_(student.parameters(), float('inf'), error_if_nonfinite=True)
        optimizer.step()
        if teacher is not None:
            with torch.no_grad():
                for avg, online in zip(teacher.parameters(),student.parameters(),strict=True):
                    avg.lerp_(online,.01)
        torch.cuda.synchronize()
        rows.append({'step':len(rows)+1,'known_positive_bce':pos.item(),
                     'U_consistency_mse':cons.item() if cons is not None else None,
                     'total_loss':loss.item(),'gradient_l2':grad.item(),'seconds':time.perf_counter()-tick})
    peak = torch.cuda.max_memory_allocated()/2**30
    end = probe(student,cases,args.seed,args.patch,empty)
    fprint = fingerprint(student.state_dict())
    if fprint == common['initial_weights_sha256']:
        raise ValueError('No student parameter change')
    saved = args.output/(arm+'.pt')
    torch.save({'student':student.state_dict(),'teacher':teacher.state_dict() if teacher is not None else None,
                'optimizer':optimizer.state_dict(),'arm':arm,'common':common,'steps':len(rows)},saved)
    metrics = {'arm':arm,'steps':len(rows),'start_train_probe':begin,'end_train_probe':end,
               'student_final_weights_sha256':fprint,'checkpoint_sha256':sha256(saved),
               'GPU_peak_allocated_GiB':peak,'training_wall_seconds':sum(r['seconds'] for r in rows),
               'median_training_step_seconds':float(np.median([r['seconds'] for r in rows])),
               'positive_bce_first5':float(np.mean([r['known_positive_bce'] for r in rows[:5]])),
               'positive_bce_last5':float(np.mean([r['known_positive_bce'] for r in rows[-5:]])),
               'U_consistency_mse_mean':float(np.mean([r['U_consistency_mse'] for r in rows])) if teacher is not None else None,
               'labelled_true_report_steps':sum(not s['drop_T_text'] for s in steps) if arm=='report_ema' else 0,
               'unlabelled_teacher_true_report_steps':len(steps) if arm=='report_ema' else 0,
               'unlabelled_student_true_report_steps':0}
    write_json(args.output/(arm+'_steps.json'),rows)
    write_json(args.output/(arm+'_metrics.json'),metrics)
    print('completed',arm,'steps',len(rows),'sec',round(metrics['training_wall_seconds'],1),flush=True)
    return metrics


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--qc',type=Path,required=True)
    p.add_argument('--output',type=Path,default=Path('phase1/baselines/partial_v1'))
    p.add_argument('--seed',type=int,default=891)
    p.add_argument('--steps',type=int,default=32)
    p.add_argument('--patch',type=int,default=64)
    a=p.parse_args()
    if a.steps<8 or a.patch<16 or a.patch%8:
        p.error('steps >= 8 and patch a multiple of 8, >=16')
    os.umask(0o077)
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '0':
        raise ValueError('Reserve GPU 1; set CUDA_VISIBLE_DEVICES=0')
    if a.qc.resolve().parent != (BASE/'T_partial').resolve() or a.qc.name not in ('qc_release.private.json','provisional_qc.private.json'):
        raise ValueError('Use an independent engineering QC release for T')
    a.output.mkdir(parents=True,exist_ok=True)
    if any(a.output.iterdir()):
        raise FileExistsError('Provide a fresh output directory')
    torch.manual_seed(a.seed)
    torch.set_num_threads(2)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cuda.matmul.allow_tf32=False
    cases, provenance = load_cases(a.qc,a.patch)
    if a.steps<max(map(len,cases.values())):
        raise ValueError('All frozen participants must be exposed at least once per trained arm')
    steps = schedule(cases,a.steps,a.seed)
    init = ReportUNet().cpu()
    common = {'seed':a.seed,'patch_size':a.patch,'steps':a.steps,'initial_weights_sha256':fingerprint(init.state_dict()),
              'inputs_sha256':digest(provenance),'schedule_sha256':digest(steps),'T':len(cases['T']),'U':len(cases['U']),
              'script_sha256':{f:sha256(Path(f)) for f in ('train_partial_baselines.py','volume_benchmark.py','replay_controls.py','real_pair_smoke.py','audit_pairs.py','fetch_four.py','fetch_manifest.py')},
              'torch':str(torch.__version__),'cuda':torch.version.cuda,'GPU':torch.cuda.get_device_name(0),
              'supervision':'T published foreground only, softplus(-logit), mean across known-positive voxels; outside is unknown',
              'consistency':'EMA=0.99, U weak own CT/prespecified text, strong CT/null student, probability MSE, coefficient 1',
              'image':'fixed CT 2 mm RAS+, clip [-1000,200] HU and scale to [0,1]; paired training views retain spatial coordinates',
              'probe':'one fixed training crop per patient pooled as patient mean; T centered on known finding, U geometry-only; no external or independent outcomes',
              'text':'two hash token blocks of 32 positions, max 31 tokens each for findings/impression; constant all-zero learned null input; no pretrained text encoder',
              'numeric':'float32, same CPU initialization and frozen schedule, deterministic cuDNN, TF32 off; no assertion of bitwise cross-platform reproducibility'}
    write_json(a.output/'input_lock.private.json',provenance)
    write_json(a.output/'schedule.private.json',steps)
    write_json(a.output/'run_config.json',common)
    metrics = {arm:train_arm(arm,init,cases,steps,a,common) for arm in ARMS}
    if (len({digest(r['start_train_probe']) for r in metrics.values()}) != 1 or
            len({r['student_final_weights_sha256'] for r in metrics.values()}) != len(ARMS)):
        raise ValueError('Failed common start or distinct trained weight verification')
    write_json(a.output/'summary.json',{'common':common,'arms':metrics,
        'limitations':'No vetted normal cases, dense negatives, or independent clinical segmentation references; no clinical Dice or efficacy estimate. Extreme predicted fractions are collapse diagnostics on crops.'})


if __name__=='__main__':
    main()
