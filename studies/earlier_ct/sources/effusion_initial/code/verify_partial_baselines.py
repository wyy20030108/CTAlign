"""Reload all three trained checkpoints and reproduce training-crop diagnostics on frozen inputs."""
import argparse
import json
import math
import os
from pathlib import Path

from train_partial_baselines import (ARMS, ReportUNet, digest, fingerprint, load_cases, probe,
                                     read, schedule, sha256, torch, write_json)


def close_metrics(a,b):
    return a.keys()==b.keys() and all(
        close_metrics(a[k],b[k]) if isinstance(a[k],dict) else math.isclose(a[k],b[k],abs_tol=1e-6,rel_tol=0)
        for k in a)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('run',type=Path)
    folder=p.parse_args().run
    os.umask(0o077)
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='0':
        raise ValueError('Use GPU 0')
    common=read(folder/'run_config.json'); lock=read(folder/'input_lock.private.json')
    snapshot=folder/'QC_snapshot.private.json'
    if not snapshot.exists():
        original=Path(lock['qc_path'])
        if sha256(original)!=lock['qc_sha256']:
            raise ValueError('Changed QC release without frozen copy')
        snapshot.write_bytes(original.read_bytes())
    if sha256(snapshot)!=lock['qc_sha256'] or digest(lock)!=common['inputs_sha256']:
        raise ValueError('Recorded train inputs or QC snapshot changed')
    cases, restored=load_cases(snapshot,common['patch_size']); restored['qc_path']=lock['qc_path']
    steps=read(folder/'schedule.private.json')
    if (lock!=restored or digest(steps)!=common['schedule_sha256'] or
        steps!=schedule(cases,common['steps'],common['seed']) or
        any(sha256(Path(f))!=h for f,h in common['script_sha256'].items())):
        raise ValueError('Changed arrays, reports, code or step schedule')
    torch.manual_seed(common['seed']); torch.set_num_threads(2)
    torch.backends.cudnn.benchmark=False; torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.allow_tf32=False; torch.backends.cuda.matmul.allow_tf32=False
    m=ReportUNet().cuda(); empty=torch.zeros_like(cases['T'][0][3])
    if fingerprint(m.state_dict())!=common['initial_weights_sha256']:
        raise ValueError('Initialization did not reproduce')
    init=probe(m,cases,common['seed'],common['patch_size'],empty)
    checks={}
    for arm in ARMS:
        saved=folder/(arm+'.pt'); metrics=read(folder/(arm+'_metrics.json')); rows=read(folder/(arm+'_steps.json'))
        ck=torch.load(saved,map_location='cpu',weights_only=True)
        if (sha256(saved)!=metrics['checkpoint_sha256'] or ck['arm']!=arm or ck['common']!=common or
            ck['steps']!=common['steps'] or not close_metrics(init,metrics['start_train_probe']) or
            fingerprint(ck['student'])!=metrics['student_final_weights_sha256'] or
            any(t['step'].item()!=common['steps'] for t in ck['optimizer']['state'].values()) or
            not ck['optimizer']['state'] or (ck['teacher'] is None)!=(arm=='supervised')):
            raise ValueError('Checkpoint or optimizer check failed')
        if ([row['step'] for row in rows]!=list(range(1,common['steps']+1)) or
            not all(row['gradient_l2']>0 and math.isfinite(row['total_loss']) for row in rows)):
            raise ValueError('Training step verification failed')
        m.load_state_dict(ck['student'],strict=True)
        end=probe(m,cases,common['seed'],common['patch_size'],empty)
        if not close_metrics(end,metrics['end_train_probe']):
            raise ValueError('Final training-crop probe failed to reproduce')
        checks[arm]={'checkpoint_hash_ok':True,'optimizer_steps':common['steps'],
                     'reloaded_probe_agrees_absolute_tolerance':1e-6,
                     'nonzero_finite_gradients':len(rows), 'EMA_state_verified':ck['teacher'] is not None}
        if ck['teacher'] is not None:
            m.load_state_dict(ck['teacher'],strict=True)
            if any(not torch.isfinite(v).all() for v in ck['teacher'].values()):
                raise ValueError('Nonfinite EMA state')
    out={'status':'passed','inputs_rebuilt_sha256':digest(restored),'schedule_rebuilt_sha256':digest(steps),
         'initial_model_rebuilt_sha256':common['initial_weights_sha256'], 'checks':checks,
         'verifier_sha256':sha256(Path(__file__))}
    write_json(folder/'verification.json',out)
    print(json.dumps(out))


if __name__=='__main__':
    main()
