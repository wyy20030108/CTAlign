"""Final matched two-crop, image-only EMA pilot with sparse CT hints switched off/on."""
import argparse
import copy
import hashlib
import os
from pathlib import Path
import time
import numpy as np
from sparse_hu_hints import RULE
from train_partial_baselines import (F, ReportUNet, digest, fingerprint, load_cases, probe, read,
                                     schedule, sha256, tensors, torch, write_json)
from verify_partial_baselines import close_metrics


ARMS=('weak_off','weak_on')
RULES={'backbone':'image-only EMA=.99, U probability MSE=1; both branches always get fixed null text',
       'update':'one original positive-anchored T crop + one additional T crop from same patient in a joint size-two forward; one original U weak/strong pair',
       'second_center':'within patient alternate available cue groups across visits from a seeded starting group; visit points in a fixed permutation without replacement until exhausted. If no cue anywhere, use volume FOV centre.',
       'strong_augmentation':'same clipping and gain, offset and noise bounds as original T crop, independent predetermined second-crop seed; no spatial flip',
       'weak_off':'if secondary crop contains hints: 0.5*positive_BCE, otherwise positive_BCE. Always add U consistency.',
       'weak_on':'if secondary crop contains hints: 0.5*positive_BCE+0.5*weak_negative_BCE, otherwise positive_BCE. Always add U consistency.',
       'unknown':'T positive term uses only original known foreground in original crop. Weak term uses only previously frozen selected coordinates in second crop. All other locations remain unknown.',
       'interpretation':'both arms intentionally pay for two T patches; the positive coefficient remains identical across arms to isolate toggling the weak loss.'}


def stable_seed(key):
    return int.from_bytes(hashlib.sha256(key.encode()).digest()[:8],'big')


def tensor_hash(*items):
    h=hashlib.sha256()
    for val in items:
        arr=val.detach().cpu().contiguous().numpy()
        h.update(str((arr.shape,arr.dtype)).encode()); h.update(arr.tobytes())
    return h.hexdigest()


def load_locked(ref, cues):
    common=read(ref/'run_config.json'); lock=read(ref/'input_lock.private.json'); cueconfig=read(cues/'run_config.json')
    if (read(ref/'verification.json')['status']!='passed' or
        cueconfig['reference_summary_sha256']!=sha256(ref/'summary.json') or
        cueconfig['hint_rules_sha256']!=digest(RULE) or
        any(sha256(Path(f))!=h for f,h in {**common['script_sha256'],**cueconfig['hint_scripts_sha256']}.items())):
        raise ValueError('Locked baseline, cue extraction, or dependencies changed')
    cases, reconstructed=load_cases(ref/'QC_snapshot.private.json',common['patch_size']); reconstructed['qc_path']=lock['qc_path']
    steps=read(ref/'schedule.private.json')
    if ((common['T'],common['U'],common['steps'])!=(6,8,32) or reconstructed!=lock or
        digest(lock)!=common['inputs_sha256'] or steps!=schedule(cases,32,common['seed']) or digest(steps)!=common['schedule_sha256']):
        raise ValueError('Original T/U identities, data or sample draw differs')
    records=read(cues/'seed_counts.private.json')
    if digest(records)!=cueconfig['hint_audit_sha256']:
        raise ValueError('Cue case records changed')
    groups=[]
    for c,row,record in zip(cases['T'],lock['roles']['T'],records,strict=True):
        n=row['volume']; src=cues/'hints_private'/(n.removesuffix('.nii.gz')+'.npz')
        if (n!=record['volume'] or row['npz_sha256']!=record['T_derivative_sha256'] or
            sha256(src)!=record['hint_coordinates_sha256']):
            raise ValueError('Cue coordinates have changed')
        with np.load(src,allow_pickle=False) as v:
            hints={g:v[g].copy() for g in ('gas','dense')}
        for g,coords in hints.items():
            if (coords.shape!=(record[g]['selected'],3) or len(coords) and c[1][tuple(coords.T)].any()):
                raise ValueError('Invalid cue coordinates')
        groups.append(hints)
    return cases,steps,common,lock,groups,cueconfig


def second_crop(ct,groups,center,seed,size):
    start=np.clip(np.asarray(center)-size//2,0,np.asarray(ct.shape)-size)
    slices=tuple(slice(s,s+size) for s in start)
    base=np.clip((ct[slices].astype(np.float32)+1000)/1200,0,1)
    rng=np.random.default_rng(seed)
    offset=float(rng.uniform(-.04,.04)); gain=float(rng.uniform(.90,1.10))
    noise=torch.from_numpy(rng.normal(0,.025,base.shape).astype(np.float32)).cuda()[None,None]
    image=(torch.from_numpy(base).cuda()[None,None]*gain+offset+noise).clamp(0,1)
    mask=np.zeros(base.shape,dtype='uint8'); counts={}
    for i,(g,points) in enumerate(groups.items(),1):
        loc=points-start; valid=loc[np.all((loc>=0)&(loc<size),axis=1)]; counts[g]=len(valid)
        if len(valid):
            if mask[tuple(valid.T)].any():
                raise ValueError('Cue groups overlap')
            mask[tuple(valid.T)]=i
    return image,torch.from_numpy(mask>0).cuda()[None,None],counts,start.tolist()


def prepare(cases,steps,common,lock,groups):
    seed=common['seed']; size=common['patch_size']; participants=[]
    for row,hints in zip(lock['roles']['T'],groups,strict=True):
        name=row['volume']; available=[g for g in ('gas','dense') if len(hints[g])]
        r=np.random.default_rng(stable_seed(f'{seed}:{name}:cue-order'))
        order=[str(g) for g in r.permutation(available)]
        sequences={g:r.permutation(len(hints[g])) for g in order}
        participants.append({'volume':name,'groups':order,'sequence':sequences,'visits':0,'weak_steps':0,
                             'target_centers':{g:0 for g in available},'cue_exposures':{'gas':0,'dense':0}})
    batches=[]; ledger=[]
    for i,s in enumerate(steps):
        t=s['T']; v=participants[t]; case=cases['T'][t]; hints=groups[t]; order=v['groups']
        group=order[v['visits']%len(order)] if order else None
        if group is None:
            center=(np.array(case[0].shape)-1)//2
        else:
            seq=v['sequence'][group]; center=hints[group][seq[v['target_centers'][group]%len(seq)]]
            v['target_centers'][group]+=1
        v['visits']+=1
        sec_seed=stable_seed(f'{seed}:second-augmentation:{i}')
        x2,neg,counts,start=second_crop(case[0],hints,center,sec_seed,size)
        if bool(group) != bool(sum(counts.values())):
            raise ValueError('Secondary crop lost its target cue or used a cue outside planned patients')
        v['weak_steps']+=bool(group)
        for g in counts:
            v['cue_exposures'][g]+=counts[g]
        x1,fg,_,_=tensors(case,s['crop_T'],size,True)
        strong,_,_,weak=tensors(cases['U'][s['U']],s['crop_U'],size,True)
        x=torch.cat([x1,x2]); active=bool(group)
        batches.append((x,fg,neg,strong,weak,active))
        ledger.append({'step':i+1,'T':t,'U':s['U'],'center_group':group,'center':center.tolist(),'second_start':start,
                       'second_augmentation_seed':sec_seed,'cue_exposure_count':counts,
                       'all_actual_tensor_sha256':tensor_hash(x,fg,neg,strong,weak)})
    contrib=[{k:v[k] for k in ('volume','visits','weak_steps','groups','target_centers','cue_exposures')} for v in participants]
    return batches,ledger,contrib


def train(arm,init,cases,batches,common,out):
    model=copy.deepcopy(init).cuda().train(); teacher=copy.deepcopy(model).eval().requires_grad_(False)
    optimizer=torch.optim.Adam(model.parameters(),lr=1e-4); empty=torch.zeros_like(cases['T'][0][3])
    start=probe(model,cases,common['seed'],common['patch_size'],empty); rows=[]
    torch.cuda.reset_peak_memory_stats(); tick=time.perf_counter()
    for i,(x,fg,neg,strong,weak,active) in enumerate(batches):
        optimizer.zero_grad(set_to_none=True)
        logits=model(x,empty.expand(2,-1))
        pos=F.softplus(-logits[:1][fg]).mean()
        # Calculate the second score identically in both arms, then toggle only its gradient coefficient.
        neg_loss=F.softplus(logits[1:][neg]).mean() if active else logits[1:].sum()*0
        pos_weight=.5 if active else 1.
        neg_weight=.5 if active and arm=='weak_on' else 0.
        with torch.no_grad():
            pseudo=teacher(weak,empty).sigmoid()
        cons=F.mse_loss(model(strong,empty).sigmoid(),pseudo)
        objective=pos_weight*pos+neg_weight*neg_loss+cons
        if not torch.isfinite(objective):
            raise ValueError('Nonfinite training objective')
        objective.backward(); grad=torch.nn.utils.clip_grad_norm_(model.parameters(),float('inf'),error_if_nonfinite=True)
        if grad.item()<=0:
            raise ValueError('No gradient')
        optimizer.step()
        with torch.no_grad():
            for ema,online in zip(teacher.parameters(),model.parameters(),strict=True):
                ema.lerp_(online,.01)
        rows.append({'step':i+1,'positive_bce':pos.item(),'weak_bce_measured':neg_loss.item() if active else None,
                     'positive_weight':pos_weight,'weak_weight':neg_weight,'U_mse':cons.item(),
                     'total':objective.item(),'gradient_l2':grad.item()})
    torch.cuda.synchronize(); secs=time.perf_counter()-tick
    peak=torch.cuda.max_memory_allocated()/2**30; end=probe(model,cases,common['seed'],common['patch_size'],empty)
    path=out/(arm+'.pt')
    torch.save({'arm':arm,'common':common,'student':model.state_dict(),'teacher':teacher.state_dict(),
                'optimizer':optimizer.state_dict(),'steps':len(rows)},path)
    data={'arm':arm,'initial_probe':start,'end_probe':end,'steps':len(rows),'T_forward_blocks':2*len(rows),
          'U_forward_pairs':len(rows),'actual_batches_digest':common['actual_batches_digest'],
          'checkpoint_sha256':sha256(path),'student_sha256':fingerprint(model.state_dict()),
          'active_weak_steps':sum(r['weak_weight']>0 for r in rows),'steps_with_measured_weak_examples':sum(r['weak_bce_measured'] is not None for r in rows),
          'positive_weight_sum':sum(r['positive_weight'] for r in rows),'mean_U_consistency_mse':float(np.mean([r['U_mse'] for r in rows])),
          'wall_seconds':secs,'peak_GiB_includes_shared_gpu_batch_cache':peak}
    write_json(out/(arm+'_metrics.json'),data); write_json(out/(arm+'_steps.json'),rows)
    print('trained',arm,len(rows),flush=True)
    return data


def verify(out,init,cases,common,ledger,contrib):
    if (read(out/'schedule_two_crops.private.json')!=ledger or read(out/'contributions.private.json')!=contrib or
        read(out/'run_config.json')!=common):
        raise ValueError('Pairing or reconstruction failed')
    empty=torch.zeros_like(cases['T'][0][3]); net=copy.deepcopy(init).cuda()
    p0=probe(net,cases,common['seed'],common['patch_size'],empty); metrics={}; logs={}
    for arm in ARMS:
        path=out/(arm+'.pt'); info=read(out/(arm+'_metrics.json')); rows=read(out/(arm+'_steps.json'))
        ck=torch.load(path,weights_only=True,map_location='cpu')
        if (info['checkpoint_sha256']!=sha256(path) or fingerprint(ck['student'])!=info['student_sha256']
            or info['student_sha256']==common['initial_weights_sha256'] or not close_metrics(p0,info['initial_probe'])
            or ck['common']!=common or ck['steps']!=32 or ck['arm']!=arm or len(rows)!=32 or not ck['optimizer']['state']
            or any(v['step'].item()!=32 for v in ck['optimizer']['state'].values()) or
            any(not np.isfinite(r['gradient_l2']) or r['gradient_l2']<=0 for r in rows)):
            raise ValueError('Invalid training checkpoint')
        net.load_state_dict(ck['teacher'],strict=True)
        if any(not t.isfinite().all() for t in ck['teacher'].values()):
            raise ValueError('Invalid EMA state')
        net.load_state_dict(ck['student'],strict=True)
        if not close_metrics(probe(net,cases,common['seed'],common['patch_size'],empty),info['end_probe']):
            raise ValueError('Held-in diagnostic failed to reproduce')
        metrics[arm]=info;logs[arm]=rows
    if (metrics['weak_off']['student_sha256']==metrics['weak_on']['student_sha256'] or
        any(x['positive_weight']!=y['positive_weight'] or x['weak_weight']!=0 or
            y['weak_weight']!=(.5 if r['center_group'] else 0.) for x,y,r in zip(logs['weak_off'],logs['weak_on'],ledger,strict=True))):
        raise ValueError('Matched gradient coefficients failed')
    outdata={'status':'passed','32_actual_batches_rebuilt':True,'paired_byte_digest':common['actual_batches_digest'],
             'shared_initialization_rebuilt':True,'both_Adam_states_reloaded_at_step':32,
             'identical_positive_coefficients':True,'deployment_probe_reproduced_absolute_tolerance':1e-6}
    write_json(out/'verification.json',outdata)
    print('both arms independently reloaded and verified',flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reference',type=Path,default=Path('phase1/baselines/partial_v1'))
    p.add_argument('--cues',type=Path,default=Path('phase1/baselines/sparse_hu_v1'))
    p.add_argument('--output',type=Path,default=Path('phase1/baselines/matched_two_crop_v1'))
    p.add_argument('--verify-only',action='store_true')
    a=p.parse_args(); os.umask(0o077)
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='0':
        raise ValueError('Reserve GPU 1')
    if not a.verify_only:
        a.output.mkdir(parents=True,exist_ok=False); write_json(a.output/'prespecified_rules.json',RULES)
    cases,steps,original,lock,groups,cuemeta=load_locked(a.reference,a.cues)
    torch.set_num_threads(2); torch.manual_seed(original['seed'])
    torch.backends.cudnn.benchmark=False; torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
    init=ReportUNet().cpu()
    if fingerprint(init.state_dict())!=original['initial_weights_sha256']:
        raise ValueError('Initialization changed')
    batches,ledger,contrib=prepare(cases,steps,original,lock,groups)
    common={k:original[k] for k in ('seed','patch_size','steps','T','U','initial_weights_sha256','inputs_sha256','schedule_sha256','GPU','torch','cuda')}
    common.update(rules_sha256=digest(RULES),actual_batches_digest=digest(ledger),T_contribution_digest=digest(contrib),
                  locked_cue_digest=cuemeta['hint_audit_sha256'],code_sha256=sha256(Path(__file__)),
                  dependency_code_sha256={**original['script_sha256'],**cuemeta['hint_scripts_sha256']},
                  reference_summary_sha256=sha256(a.reference/'summary.json'),cue_summary_sha256=sha256(a.cues/'summary.json'))
    if a.verify_only:
        return verify(a.output,init,cases,common,ledger,contrib)
    write_json(a.output/'schedule_two_crops.private.json',ledger); write_json(a.output/'contributions.private.json',contrib)
    write_json(a.output/'run_config.json',common)
    metrics={arm:train(arm,init,cases,batches,common,a.output) for arm in ARMS}
    summary={'T_with_selected_hints':sum(bool(r['groups']) for r in contrib),'T_supplying_hints_in_crops':sum(r['weak_steps']>0 for r in contrib),
             'T_with_geometry_fallback':sum(not r['groups'] for r in contrib),'steps_with_hint_location':sum(r['weak_steps'] for r in contrib),
             'center_by_group':{g:sum(r['target_centers'].get(g,0) for r in contrib) for g in ('gas','dense')},
             'point_exposures_by_group':{g:sum(r['cue_exposures'][g] for r in contrib) for g in ('gas','dense')},
             'T_cases_with_crop_exposures_by_group':{g:sum(r['cue_exposures'][g]>0 for r in contrib) for g in ('gas','dense')}}
    write_json(a.output/'summary.json',{'common':common,'rules':RULES,'coverage':summary,'arms':metrics,
               'limitations':'8 held-in U geometry-based crops; no clinical truth on any U crop and no certified whole-volume reference. A difference in foreground fraction cannot establish segmentation accuracy.'})
    print('completed matched arms; ready for independent verification')


if __name__=='__main__':
    main()
