"""Create one prospective, nested development allocation without opening any masks.

Expected project layout is the downloaded CT-RATE/ReX pilot. Patient identifiers
and report-derived strata remain on the server. Only summary.json is distributable.
"""
import argparse
import csv
import hashlib
import json
import os
import shutil
import tempfile
from collections import Counter, defaultdict
from fractions import Fraction
from pathlib import Path

from audit_pairs import identity, run as audit_join, split_names
from report_strata import describe_report


ROLES = ('T', 'R1', 'R2')
BUDGETS = (5, 10, 20)
STRATA = ('affirmed', 'explicitly_negated', 'uncertain_or_unmentioned')


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def rank(seed, tag, key):
    return hashlib.sha256(f'ctreport-phase1-v1|{seed}|{tag}|{key}'.encode()).hexdigest()


def csvfile(path, fields, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields, lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def jsonfile(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, sort_keys=True, indent=2)+'\n',encoding='utf-8')


def sizes(n):
    target = [n*k//10 for k in (7,2,1)]
    for i in sorted(range(3), key=lambda i: (-(n*(7,2,1)[i]%10),i))[:n-sum(target)]:
        target[i] += 1
    return dict(zip(ROLES,target))


def build(root, dest, seed):
    meta = root/'metadata'
    base,challenge = (meta/'rex'/s for s in ['dataset.json','MICCAI_challenge_dataset.json'])
    reports = [meta/'ct_rate'/'dataset'/'radiology_text_reports'/s for s in ['train_reports.csv','validation_reports.csv']]
    reservation = root/'data'/'four_train'/'pilot_T_reservation.json'
    matched = []
    audit = audit_join(base,reports,[challenge],matched_names=matched)
    assert audit == json.loads((root/'logs'/'pair_audit.json').read_bytes()), 'Released metadata differ from the previous audit'
    grouped=defaultdict(list)
    for name in matched:
        grouped['_'.join(identity(name)[1])].append(name)
    assert (len(matched),len(grouped)) == (2987,2712)
    pinned_raw = json.loads(reservation.read_bytes())
    pinned = {'_'.join(identity(v)[1]):v for v in pinned_raw['volumes']}
    assert set(pinned)==set(pinned_raw['patients']) and len(pinned)==4
    assert all(v in grouped[p] for p,v in pinned.items())
    chosen={p:pinned.get(p) or min(vols,key=lambda v:rank(seed,'one-volume',v)) for p,vols in grouped.items()}
    selected_names=set(chosen.values())
    complete={}
    for report_csv in reports:
        with report_csv.open(encoding='utf-8-sig',newline='') as f:
            for row in csv.DictReader(f):
                v=row['VolumeName'].strip()
                if v not in selected_names:
                    continue
                desc=describe_report(row['Findings_EN'],row['Impressions_EN'])
                assert desc['stratum'] in STRATA and v not in complete
                complete[v]=(desc['stratum'],desc['state'])
    assert selected_names==complete.keys()
    excluded=set()
    for path in (base,challenge):
        for split,names in split_names(path).items():
            if split in ('test','val'):
                excluded.update('_'.join(identity(v)[1]) for v in names)
    assert not excluded.intersection(chosen)
    stratum={p:complete[v][0] for p,v in chosen.items()}
    counts=Counter(stratum.values())
    pools={s:sorted([p for p in chosen if stratum[p]==s and p not in pinned],
                    key=lambda p:rank(seed,'stratified-patient-priority',p),reverse=True) for s in STRATA}
    available={s:len(ids) for s,ids in pools.items()}
    support=[s for s in STRATA if counts[s]]
    # Each stratum has 75% proportional mass and a 25% equal-allocation component.
    weights={s:Fraction(3*counts[s],4*len(chosen))+Fraction(1,4*len(support)) for s in support}
    assigned={'T':set(pinned),'R1':set(),'R2':set()}
    rep={r:Counter(stratum[p] for p in ids) for r,ids in assigned.items()}
    snapshots={}
    for pct in BUDGETS:
        target=sizes(len(chosen)*pct//100)
        for r in ROLES:
            while len(assigned[r])<target[r]:
                n=len(assigned[r])+1
                s=min((s for s in support if pools[s]),key=lambda s:(-(n*weights[s]-rep[r][s]),s))
                p=pools[s].pop()
                assigned[r].add(p)
                rep[r][s]+=1
        snapshots[pct]={r:set(assigned[r]) for r in ROLES}
    maximum=snapshots[20]
    outside=set(chosen)-set.union(*maximum.values())
    candidate_audit=set(sorted(outside,key=lambda p:rank(seed,'independent-unlabeled-audit',p))[:len(outside)//10])
    permanently_reserved=maximum['R1']|maximum['R2']|candidate_audit
    source_hash={str(p.relative_to(root)):sha(p) for p in [base,challenge,*reports,reservation,meta/'source_versions.json']}
    import audit_pairs,report_strata
    code_hash={Path(p).name:sha(Path(p)) for p in [__file__,audit_pairs.__file__,report_strata.__file__]}
    summary={
        'schema_version':'phase1-development-v1','seed':seed,'replicate':1,
        'design_scope':'Development only; no positive/negative ground truth inferred. E remains unallocated; no external dataset loaded.',
        'source_sha256':source_hash,'code_sha256':code_hash,'matched_checks':len(matched),'development_patients':len(chosen),
        'unselected_repeat_volumes':len(matched)-len(chosen),'reserved_engineering_T':len(pinned),
        'candidate_audit_policy':'10% (floor) of patients outside the 20% paid union. These patients never enter model or gate fitting.',
        'unlabeled_audit_patients':len(candidate_audit),'official_heldout_overlap':0,
        'first_volume_policy':'Four previously observed engineering patients keep their exact paid volume; otherwise lowest SHA-256 priority by patient volume name.',
        'population_report_strata':dict(counts),'stratum_target_mixture':'3/4 observed prevalence + 1/4 equal share, renormalized on present strata.',
        'allocation':'Hamilton apportionment at each floor(p*N), ties by T/R1/R2 order. Fill nested role quotas by cumulative stratum deficit; conditional on four fixed T patients.',
        'inference_weight':'Nonpilot within-stratum role inclusion = number of regular patients selected into that role / regular stratum size. Pilot T inclusion probability is one conditional on prior observation. Use role-specific probabilities for eligible samples only.',
        'budget_groups':{},'invariants':{'one_volume_per_patient':True,'nested_each_role':True,'patient_roles_disjoint_within_budget':True,'reference_union_always_quarantined_from_training':True},
        'annotation_status':'Nominated only. R1/R2 need new complete blinded masks. No mask read, opened or marked complete by this assignment.'}
    private=dest/'private'
    allocation=[]
    for p in sorted(chosen):
        row=dict(patient=p,volume=chosen[p],report_stratum=stratum[p],report_state=complete[chosen[p]][1],pinned_T=int(p in pinned),
                 independent_candidate_audit=int(p in candidate_audit))
        for pct in BUDGETS:
            role=next((r for r in ROLES if p in snapshots[pct][r]),None)
            if role is None:
                role='U_audit_sealed' if p in candidate_audit else ('reference_future_sealed' if p in permanently_reserved else 'U_train')
            row[f'role_{pct}']=role
        allocation.append(row)
    csvfile(private/'allocation.csv',list(allocation[0]),allocation)
    for pct in BUDGETS:
        roles=snapshots[pct]
        assert all(not roles[a]&roles[b] for i,a in enumerate(ROLES) for b in ROLES[i+1:])
        assert set(pinned)<=roles['T']
        if pct!=20:
            nxt=snapshots[10 if pct==5 else 20]
            assert all(roles[r]<=nxt[r] for r in ROLES)
        role_dir=dest/f'b{pct:02d}'
        unlab=set(chosen)-permanently_reserved-roles['T']
        assert not unlab&(roles['T']|roles['R1']|roles['R2']|candidate_audit)
        for file,ids in [(role_dir/'T.csv',roles['T']),(role_dir/'U_train.csv',unlab),
                         (private/f'b{pct:02d}_R1.csv',roles['R1']),(private/f'b{pct:02d}_R2.csv',roles['R2'])]:
            csvfile(file,['patient','volume'],[dict(patient=p,volume=chosen[p]) for p in sorted(ids)])
        blind=[]
        for r in ROLES:
            for p in sorted(roles[r]):
                blind.append(dict(code=rank(seed,'blinded-reader',p)[:20],role=r,patient=p,volume=chosen[p]))
        assert len({b['code'] for b in blind})==len(blind)
        csvfile(private/f'b{pct:02d}_annotation_key.csv',['code','role','patient','volume'],blind)
        csvfile(private/f'b{pct:02d}_blinded_complete_mask_queue.csv',['code'],[{'code':b['code']} for b in sorted(blind,key=lambda x:x['code']) if b['role'] in ('R1','R2')])
        rolecounts={r:dict(Counter(stratum[p] for p in roles[r])) for r in ROLES}
        prob={}
        for s in support:
            prob[s]={r: {'regular_n':sum(stratum[p]==s and p not in pinned for p in roles[r]),'eligible_regular_N':available[s]} for r in ROLES}
        summary['budget_groups'][str(pct)]={
            'paid_patients':sum(map(len,roles.values())),'roles':{r:len(roles[r]) for r in ROLES},
            'U_train':len(unlab),'report_strata_by_role':rolecounts,'regular_role_probabilities_as_n_over_N':prob}
    csvfile(private/'unused_volume_registry.csv',['patient','volume'],
            [dict(patient=p,volume=v) for p in sorted(grouped) for v in sorted(grouped[p]) if v!=chosen[p]])
    csvfile(private/'independent_candidate_audit.csv',['patient','volume'],
            [dict(patient=p,volume=chosen[p]) for p in sorted(candidate_audit)])
    jsonfile(dest/'summary.json',summary)
    files=sorted(p for p in dest.rglob('*') if p.is_file())
    manifest=''.join(f'{sha(p)}  {p.relative_to(dest).as_posix()}\n' for p in files)
    (dest/'SHA256SUMS').write_text(manifest,encoding='utf-8')
    return summary


if __name__=='__main__':
    import fcntl
    a=argparse.ArgumentParser(description=__doc__);a.add_argument('--root',type=Path,required=True);a.add_argument('--out',type=Path,required=True);a.add_argument('--seed',type=int,default=891)
    args=a.parse_args()
    os.umask(0o077)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    with (args.out.parent/'.cohort-build.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        tmp=Path(tempfile.mkdtemp(dir=args.out.parent,prefix='.cohort-'))
        try:
            info=build(args.root.resolve(),tmp,args.seed)
            if args.out.exists():
                if (args.out/'SHA256SUMS').read_bytes()!=(tmp/'SHA256SUMS').read_bytes():
                    raise RuntimeError('Previous allocation differs; use a new version path.')
                for p in tmp.rglob('*'):
                    if p.is_file() and sha(p)!=sha(args.out/p.relative_to(tmp)):
                        raise RuntimeError('Existing allocation failed integrity check')
                print('Independent deterministic rebuild matched all previous output bytes.')
            else:
                tmp.rename(args.out)
            print(json.dumps({'N':info['development_patients'],'budgets':{x:r['roles'] for x,r in info['budget_groups'].items()},
                              'manifest_sha256':sha(args.out/'SHA256SUMS')}))
        finally:
            if tmp.exists():
                shutil.rmtree(tmp)
