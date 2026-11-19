"""Metadata-only pleural-effusion feasibility. ReX annotation values only for signed T volumes."""
import argparse
from collections import Counter
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import re
import tempfile

from audit_pairs import canonical_section, identity, run as join_audit
from report_strata import describe_report, laterality, VERSION


MIXED = re.compile(r'\b(?:thicken\w*|plaques?|calcif\w*|pneumothorax|hydropneumothorax|'
                   r'pericard\w*|consolidat\w*|atelecta\w*|nodul\w*|mass(?:es)?|'
                   r'fibro\w*|scarr?\w*|emphysem\w*|tumou?r\w*|pneumoni\w*)\b',re.I)
FLUID = re.compile(r'\b(?:effusions?|fluid|hydrothorax)\b',re.I)
THICK = re.compile(r'\b(?:thicken\w*|plaques?|calcif\w*)\b',re.I)


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1<<20),b''):h.update(block)
    return h.hexdigest()


def save(path,data,jsonl=False):
    with tempfile.NamedTemporaryFile(mode='w',dir=path.parent,encoding='utf-8',delete=False) as f:
        if jsonl:
            for obj in data: f.write(json.dumps(obj,ensure_ascii=False)+'\n')
        else:
            json.dump(data,f,indent=2,ensure_ascii=False); f.write('\n')
    os.replace(f.name,path)


def scan_corrections(root,names):
    status={}
    excludes=set()
    for table,src in [('train','train'),('validation','valid')]:
        excludes.update(Path(line.strip()).name for line in (root/f'no_chest_{src}.txt').read_text().splitlines())
        with (root/f'{table}_metadata.csv').open(encoding='utf-8-sig',newline='') as file:
            for row in csv.DictReader(file):
                name=row['VolumeName']
                if name not in names: continue
                try:
                    slope,inter,spacing=[float(row[k]) for k in ('RescaleSlope','RescaleIntercept','ZSpacing')]
                    ok=all(math.isfinite(v) for v in (slope,inter,spacing)) and min(slope,spacing)>0
                except (ValueError,TypeError):ok=False
                status[name]='duplicate_calibration_row' if name in status else ('listed_calibration_finite' if ok else 'check_calibration')
    return {n:('listed_nonchest' if n in excludes else status.get(n,'no_calibration_row')) for n in names}


def classify(text):
    info=describe_report(text,'')
    if FLUID.search(text) and (MIXED.search(text) or info['state']=='conflict'):
        return 'mixed_finding_or_conflict'
    if info['state']=='affirmed':
        return 'isolated_affirmative_fluid_wording'
    if info['state'] in {'uncertain','negated','historical'} or FLUID.search(text):
        return 'uncertain_negated_historical_or_location_unspecified'
    return 'thickening_plaque_without_fluid' if THICK.search(text) else 'other_or_unresolved'


def examine_t(rows, features, authorized):
    result=[]
    for case in rows:
        name=case['name']
        if name not in authorized: continue
        findings,categories=case['findings'],case['categories']
        entries=[]
        outside_count=0
        for key,text in findings.items():
            category=categories.get(key)
            if category!='2e':
                outside_count+=int(describe_report(text,'')['state']=='affirmed')
                continue
            n=case.get('entity_counts',{}).get(key)
            vox=case.get('pixels',{}).get(key)
            shape=case.get('shape',[])
            plausible=(key.isdigit() and int(key)<len(findings) and len(shape)==3
                       and all(isinstance(s,int) and s>0 for s in shape)
                       and isinstance(n,int) and 0<n<=3 and isinstance(vox,int) and 0<vox<=math.prod(shape))
            cls=classify(text)
            entries.append({'channel':key,'wording':cls,'side_words':laterality(text),
                            'entity_count':n,'annotated_voxels':vox,'plausible_index_and_counts':plausible,
                            'finding_quote_private':text})
        info=features[name]
        result.append({'volume':name,'patient':'_'.join(identity(name)[1]),'report_state':info['state'],
                       'data_check':info['calibration'],'total_annotated_findings':len(findings),
                       'distinct_annotated_categories':len(set(categories.values())),
                       'other_category_affirmative_mention_count':outside_count,'2e_findings':entries})
    if len(result)!=len(authorized): raise ValueError('Signed T volumes are not unique in original train manifest')
    return result


def counts(rows,key):
    return dict(sorted(Counter(r[key] for r in rows).items()))


def audit(args):
    os.umask(0o077)
    args.out.mkdir(parents=True,exist_ok=True,mode=0o700)
    source=args.metadata/'rex/dataset.json'
    reportdir=args.metadata/'ct_rate/dataset/radiology_text_reports'
    reports=[reportdir/'train_reports.csv',reportdir/'validation_reports.csv']
    names=[]
    join_audit(source,reports,[args.metadata/'rex/MICCAI_challenge_dataset.json'],matched_names=names)
    names=set(names)
    meta=scan_corrections(args.calibration,names)
    features={}
    for path in reports:
        with path.open(encoding='utf-8-sig',newline='') as f:
            for row in csv.DictReader(f):
                name=row['VolumeName']
                if name not in names: continue
                parts=[canonical_section(row[k]) for k in ('Findings_EN','Impressions_EN')]
                features[name]={**describe_report(*parts),'volume':name,'patient':'_'.join(identity(name)[1]),
                                'report_words':sum(len(s.split()) for s in parts),'calibration':meta[name]}
    if set(features)!=names:raise ValueError('Unexpected reports missing from audited pool')
    if args.approved.suffix=='.csv':
        if args.approved.name!='T.csv':raise ValueError('Expected a T-only cohort CSV')
        with args.approved.open(encoding='utf-8-sig',newline='') as f:
            reader=csv.DictReader(f)
            if reader.fieldnames!=['patient','volume']: raise ValueError('Unexpected T whitelist header')
            pairs=list(reader)
        if len({r['patient'] for r in pairs})!=len(pairs):raise ValueError('T CSV has repeated patient IDs')
        if any(r['patient']!='_'.join(identity(r['volume'])[1]) for r in pairs):
            raise ValueError('T CSV patient and volume disagree')
        vals=[r['volume'] for r in pairs]
    else:
        allowed=json.loads(args.approved.read_text(encoding='utf-8'))
        if allowed.get('role') not in ('T_paid_engineering_pilot','T_paid_training'):
            raise ValueError('Pass a T-only CSV, a signed T manifest, or the pilot reservation')
        vals=allowed['volumes']
    if not vals or not isinstance(vals,list) or len(set(vals))!=len(vals) or not set(vals)<=names:
        raise ValueError('Unexpected training whitelist; every volume must be audited and eligible')
    trows=examine_t(json.loads(source.read_text())['train'],features,set(vals))
    frows=[features[n] for n in sorted(names)]
    entries=[entry for r in trows for entry in r['2e_findings']]
    isolated=[e for e in entries if e['wording']=='isolated_affirmative_fluid_wording']
    n_eligible=lambda row:any(e['wording']=='isolated_affirmative_fluid_wording' and e['plausible_index_and_counts'] for e in row['2e_findings'])
    silver=[r for r in trows if r['data_check']=='listed_calibration_finite' and n_eligible(r)]
    report_states=Counter((r['report_state'],int(n_eligible(r))) for r in trows)
    summary={'scope':'Feasibility only. Pool counts use raw report text and correction metadata only. ReX findings inspected only after volume membership in the supplied T whitelist. No new patient assignment, CT, mask pixels or external outcomes.',
             'report_rule':VERSION,'report_rule_sha256':digest(Path(__file__).with_name('report_strata.py')),
             'whitelist_sha256':digest(args.approved),'original_release_manifest_sha256':digest(source),
             'eligible_report_pool':{'volumes':len(frows),'patients':len({r['patient'] for r in frows}),
                'counts_by_volume_report_strata':counts(frows,'stratum'),'counts_by_volume_report_state':counts(frows,'state'),
                'counts_by_volume_report_affirmed_side':counts([r for r in frows if r['state']=='affirmed'],'side'),
                'counts_by_volume_calibration':counts(frows,'calibration')},
             'paid_T_only':{'volumes':len(trows),'patients':len({r['patient'] for r in trows}),
                'volumes_with_any_2e':sum(bool(r['2e_findings']) for r in trows),'2e_channels':len(entries),
                '2e_channel_wording':counts(entries,'wording'),'isolated_fluid_wording_side':counts(isolated,'side_words'),
                'isolated_fluid_wording_with_multiple_entities':sum(isinstance(e['entity_count'],int) and e['entity_count']>1 for e in isolated),
                'isolated_fluid_wording_index_and_counts_plausible':sum(e['plausible_index_and_counts'] for e in isolated),
                'eligible_for_image_review_volumes':len(silver),'eligible_for_image_review_patients':len({r['patient'] for r in silver}),
                'review_volumes_with_additional_other_finding_channels':sum(r['total_annotated_findings']>sum(e['wording']=='isolated_affirmative_fluid_wording' for e in r['2e_findings']) for r in silver),
                'multi_2e_channel_volumes':sum(len(r['2e_findings'])>1 for r in trows),
                'volumes_with_affirmative_mention_in_non_2e_category':sum(bool(r['other_category_affirmative_mention_count']) for r in trows),
                'report_state_by_isolated_2e_candidate_volume':{f'{state}|{flag}':n for (state,flag),n in sorted(report_states.items())}},
             'interpretation':'Isolated affirmative wording and valid indices permit manual image-mask review only. Unmarked regions remain unknown. Non-effusion and unmentioned reports do not establish whole-volume negatives. Clinical truth/completeness unmeasured.'}
    save(args.out/'report_only_pool.private.jsonl',frows,True)
    save(args.out/'paid_T_finding_candidates.private.jsonl',trows,True)
    save(args.out/'silver_review_candidates.private.jsonl',
        [{'volume':r['volume'],'patient':r['patient'],'candidate_channels':[int(e['channel']) for e in r['2e_findings'] if e['wording']=='isolated_affirmative_fluid_wording' and e['plausible_index_and_counts']],
          'report_state':r['report_state'],'interpretation':'positive candidate finding only; image overlay and clinical review pending; whole-volume completeness unknown'} for r in silver],True)
    save(args.out/'screening_summary.json',summary)
    print(json.dumps(summary,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--metadata',type=Path,default=Path('metadata'))
    p.add_argument('--calibration',type=Path,default=Path('data/four_train/ct_rate/dataset/metadata'))
    p.add_argument('--approved',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    audit(p.parse_args())
