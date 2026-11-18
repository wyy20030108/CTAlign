"""Audit four train-only CT/report/ReX mask triples after preflight_data.sh.

Use the original dataset/train or dataset/valid tree at the pinned revision.
The release also includes separate *_fixed trees. Do not silently mix them,
assume CT intensities are calibrated, or transfer an affine to a ReX mask.
"""

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
from tempfile import NamedTemporaryFile

import nibabel as nib
import numpy as np
from huggingface_hub import hf_hub_download

from audit_pairs import canonical_section, identity, run as audit_pairs
from fetch_manifest import FILES


def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, obj):
    with NamedTemporaryFile('w', encoding='utf-8', dir=path.parent, delete=False) as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
        f.write('\n')
    os.replace(f.name, path)


def fetch(label, filename, out):
    repo, revision, _ = FILES[label]
    path = hf_hub_download(repo_id=repo, repo_type='dataset', revision=revision, filename=filename,
                           endpoint='https://huggingface.co', local_dir=out / label)
    return Path(path)


def headers(ct_path, mask_path, row):
    ct = nib.load(ct_path)
    mask = nib.load(mask_path)
    nominal = tuple(row['shape'])
    n = len(row['findings'])
    shape_ok = len(nominal) == 3 and n > 0 and ct.shape == nominal and mask.shape == (n, *nominal)
    index_ok = set(row['findings']) == {str(i) for i in range(n)}
    def describe(img):
        return {'shape': list(img.shape), 'dtype': str(img.get_data_dtype()),
                'zooms_raw': [float(z) for z in img.header.get_zooms()],
                'sform_code': int(img.header['sform_code']), 'qform_code': int(img.header['qform_code']),
                'affine_raw': img.affine.tolist(),
                'data_scaling': [float(img.dataobj.slope), float(img.dataobj.inter)]}
    ct_ok = (np.isfinite(ct.affine).all() and abs(np.linalg.det(ct.affine[:3, :3])) > 1e-9
             and all(np.isfinite(v) and v > 0 for v in ct.header.get_zooms()[:3]))
    return {'ct': describe(ct), 'mask': describe(mask), 'expected_HWD': nominal, 'expected_F': n,
            'documented_layout_and_indices_match': shape_ok and index_ok, 'ct_grid_finite': bool(ct_ok),
            'voxel_alignment_verified': False, 'complete_effusion_mask_verified': False}


def get_four(metadata, out, *, seed=891, extra=(), downloader=fetch):
    os.umask(0o077)
    out.mkdir(mode=0o700, parents=True, exist_ok=True)
    inventory = json.loads((metadata/'source_versions.json').read_text())
    recorded = {(v['repo'], v['revision'], v['file']) for v in inventory['files']}
    expected = {(repo, rev, fn) for repo, rev, files in FILES.values() for fn in files}
    if not inventory['downloaded'] or recorded != expected:
        raise ValueError('Run preflight_data.sh with the pinned releases before fetching the four cases.')
    rex, challenge = [metadata/'rex'/f for f in FILES['rex'][2]]
    reports = [metadata/'ct_rate'/f for f in FILES['ct_rate'][2]]
    candidates = []
    audit = audit_pairs(rex, reports, [challenge, *extra], matched_names=candidates)
    if any(audit['rex_by_split'].get(k, {}).get('rows', 0) == 0 for k in ('train','val','test','holdout_1_val','holdout_1_test')):
        raise ValueError('Original and challenge holdout manifests must be populated.')
    # Rank using volume keys only. Commit these patients to T before accessing their annotations.
    candidates.sort(key=lambda name: hashlib.sha256(f'{seed}:{name}'.encode()).digest())
    chosen, patients = [], set()
    for name in candidates:
        patient = identity(name)[1]
        if patient not in patients:
            chosen.append(name)
            patients.add(patient)
        if len(chosen) == 4:
            break
    if len(chosen) != 4:
        raise ValueError('Fewer than four eligible distinct patients.')
    lock = {'role': 'T_paid_engineering_pilot', 'seed': seed, 'volumes': chosen,
            'patients': sorted('_'.join(p) for p in patients)}
    reservation = out/'pilot_T_reservation.json'
    try:
        with reservation.open('x', encoding='utf-8') as f:
            json.dump(lock, f, indent=2)
            f.write('\n')
    except FileExistsError:
        if json.loads(reservation.read_text()) != lock:
            raise ValueError('Existing T reservation differs; keep the previous reservation intact.')
    selected = set(chosen)
    rows = {r['name']:r for r in json.loads(rex.read_text())['train'] if r['name'] in selected}
    texts = {}
    for report_csv in reports:
        with report_csv.open(encoding='utf-8-sig', newline='') as f:
            for record in csv.DictReader(f):
                name = record['VolumeName'].strip()
                if name in selected:
                    texts[name] = {k: canonical_section(record[k]) for k in ('Findings_EN','Impressions_EN')}
    if set(texts) != selected or set(rows) != selected:
        raise ValueError('A reserved training volume lacks a joined report or ReX row.')
    report_path = out/'four_reports.private.json'
    write_json(report_path, texts)
    note = downloader('ct_rate', 'dataset/data_correction_note.md', out)
    results = []
    for name in chosen:
        _, (split, number), (_, _, exam) = identity(name)
        patient = f'{split}_{number}'
        image_file = f'dataset/{split}/{patient}/{patient}_{exam}/{name}'
        mask_file = f'segmentations/{name}'
        ct = downloader('ct_rate', image_file, out)
        mask = downloader('rex', mask_file, out)
        result = {'name':name, 'ct_release_path':image_file, 'ct_sha256':sha256(ct),
                  'mask_release_path':mask_file, 'mask_sha256':sha256(mask),
                  'report_file':report_path.name, 'report_key':name,
                  'headers':headers(ct, mask, rows[name])}
        results.append(result)
    record = {'repository_revisions': {repo: rev for repo, rev, _ in FILES.values()},
              'correction_note_local':str(note), 'correction_note_sha256':sha256(note),
              'report_subset_sha256':sha256(report_path), 'reservation':reservation.name,
              'clinical_status':'4 training patients reserved within T. No target-label selection, no confirmed effusion labels.',
              'geometry_review':'Read the pinned correction note for original vs fixed volumes, including HU/spacing. Review CT-mask overlays and laterality before training. ReX has finding-first FHWD storage; its raw NIfTI affine cannot be assumed to map HWD to CT physical space. Category 2e includes thickening and effusion.',
              'cases':results}
    write_json(out/'four_header_audit.json', record)
    bad = sum(not (r['headers']['documented_layout_and_indices_match'] and r['headers']['ct_grid_finite']) for r in results)
    return {'unique_reserved_train_patients':4,'header_mismatches':bad,'ready_for_training':False}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--metadata', type=Path, default=Path('metadata'))
    parser.add_argument('--out', type=Path, default=Path('data/four_train'))
    parser.add_argument('--extra-holdout', type=Path, action='append', default=[],
                        help='Additional patient exclusions as a val/test volume-name JSON, if already defined.')
    args = parser.parse_args()
    result = get_four(args.metadata, args.out, extra=args.extra_holdout)
    print(json.dumps(result))
    raise SystemExit(2 if result['header_mismatches'] else 0)
