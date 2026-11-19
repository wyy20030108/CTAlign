"""Coordinate two CT-only Basel workers behind the original R1 lock. No clinical endpoint is read."""
import argparse
import concurrent.futures
import csv
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import threading
import time


def read(p):
    return json.loads(p.read_text())


def sha(p):
    with p.open('rb') as f:
        return hashlib.file_digest(f,'sha256').hexdigest()


def put(p,obj):
    with tempfile.NamedTemporaryFile('w',dir=p.parent,delete=False) as f:
        json.dump(obj,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
    os.replace(f.name,p)


def header(ct):
    import nibabel as nib
    import numpy as np
    x=nib.load(ct); h=x.header
    if len(x.shape)!=3 or h.get_xyzt_units()[0]!='mm' or not np.isfinite(x.affine).all() or abs(np.linalg.det(x.affine[:3,:3]))<1e-6:
        raise ValueError('Invalid native CT header')
    return {'shape':list(x.shape),'affine':x.affine.tolist(),'spatial_unit':'mm'}


def cache_entry(root,r,source,digest):
    if sha(source)!=digest:
        raise ValueError('Input digest differs from original pinned download')
    out=root/'phase1/R1_AI_drafts'; folder=out/'cache_private'/r['case'];folder.mkdir(parents=True,exist_ok=True)
    file=folder/'ct.nii.gz'; note=folder/'source.json'
    if file.exists():
        if sha(file)!=digest:
            raise ValueError('Conflicting cached CT')
    else:
        os.link(source,file)
    item={'case':r['case'],'volume':r['volume'],'ct_sha256':digest,'grid':header(file)}
    if note.exists() and read(note)!=item:
        raise ValueError('Conflicting source header')
    put(note,item)
    native=out/'native_headers_private'; native.mkdir(exist_ok=True)
    put(native/(r['case']+'.json'),item)
    return file,digest


def transfer_legacy_batch(root,work):
    out=root/'phase1/R1_AI_drafts'; mapping={r['case']:r for r in read(out/'mapping.private.json')}
    if not work.is_relative_to(out) or not work.name.startswith('stream_'):
        raise ValueError('Not a legacy batch')
    batch=read(work/'batch.private.json')
    for r in batch:
        src=Path(r['image_path']).resolve()
        if not src.is_relative_to(work/'download') or src.name!=mapping[r['case']]['volume']:
            raise ValueError('Unexpected legacy cached source')
        cache_entry(root,mapping[r['case']],src,r['sha256'])
    put(out/'handoff_completed.json',{'phase':'legacy stopped at next batch entry before inference','legacy_parent_pid':os.getppid(),
                                    'transferred_verified_sources':len(batch),'committed_receipts':len(list((out/'receipts').glob('*.json')))})


def source_for(root,r):
    from fetch_manifest import FILES
    from prepare_unlabeled import obtain_file,release_path
    out=root/'phase1/R1_AI_drafts';folder=out/'cache_private'/r['case'];folder.mkdir(parents=True,exist_ok=True)
    note=folder/'source.json'; file=folder/'ct.nii.gz'
    if note.exists() and file.exists():
        n=read(note)
        if n['volume']!=r['volume'] or sha(file)!=n['ct_sha256']:
            raise ValueError('Cached CT checksum mismatch')
        return file,n['ct_sha256']
    repo,rev,_=FILES['ct_rate']; src,digest=obtain_file(repo,rev,release_path(r['volume']),folder/'publisher')
    return cache_entry(root,r,src,digest)


def completed(out,mapping):
    done=set()
    for r in mapping:
        tag=r['case'];f=out/'receipts'/(tag+'.json')
        if f.exists():
            record=read(f); mask=out/'masks'/(tag+'_AI_effusion.nii.gz')
            if (record['case']!=tag or not record['full_input_grid_and_affine_verified'] or sha(mask)!=record['export_sha256']):
                raise ValueError('R1 committed receipt/mask fails verification')
            done.add(tag)
    put(out/'progress.json',{'scheduled':len(mapping),'completed':len(done),'verified_native_grid':len(done),'human_verified':0,
                             'reference_losses_computed':0,'role':'algorithmic drafts; disease counts sealed'})
    return done


def run_batch(root,gpu,rows):
    from fetch_four import sha256
    out=root/'phase1/R1_AI_drafts'; base=root/'phase1/ai_contours'
    if shutil.disk_usage(out).free<3*1024**3:
        raise RuntimeError('Insufficient transient disk space')
    env=os.environ.copy(); env.update(CUDA_VISIBLE_DEVICES=str(gpu),PYTHONPATH=os.pathsep.join([str(root),str(base)]),
                                     TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD='1',nnUNet_n_proc_DA='2')
    for k in ('nnUNet_raw_data_base','nnUNet_preprocessed','RESULTS_FOLDER'):
        env[k]=str(base/'nnunet_runtime'/k)
    with tempfile.TemporaryDirectory(dir=out,prefix=f'gpu{gpu}_') as td:
        work=Path(td);inp=work/'ct_only';inp.mkdir();records=[]
        for r in rows:
            file,h=source_for(root,r); alias=inp/(r['case']+'_0000.nii.gz');alias.symlink_to(file.resolve())
            records.append({'case':r['case'],'image_path':str(alias),'sha256':h})
        put(work/'batch.private.json',records)
        log=out/'logs_private';log.mkdir(exist_ok=True)
        with (log/(rows[0]['case']+'.log')).open('ab') as f:
            subprocess.run([str(base/'env/bin/python'),str(root/'r1_drafts.py'),'--child',str(work),'--project',str(root)],
                           cwd=root,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
        qc=read(work/'case_geometry_qc.json');by_case={r['case']:r for r in qc}
        if set(by_case)!={r['case'] for r in rows} or len(qc)!=len(rows):
            raise ValueError('Incomplete batch')
        import nibabel as nib
        import numpy as np
        for r,src in zip(rows,records,strict=True):
            tag=r['case']; rec=by_case[tag]; p=work/'ai_binary'/(tag+'_AI_effusion.nii.gz');g=header(p); n=read(out/'native_headers_private'/(tag+'.json'))
            if (rec['ct_sha256']!=src['sha256'] or not rec['full_input_grid_and_affine_verified'] or sha256(p)!=rec['export_sha256'] or
                g['shape']!=n['grid']['shape'] or not np.allclose(g['affine'],n['grid']['affine'],atol=1e-5,rtol=0) or
                nib.load(p).get_data_dtype()!=np.uint8):
                raise ValueError('Native-grid or checksum mismatch on commit')
            mask=out/'masks'/p.name; receipt=out/'receipts'/(tag+'.json')
            if receipt.exists():
                if sha(mask)!=read(receipt)['export_sha256']:
                    raise ValueError('Previously committed mask changed')
                continue
            p.replace(mask);put(receipt,rec)
    # Only this worker's own ephemeral source cache is retired after all receipts have committed.
    for r in rows:
        folder=(out/'cache_private'/r['case']).resolve()
        if folder.parent==(out/'cache_private').resolve() and (out/'receipts'/(r['case']+'.json')).is_file():
            shutil.rmtree(folder)


def run(root):
    os.umask(0o077);out=root/'phase1/R1_AI_drafts'; (out/'native_headers_private').mkdir(exist_ok=True)
    lock=(out/'worker.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX)
    mapping=read(out/'mapping.private.json');plan=read(out/'frozen_protocol.json')
    roster=root/'phase1/cohort/split_v1/private/b10_R1.csv'
    with roster.open(newline='') as f:
        original=sorted(csv.DictReader(f), key=lambda r:hashlib.sha256(('R1-drafts:891:'+r['patient']).encode()).digest())
    if (len(mapping)!=54 or plan['source_roster_sha256']!=sha(roster) or plan['weights_sha256']!=read(root/'phase1/ai_contours/model_provenance.json')['weights_archive_sha256']
        or mapping!=[{'case':f'R1A{i:03d}','volume':r['volume']} for i,r in enumerate(original,1)]):
        raise ValueError('Changed roster or protocol')
    done=completed(out,mapping); queue=[r for r in mapping if r['case'] not in done]; mutex=threading.Lock()
    def worker(gpu):
        while True:
            with mutex:
                if not queue:
                    return
                if not (out/f'pause_gpu{gpu}').exists():
                    batch=queue[:2];del queue[:len(batch)]
                else:
                    batch=[]
            if not batch:
                time.sleep(5); continue
            for attempt in range(3):
                try:
                    run_batch(root,gpu,batch);break
                except Exception as error:
                    print('worker',gpu,'batch',batch[0]['case'],'attempt',attempt+1,'failed',type(error).__name__,flush=True)
                    if attempt==2:
                        raise
                    time.sleep(5*(attempt+1))
            with mutex:
                done=completed(out,mapping)
                print('worker',gpu,'verified',len(done),'/54',flush=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        tasks=[pool.submit(worker,g) for g in (0,1)]
        for task in concurrent.futures.as_completed(tasks):
            task.result()
    if len(completed(out,mapping))!=54:
        raise ValueError('Incomplete R1 panel')
    print('54/54 ready for independent original-header audit',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--project',type=Path,default=Path('.'))
    run(p.parse_args().project.resolve())
