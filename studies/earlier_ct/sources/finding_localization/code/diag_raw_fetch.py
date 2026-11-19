from pathlib import Path
import json,concurrent.futures
from prepare_unlabeled import obtain_file,release_path
from fetch_manifest import FILES
d=Path('phase5_diag');anchor=d/'raw_cache_private';anchor.mkdir(exist_ok=True,mode=0o700)
rows=json.loads(Path('phase3_swap/fallback/fallback_train.private.json').read_text())
ids=json.loads((d/'selection_clean.json').read_text())['source_audit_codes']
jobs=[]
for r in rows:
 if r['code'] in ids:
  fixed=release_path(r['name']);repo,rev,_=FILES['ct_rate']
  jobs.extend([(r['code'],repo,rev,fixed),(r['code'],repo,rev,fixed.replace('_fixed/','/'))])
  repo,rev,_=FILES['rex'];jobs.append((r['code'],repo,rev,'segmentations/'+r['name']))
def go(j):
 c,repo,v,n=j;p,h=obtain_file(repo,v,n,anchor)
 print(c,n.split('/')[1],h,flush=True)
 return dict(code=c,repo=repo,rev=v,publisher_path=n,local=str(p),sha256=h)
with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool: result=list(pool.map(go,jobs))
(d/'raw_receipts.private.json').write_text(json.dumps(result,indent=2))
