"""Download publisher-pinned data via official signed CDN links with a resumable cache."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Lock
import os,re,shutil,time
import requests
from huggingface_hub import HfApi,get_token,hf_hub_url
from frozen_priors import sha
from thesis_expand_screen import ROOT

CHUNK=6*1024**2

def download(repo,revision,name,temp):
    home=hf_hub_url(repo,name,repo_type="dataset",revision=revision)
    headers={"Authorization":"Bearer "+get_token()}
    lock=Lock(); lease=None
    def renew(force=False):
        nonlocal lease
        with lock:
            if lease and not force and time.monotonic()-lease[3]<900:return lease
            for i in range(8):
                try:
                    response=requests.head(home,headers=headers,timeout=(12,20),allow_redirects=False)
                    if response.status_code==403:
                        raise PermissionError("Publisher access denied")
                    response.raise_for_status()
                    target=response.headers.get("Location",home)
                    etag=response.headers.get("X-Linked-Etag",response.headers.get("ETag","")).strip('"')
                    size=response.headers.get("X-Linked-Size",response.headers.get("Content-Length"))
                    if not (re.fullmatch("[0-9a-f]{64}",etag) and size):
                        api=HfApi(endpoint="https://huggingface.co")
                        info=api.get_paths_info(repo,[name],repo_type="dataset",revision=revision)[0].lfs
                        etag=info.sha256;size=info.size
                    lease=(target,int(size),etag,time.monotonic())
                    return lease
                except PermissionError:raise
                except Exception:
                    if i==7:raise
                    time.sleep(2+min(i,4))
    _,filesize,digest,_=renew()
    cache=ROOT/"download_spool"/digest
    cache.mkdir(exist_ok=True,parents=True)
    ranges=list(range(0,filesize,CHUNK))
    def get(offset):
        end=min(filesize,offset+CHUNK)-1
        count=end+1-offset
        dest=cache/f"{offset:012}"
        if dest.exists() and dest.stat().st_size==count:return dest
        for n in range(8):
            try:
                remote,total,etag,_=renew()
                assert total==filesize and etag==digest
                h={"Range":f"bytes={offset}-{end}"}
                if remote==home:h.update(headers)
                with requests.get(remote,headers=h,stream=True,timeout=(10,43)) as r:
                    if r.status_code in (403,410) and remote!=home:
                        renew(force=True);continue
                    if not ((r.status_code==206 and r.headers.get("Content-Range")==f"bytes {offset}-{end}/{filesize}")
                        or (len(ranges)==1 and r.status_code==200)):
                        raise ValueError(f"Publisher range response {r.status_code}")
                    raw=r.content
                if len(raw)!=count:raise ValueError("Publisher range length differs")
                pending=dest.with_suffix(f".part-{os.getpid()}")
                pending.write_bytes(raw)
                pending.replace(dest)
                return dest
            except Exception:
                if n==7:raise
                time.sleep(1+min(n,4))
        raise OSError("Publisher range lease was unavailable")
    with ThreadPoolExecutor(max_workers=2) as pool:
        parts=list(pool.map(get,ranges))
    target=Path(temp)/name
    target.parent.mkdir(exist_ok=True,parents=True)
    with target.open("wb") as out:
        for path in parts:
            with path.open("rb") as f:shutil.copyfileobj(f,out)
    if sha(target)!=digest:
        for f in parts:f.unlink(missing_ok=True)
        raise ValueError("Publisher SHA-256 mismatch")
    for path in parts:path.unlink(missing_ok=True)
    return target,digest

