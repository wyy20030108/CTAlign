"""Resume case downloads across HTTPS resets without discarding verified chunks."""
import os, re, shutil, time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import requests
from huggingface_hub import HfApi, hf_hub_url, get_token
from atelectasis_prepare import convert, ROOT, RECEIPTS, read
import atelectasis_prepare
from frozen_priors import sha, write

SIZE=6*1024**2

def download(repo, revision, name, destination):
    api=HfApi(endpoint="https://huggingface.co")
    file=None
    for attempt in range(6):
        try:
            file=api.get_paths_info(repo,[name],repo_type="dataset",revision=revision)[0]
            if file.lfs is None:raise ValueError("Missing release checksum")
            break
        except Exception:
            if attempt==5:raise
            time.sleep(2+attempt)
    digest=file.lfs.sha256
    extent=file.lfs.size
    url=hf_hub_url(repo,name,repo_type="dataset",revision=revision)
    token=get_token()
    chunks=ROOT/"download_spool"/digest
    chunks.mkdir(exist_ok=True,parents=True)
    output=Path(destination)/name
    output.parent.mkdir(exist_ok=True,parents=True)
    starts=list(range(0,extent,SIZE))
    # The Hub downloader keeps a sequential temporary prefix while it retries.
    # Seed whole chunks from that prefix. The final publisher hash remains mandatory.
    sources=list(ROOT.glob("L*-*/**/*"+digest+".incomplete"))
    if sources:
        source=max(sources,key=lambda p:p.stat().st_size if p.exists() else 0)
        try:
            with source.open("rb") as old:
                available=os.fstat(old.fileno()).st_size
                for offset in starts:
                    length=min(SIZE,extent-offset)
                    if offset+length>available:break
                    target=chunks/f"{offset:012}"
                    if target.exists():continue
                    old.seek(offset);segment=old.read(length)
                    if len(segment)==length:
                        part=target.with_suffix(f".seed-{os.getpid()}")
                        part.write_bytes(segment);part.replace(target)
        except FileNotFoundError:
            pass

    def retrieve(offset):
        last=min(offset+SIZE,extent)-1
        target=chunks/f"{offset:012}"
        if target.exists() and target.stat().st_size==last-offset+1:return target
        for attempt in range(7):
            try:
                headers={"Authorization":f"Bearer {token}","Range":f"bytes={offset}-{last}"}
                with requests.get(url,headers=headers,stream=True,timeout=(12,36)) as response:
                    if not (response.status_code==206 and response.headers.get("Content-Range")==f"bytes {offset}-{last}/{extent}"):
                        if not (len(starts)==1 and response.status_code==200):raise ValueError(f"Range status {response.status_code}")
                    data=response.content
                if len(data)!=last-offset+1:raise ValueError("Unexpected response length")
                part=target.with_suffix(f".part-{os.getpid()}")
                part.write_bytes(data);part.replace(target)
                return target
            except Exception:
                if attempt==6:raise
                time.sleep(1+min(attempt,3))
    with ThreadPoolExecutor(max_workers=3) as pool:
        parts=list(pool.map(retrieve,starts))
    with output.open("wb") as sink:
        for path in parts:
            with path.open("rb") as f:shutil.copyfileobj(f,sink)
    if sha(output)!=digest:
        for p in parts:p.unlink(missing_ok=True)
        raise ValueError("Full-file publisher checksum mismatch")
    for p in parts:p.unlink(missing_ok=True)
    return output,digest

def main():
    import sys
    os.umask(0o077)
    picker=set(sys.argv[1:])
    atelectasis_prepare.obtain_file=download
    for r in reversed(read(ROOT/"screen.private.json")):
        code=r["code"]
        if code not in picker:continue
        for retry in range(3):
            try:
                out=convert(r,True)
                print(code,out["status"],out.get("reason",""),flush=True)
                break
            except Exception as e:
                print(code,"failed",retry+1,type(e).__name__,str(e).split("http",1)[0][:55],flush=True)
                if retry==2:write(RECEIPTS/(code+".pending.json"),dict(code=code,error=type(e).__name__))
if __name__=="__main__":main()

