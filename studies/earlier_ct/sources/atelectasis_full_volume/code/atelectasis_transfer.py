"""Fetch authorized, pinned release files with validated HTTP range requests."""
from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
import time
import requests
from huggingface_hub import HfApi, get_token, hf_hub_url
from frozen_priors import sha

CHUNK=12*1024**2

def obtain_file(repo,revision,name,destination):
    api=HfApi(endpoint="https://huggingface.co")
    metadata=api.get_paths_info(repo,[name],repo_type="dataset",revision=revision)
    if len(metadata)!=1 or not metadata[0].lfs:
        raise ValueError("Published LFS checksum missing")
    digest=metadata[0].lfs.sha256
    size=metadata[0].lfs.size
    url=hf_hub_url(repo,name,repo_type="dataset",revision=revision)
    token=get_token()
    path=Path(destination)/name
    path.parent.mkdir(parents=True,exist_ok=True)
    partial=path.with_suffix(".ranging")
    if size<CHUNK:
        with requests.get(url,headers={"Authorization":"Bearer "+token},timeout=(12,90)) as response:
            response.raise_for_status();partial.write_bytes(response.content)
    else:
        requested=range(0,size,CHUNK)
        with partial.open("w+b") as stream:
            stream.truncate(size)
            def fetch(offset):
                end=min(size,offset+CHUNK)-1
                header={"Authorization":"Bearer "+token,"Range":f"bytes={offset}-{end}"}
                for attempt in range(4):
                    try:
                        with requests.get(url,headers=header,stream=True,timeout=(12,40)) as response:
                            if response.status_code!=206 or response.headers.get("Content-Range")!=f"bytes {offset}-{end}/{size}":
                                raise RuntimeError("HTTP range mismatch: "+str(response.status_code))
                            chunk=response.content
                            if len(chunk)!=end-offset+1:
                                raise ValueError("HTTP content length differs from the requested range")
                            if os.pwrite(stream.fileno(),chunk,offset)!=len(chunk):
                                raise OSError("Short write in release file")
                        return
                    except Exception:
                        if attempt==3:raise
                        time.sleep(1+attempt)
            with ThreadPoolExecutor(max_workers=5) as pool:
                for _ in pool.map(fetch,requested):pass
    if sha(partial)!=digest:raise ValueError("Final release SHA-256 differs")
    partial.replace(path)
    return path,digest

