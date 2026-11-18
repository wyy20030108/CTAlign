"""Create a private standalone training-data archive on the authorised server."""
import hashlib,json,os,tarfile,io
from pathlib import Path
from swap_data import read,write_json,sha256,out
from audit_pairs import identity

def main():
    os.umask(0o077);src=out/"fallback"; target=src/"clinical_rex_pairs_private.tar"
    records={};data={};seen=set()
    for role in ("fallback_train","fallback_val","swap_dev","unlabelled_train"):
        records[role]=[]
        for i,r in enumerate(read(src/(role+".private.json")),1):
            person=identity(r["name"])[1];assert person not in seen;seen.add(person)
            is_u=role=="unlabelled_train";p=Path(r["file"] if is_u else r["path"])
            sig=r["npz_sha256"] if is_u else r["sha256"]
            assert sha256(p)==sig
            code=f"U{i:02}" if is_u else r["code"]; key="data/"+code+".npz"
            data[key]=(p,sig)
            row=dict(code=code,path=key,source_volume=r["name"],sha256=sig,report=r["report"],
                     CT_key="ct_hu" if is_u else "ct",affine_key="affine")
            if not is_u:row.update(queries=r["queries"],original_channels=r["channels"],categories=r["categories"],mask_key="masks",
                                  side_qc_fraction=r["side_qc_fraction"],source_ct_sha256=r["source_ct"],source_rex_sha256=r["source_rex"])
            records[role].append(row)
    blob=json.dumps(records,ensure_ascii=False,indent=2).encode()
    with tarfile.open(target.with_suffix(".part"),"w") as tar:
        for name,(path,_) in data.items():tar.add(path,arcname=name,recursive=False)
        for n,b in [("splits.private.json",blob),("README.txt",b"Private research data. 2 mm RAS+ CT in HU. See splits.private.json for NPZ keys. Two finding masks per labelled patient; outside a finding is not certified disease-negative. U has only CT and report. swap_dev and fallback_val have already been used for development. Respect original CT-RATE and ReXGroundingCT access terms.\n")]:
            meta=tarfile.TarInfo(n);meta.size=len(b);meta.mode=0o600;tar.addfile(meta,io.BytesIO(b))
    target.with_suffix(".part").replace(target)
    with tarfile.open(target) as tar:
        assert len(tar.getnames())==len(data)+2
        assert tar.extractfile("splits.private.json").read()==blob
        for k,(_,sig) in data.items():
            with tar.extractfile(k) as f:assert hashlib.file_digest(f,"sha256").hexdigest()==sig
    write_json(src/"bundle_receipt.json",dict(path=str(target.resolve()),sha256=sha256(target),bytes=target.stat().st_size,
        patients={k:len(v) for k,v in records.items()},npz_members_rechecked=len(data)))
    print("verified private bundle",len(data),flush=True)
if __name__=="__main__":main()
