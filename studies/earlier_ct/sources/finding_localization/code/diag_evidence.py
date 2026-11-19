"""Verify and assemble private phase-five evidence at its dataset boundary."""
import hashlib, json, os
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

P=Path("phase5_diag")
def digest(b):return hashlib.sha256(b).hexdigest()
def sha(f):return digest(f.read_bytes())
def load(f):return json.loads(f.read_bytes())

def main():
    os.umask(0o077)
    entries={}
    def add(f,name=None):
        k=name or f.as_posix()
        if k in entries:
            assert entries[k]==f.read_bytes()
        else:
            entries[k]=f.read_bytes()
    def cp(f):add(Path(f),"code/"+Path(f).name)
    for name in ("diag_raw","diag_raw_review","diag_original","diag_replay","diag_common","diag_common_verify","diag_localize","diag_localize_refine","diag_evidence"):
        cp(name+".py")
    add(P/"raw_fetch.py","code/diag_raw_fetch.py")
    for n in ("selection.json","selection_clean.json","raw_audit.json","raw_visual_review.json","raw_s002_planes.json"):add(P/n)
    audit=load(P/"raw_audit.json");clean=load(P/"selection_clean.json")["codes"]
    assert clean==["S002","S010","S046"] and set(audit["cases"])==set(clean+["S030"])
    assert sha(Path("diag_raw.py"))==audit["script_sha256"]
    assert sha(P/"raw_visual/S002_A_stack.png")==load(P/"raw_s002_planes.json")["figure_sha256"]
    for f in sorted((P/"raw_visual").glob("*.png")):add(f)
    assert len(list((P/"raw_visual").glob("*.png")))==5
    for mode in ("reference","positive_weight"):
        folder=P/"common_clean"/mode
        summary=load(folder/"summary.json");verified=load(folder/"verification.json")
        assert verified["verified"] and sha(folder/"model.pt")==summary["weights_sha256"] and sha(Path("diag_common.py"))==summary["script_sha256"]
        assert [r["code"] for r in verified["cases"]]==clean
        for row in verified["cases"]:
            assert row["reference_pixels_recounted"] and row["probability_max_abs_reload_diff"]==0
            assert sha(folder/(row["code"]+"_probabilities.png"))==row["figure_sha"]
        for f in sorted(folder.iterdir()):
            if f.suffix in (".pt",".npz",".png",".json"):add(f)
    rep=P/"replay";receipt=load(rep/"receipt.json")
    assert sha(Path("diag_replay.py"))==receipt["source"] and sha(Path("diag_original.py"))==receipt["original_callable_sha"]
    assert sha(rep/"independent_replays.json")==receipt["replays_sha"] and sha(rep/"S002_replay.png")==receipt["png"]
    for row in load(rep/"independent_replays.json")["repeat_results"]:
        i=row["repeat"];d=rep if i==1 else rep/f"unmodified_{i}"
        assert sha(rep/f"unmodified_maps_{i}.npz")==row["maps_sha256"] and sha(d/"original_callable.pt")==row["weights_sha256"]
    for v,sig in receipt["variants"].items():
        assert sha(rep/(v+".json"))==sig and sha(rep/(v+".npz"))==load(rep/(v+".json"))["output_sha256"]
    for f in sorted(rep.rglob("*")):
        if f.is_file() and f.suffix in (".pt",".npz",".png",".json"):add(f)
    # The frozen localisation subrelease carries independently locked prediction and candidate manifests.
    loc=P/"localization";loczip=loc/"localization_evidence.zip"
    with ZipFile(loczip) as src:
        hashes=json.loads(src.read("SHA256.json"))
        assert src.testzip() is None and set(src.namelist())==set(hashes)|{"SHA256.json"}
        for name,sig in hashes.items():
            remote=Path(name[5:]) if name.startswith("code/") else loc/name
            b=src.read(name);assert digest(b)==sig==sha(remote)
            if name.startswith("code/"):add(remote,name)
            else:add(remote)
        entries[(loc/"SUBRELEASE_SHA256.json").as_posix()]=src.read("SHA256.json")
    loc_table=load(loc/"table.json")
    assert set(loc_table)=={"supervised","semi","new_positive_weight"} and all(loc_table[v]["auto"]["targets"]==36 for v in loc_table)
    source_receipts=load(P/"raw_receipts.private.json")
    sanitised=[{k:r[k] for k in ("code","repo","rev","sha256")} for r in source_receipts]
    entries["phase5_diag/raw_source_hashes.json"]=json.dumps(sanitised,indent=2).encode()
    count_by_stage={k:sum(n.startswith(k) for n in entries) for k in ("code/","phase5_diag/common_clean/","phase5_diag/localization/","phase5_diag/replay/","phase5_diag/raw_visual/")}
    meta={"stage_counts":count_by_stage,"raw_audit_codes":list(audit["cases"]),"common_fov_codes":clean,
          "number_of_unmodified_replay_runs":5,"localization_models":list(loc_table),
          "localization_cases_per_model":18,"localization_targets_per_model":36,
          "localization_source_zip_sha256":sha(loczip),"requires":"Authorised phase3_swap/fallback patient data, phase4_finding weights and original project dependency scripts for reruns.",
          "incomplete_archived_task":"common_fov was superseded after semantic QC; released common-field experiments use common_clean."}
    entries["INDEX.json"]=json.dumps(meta,indent=2).encode()
    manifest={n:digest(b) for n,b in entries.items()}
    entries["SHA256.json"]=json.dumps(manifest,indent=2,sort_keys=True).encode()
    target=P/"diagnostic_evidence.zip";tmp=target.with_suffix(".part")
    with ZipFile(tmp,"w",ZIP_DEFLATED,compresslevel=4) as z:
        for k,b in sorted(entries.items()):z.writestr(k,b)
    with ZipFile(tmp) as z:
        assert z.testzip() is None and set(z.namelist())==set(entries)
        assert all(digest(z.read(k))==sig for k,sig in manifest.items())
    tmp.replace(target)
    print(json.dumps(dict(count=len(entries),nbytes=target.stat().st_size,sha256=sha(target),stages=count_by_stage)))
if __name__=="__main__":main()
