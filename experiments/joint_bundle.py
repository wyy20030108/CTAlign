"""Archive the locked comparisons and their reproducibility evidence without source scans or reports."""
from pathlib import Path
import json
import os
from zipfile import ZipFile,ZIP_DEFLATED

from frozen_priors import read,sha
from joint_metrics import ARM
from joint_nodule_run import DEST
from joint_roi import ROOT
from thesis_ssl import SEEDS
from thesis_release import source_dependencies


def bundle():
    choose=read(DEST/"validation_selection.json")
    root=DEST/choose["chosen"]
    fixed=read(root/"external_lock.json")
    principal=read(root/"confirmed.json")
    assert principal["lock"]==sha(root/"external_lock.json")
    assert len(fixed["reference"])==22
    entries=[root/"confirmed.json",root/"protocol.json",root/"development_study.json",root/"external_lock.json",
             DEST/"validation_selection.json",DEST/"lock.json",DEST/"external_localization_summary.json",
             DEST/"report_size_calibration.json",DEST/"growth_summary.json",
             ROOT/"factorial_matched"/"confirmed.json",ROOT/"factorial_matched"/"development_study.json",
             ROOT/"factorial_direct"/"development_study.json",ROOT/"candidate_decision.json",ROOT/"spatial_summary.json"]
    for folder in ("factorial_matched","growth_matched","external_residual","external_matched","factorial_nosize"):
        p=DEST/folder/"development_study.json"
        if p.is_file():entries.append(p)
    auxiliary_development=DEST/"boundary_prototypes"/"development_study.json"
    if auxiliary_development.is_file():entries.append(auxiliary_development)
    for folder in (ROOT/"target_audit",DEST/"target_audit",DEST/"local_pseudo_diagnostic"):
        entries.extend(folder.glob("?.json"))
    for arm in ARM:
        for seed in SEEDS:
            folders=root/arm[0]/str(seed)/arm[1],root/"heldout"/arm/str(seed)
            training, held=folders
            entries.extend(training/name for name in ("model.pt","development.json"))
            if (training/"training.json").is_file():entries.append(training/"training.json")
            entries.extend(held.glob("*.json"));entries.extend(held.glob("*.npz"))
    extra=DEST/"external_proto_raw"
    if (extra/"confirmed.json").is_file():
        answer=read(extra/"confirmed.json")
        assert answer["lock"]==sha(extra/"evaluation_lock.json")
        entries.extend(extra.rglob("*.json"));entries.extend(extra.rglob("*.npz"))
        for seed in SEEDS:
            only=DEST/"external_proto_only"/"B"/str(seed)/"R"
            entries.extend(only/name for name in ("model.pt","development.json","training.json"))
    replication=ROOT/"nodule_replication"
    if (replication/"confirmed.json").is_file():
        self_lock=replication/"model_lock.json"
        result=read(replication/"confirmed.json")
        assert read(self_lock)["original"]==sha(root/"external_lock.json") and result["lock"]==sha(self_lock)
        entries.extend(replication/name for name in ("confirmed.json","model_lock.json","design.json","receipt.json"))
        entries.extend((replication/"scored").glob("Y???.*"))
    files={str(p.relative_to(ROOT)):p for p in entries}
    files.update({"code/"+p.name:p for p in source_dependencies(sorted(Path(".").glob("joint_*.py")))})
    archive=ROOT/"joint_segmentation_evidence.zip"
    part=archive.with_suffix(".part")
    with ZipFile(part,"w",ZIP_DEFLATED) as z:
        for name,p in files.items():z.write(p,name)
        z.writestr("SHA256.json",json.dumps({name:sha(p) for name,p in files.items()}))
    part.replace(archive)
    with ZipFile(archive) as z:
        assert z.testzip() is None
    print(archive,sha(archive),len(files),flush=True)


if __name__=="__main__":
    os.umask(0o077)
    bundle()
