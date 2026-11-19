"""Count explicit side words directly linked to a pleural-fluid phrase, without exporting clinical text."""
import html
from pathlib import Path
import re
from frozen_priors import roster, read, sha, write
from replay_controls import read_reports


SIDE=r"(?P<side>bilateral(?:ly)?|both sides|both|left(?:[- ]sided)?|right(?:[- ]sided)?)"
TARGET=r"(?:pleural (?:effusions?|fluid(?: collections?)?)|hydrothorax)"
MOD=r"(?:small|mild|moderate|large|minimal|trace|loculated|free|layering|sided)"
LEFT=re.compile(r"\b"+SIDE+r"\s+(?:"+MOD+r"\s+){0,2}"+TARGET+r"\b")
RIGHT=re.compile(r"\b"+TARGET+r"\s+(?:(?:is|are|was|were|seen|noted|observed|detected|present|found|located|identified|visible|on|at|in|the)\s+){0,6}"+SIDE+r"\b")


def cues(report):
    result=set()
    for section in report.values():
        s=" ".join(html.unescape(section).lower().split())
        for pattern in (LEFT,RIGHT):
            for item in pattern.finditer(s):
                v=item["side"]
                result.add("B" if v.startswith(("both","bilat")) else v[0].upper())
    return sorted(result)


def main():
    for phrase,result in [("small bilateral pleural effusions",["B"]),("pleural effusion on the right",["R"]),
                          ("left renal cyst. no pleural effusion",[]),("left-sided pericardial effusion",[]),
                          ("no left pleural effusion",["L"]),("right pleural thickening",[])]:
        assert cues({"test":phrase})==result
    root=Path.cwd()
    manifest=read(Path("phase1/data/U_report_strata32/manifest.private.json"))["rows"]
    allowed={r[1] for r in roster(root)}
    u={r["name"]:r["stratum"] for r in manifest}
    frozen=read(Path("phase1/donor_support/training_donors.private.json"))
    donors={r["volume"]:r["tiers"]["coarse"]["selected"] for r in frozen if r["volume"] in u}
    docs=read_reports(allowed|{n for vv in donors.values() for n in vv})
    side={n:cues(v) for n,v in docs.items()}
    def normalized(doc):
        return tuple(" ".join(doc[key].lower().split()) for key in ("Findings_EN","Impressions_EN"))
    probe=read(Path("phase2/bounded/891/teacher_sensitivity.json"))
    by_code={r["code"]:r for r in probe["cases"]}
    cases=[]
    for code,name,_,_ in roster(root):
        tier=u.get(name)
        row={"code":code,"lexical_cues":side[name],"stratum":tier}
        if code.startswith("U"):
            peers=donors[name]
            comparable=[other for other in peers if side[other] and side[name]]
            row.update(n_donors=len(peers),both_cued_donors=len(comparable),
                       both_cued_different_set=sum(side[d]!=side[name] for d in comparable),
                       identical_normalized_text=sum(normalized(docs[d])==normalized(docs[name]) for d in peers),
                       sensitivity_probe_eligible=by_code[code]["scored"],uncertain_band_voxels=by_code[code]["band_voxels"])
        cases.append(row)
    us=[r for r in cases if r["stratum"] is not None]
    ts=[r for r in cases if r["stratum"] is None]
    vis=[r for r in us if r["sensitivity_probe_eligible"]]
    def summarize(rows):
        return {"cases":len(rows),"with_explicit_side":sum(bool(r["lexical_cues"]) for r in rows),
                "with_at_least_one_donor":sum(r["n_donors"]>0 for r in rows),
                "with_any_two_sided_text_comparison":sum(r["both_cued_donors"]>0 for r in rows),
                "with_any_different_explicit_sets":sum(r["both_cued_different_set"]>0 for r in rows),
                "distinct_recipient_donor_pairs":sum(r["n_donors"] for r in rows),
                "pairs_both_explicit":sum(r["both_cued_donors"] for r in rows),
                "pairs_different_sets":sum(r["both_cued_different_set"] for r in rows),
                "pairs_identical_normalized_report":sum(r["identical_normalized_text"] for r in rows)}
    result={"rule":"Case-insensitive direct lexical modifier before pleural effusion/fluid or hydrothorax, or target followed by linking words and side. Unique L/R/B cue sets across findings and impressions, regardless of negation or tense. Missing cue is unknown. Different sets are lexical differences, not certified clinical disagreement. Other phrasing and other patient attributes are not assessed.",
            "training_T":{"cases":len(ts),"direct_side_cue":sum(bool(r["lexical_cues"]) for r in ts)},
            "all_U":summarize(us),"sensitivity_subset":summarize(vis),"by_report_state":{s:summarize([r for r in us if r["stratum"]==s]) for s in sorted(set(u.values()))},
            "cases":cases,"script_sha256":sha(Path(__file__))}
    write(Path("phase2/laterality.json"),result)
    print({k:v for k,v in result.items() if k in ("training_T","all_U","sensitivity_subset")})


if __name__=="__main__":
    main()
