"""Run the frozen experiment with an equivalent component reduction."""
from pathlib import Path
import runpy
import sys
import thesis_model
import thesis_lobe_model
from thesis_component_fast import segment
from thesis_expand_screen import ROOT
from frozen_priors import read,sha

proof=read(ROOT/"component_equivalence.json")
assert proof["accelerated"]==sha(Path("thesis_component_fast.py"))
assert proof["original"]==sha(Path("thesis_model.py"))
thesis_model.segment=segment
# Reuse identical feature inputs across arms. Role checks still run on first load.
# The shared cache never stores a model or a pseudo-label.
shared_store=thesis_lobe_model.LobeStore(limit=64)
thesis_lobe_model.LobeStore=lambda **options:shared_store
sys.argv=sys.argv[1:]
runpy.run_path(sys.argv[0],run_name="__main__")

