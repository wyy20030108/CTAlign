"""Run the archived memorise function unchanged and retain its terminal weights, on a new output path."""
import inspect
import os
import sys
from pathlib import Path
import torch
import finding_native_check as orig

if __name__=="__main__":
    os.umask(0o077)
    dest=Path("phase5_diag/replay")/("unmodified_"+sys.argv[1] if len(sys.argv)>1 else ".")
    dest.mkdir(exist_ok=True,parents=True)
    configured_write=orig.write
    def save(path,data):
        assert path==orig.OUT/"native/overfit_control.json"
        configured_write(dest/"original_callable.json",data)
        frame=inspect.currentframe().f_back
        torch.save(frame.f_locals["net"].state_dict(),dest/"original_callable.pt")
    orig.write=save
    orig.configure(891)
    orig.memorise()
