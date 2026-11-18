"""Prepare an expanded patient cohort on the same 1.5 mm reference grid."""
from argparse import ArgumentParser
from pathlib import Path
import time,os
import atelectasis_prepare as prep
import thesis_transfer as resumable
from thesis_expand_screen import ROOT,SOURCES
from frozen_priors import read,write

def main():
    p=ArgumentParser();p.add_argument("--shard",type=int);p.add_argument("--shards",type=int,default=6)
    p.add_argument("--retry",action="store_true");a=p.parse_args()
    os.umask(0o077)
    prep.ROOT=ROOT
    prep.DATA=ROOT/"arrays";prep.RECEIPTS=ROOT/"receipts";prep.PANELS=ROOT/"technical_panels"
    resumable.ROOT=ROOT
    prep.obtain_file=resumable.download
    for d in (prep.DATA,prep.RECEIPTS,prep.PANELS):d.mkdir(exist_ok=True,parents=True)
    rows=read(SOURCES)[a.shard::a.shards]
    for i,row in enumerate(rows):
        code=row["code"]
        f=prep.RECEIPTS/(code+".json")
        if f.exists():continue
        for attempt in range(3):
            try:
                r=prep.convert(row,True)
                print(i,code,r["status"],r.get("reason",""),r.get("voxels",""),flush=True)
                break
            except Exception as e:
                print(i,code,"retry",attempt+1,type(e).__name__,str(e).split("http",1)[0][:45],flush=True)
                if attempt==2:write(prep.RECEIPTS/(code+".pending.json"),dict(code=code,error=type(e).__name__))
                else:time.sleep(3)
if __name__=="__main__":main()

