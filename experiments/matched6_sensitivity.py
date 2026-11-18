"""Single-seed sensitivity using the locked training loop and fixed fields."""
import sys
from pathlib import Path
import torch
import matched6 as m

def main():
    original_rows=m.get_rows
    original_loss=m.loss_fn
    flag={}
    def rows(stage):
        result=original_rows(stage)
        for row in result:flag[id(row["m"])]=row["code"]
        return result
    def loss(z,y,mode):
        if flag[id(y)]=="S002":
            return original_loss(z[1:],y[1:],mode)
        return original_loss(z,y,mode)
    m.get_rows=rows;m.loss_fn=loss
    original=m.OUT
    assert m.sha(Path(m.__file__))==m.read(original/"protocol.json")["script"]
    m.OUT=original/"sensitivity";m.OUT.mkdir(exist_ok=True)
    m.seal(m.OUT/"protocol.json",dict(script=m.sha(Path(__file__)),base=m.sha(original/"protocol.json"),scope="same seed 891 and CT fields; omit only S002 A from the training loss"))
    if sys.argv[1]=="swap_dev":
        assert (original/"development_lock.json").exists()
        m.seal(m.OUT/"development_lock.json",dict(main=m.sha(original/"development_lock.json")))
    m.run(891,sys.argv[1])
if __name__=="__main__":main()

