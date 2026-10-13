#!/usr/bin/env python3
"""Physical-grid admission and packet identity, without device side effects."""
from pathlib import Path
import sys
import threading
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'python'))
import triton
from tnca.transport import Transport,Reply
from tnca.errors import DeviceLost

def fake(status=0,cores=3,limit=16777216):
    dev=Transport.__new__(Transport);dev.calls=[];dev._io_lock=threading.RLock()
    def request(op,*args,**kwargs):
        dev.calls.append((op,args))
        return Reply(len(dev.calls),status,cores,limit) if op==10 else Reply(len(dev.calls),0,17,0)
    dev.request=request
    return dev

def rejects(dev,error,**kwargs):
    try: dev.launch_grid(**dict(dict(pc=0,arg_base=0x20000,grid=(13,7,5)),**kwargs))
    except error: pass
    else: raise AssertionError('invalid request accepted')
    assert not any(op==11 for op,_ in dev.calls)

def main():
    rejects(fake(status=3),ValueError)
    rejects(fake(cores=0),DeviceLost)
    rejects(fake(),TypeError,grid=(1.5,1,1))
    rejects(fake(),ValueError,grid=(1<<32,1,1))
    rejects(fake(),ValueError,pc=1)
    rejects(fake(),ValueError,arg_stride=-8)
    rejects(fake(),ValueError,active_cores=8)
    rejects(fake(),ValueError,role_mask=16)
    rejects(fake(),ValueError,grid=(16777217,1,1))
    dev=fake();reply=dev.launch_grid(0x80,0x20000,(13,7,5),arg_stride=256,active_cores=5)
    assert reply.ok and dev.calls[-1]==(11,(128,131072,1,13|(7<<32),5|(5<<32),256))
    dev.launch_grid(0,0x20000,(1<<31,0,1))
    assert sum(op==10 for op,_ in dev.calls)==1
    assert dev.calls[-1][1][3]==1<<31
    print('physical-grid-option-tests PASS (11 cases)',flush=True)
if __name__=='__main__':main()
