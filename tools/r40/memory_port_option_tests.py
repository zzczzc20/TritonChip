#!/usr/bin/env python3
"""Outlet negotiation/refusal and counter parsing without simulator side effects."""
from pathlib import Path
import sys,threading
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'python'))
import triton
from tnca.transport import Transport,Reply
from tnca.errors import DeviceLost
from tnca.backends.multicore_file import MulticoreFileBackend

def fake(cap,metric=Reply(2,0,17,3)):
    dev=Transport.__new__(Transport);dev._io_lock=threading.RLock();dev.calls=[]
    def request(op,*args,**kwargs):dev.calls.append((op,args));return cap if op==16 else metric
    dev.request=request;return dev

def main():
    for count,version in ((0,1),(17,1),(3,2)):
        try:fake(Reply(1,0,count,version)).memory_port_capability()
        except DeviceLost:pass
        else:raise AssertionError('invalid outlet capability accepted')
    old=fake(Reply(1,3,0,0));assert old.memory_port_capability() is None
    try:old.memory_port_metrics()
    except ValueError:pass
    else:raise AssertionError('old image replaced by inferred counters')
    assert not any(op==17 for op,_ in old.calls)
    dev=fake(Reply(1,0,3,1));assert len(dev.memory_port_metrics())==3
    dev.memory_port_metrics();assert sum(op==16 for op,_ in dev.calls)==1
    for metric in (Reply(2,1,0,0),Reply(2,0,-1,0),Reply(2,0,1<<64,0)):
        try:fake(Reply(1,0,3,1),metric).memory_port_metrics()
        except DeviceLost:pass
        else:raise AssertionError('invalid counter reply accepted')
    for ports in (0,4,-1,1.5):
        try:MulticoreFileBackend(bench='not-used',core_count=3,memory_ports=ports)
        except (ValueError,TypeError):pass
        else:raise AssertionError('invalid port count accepted')
    print('MEMORY PORT OPTIONS PASS 12 cases',flush=True)
if __name__=='__main__':main()
