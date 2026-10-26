#!/usr/bin/env python3
"""Host control refusals before writes and independent capability/readback checks."""
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'python'))
from tnca.backends.multicore_mmio import MulticoreMMIOBackend
from tnca.backends.u50 import U50Backend
from tnca.errors import ArgumentError,DeviceLost

class Registers:
    def __init__(self):
        self.cap=0x5234304d00010308
        self.config=0x308
        self.has_controls=True
        self.busy=0
        self.accept=True
        self.writes=[]
        self.reads=[]
    def describe(self):return "memory-control register fixture"
    def read_reg(self,offset):
        self.reads.append(offset)
        return {0xb8:32 if self.has_controls else 0,0x118:self.cap,0x110:self.config,0x48:self.busy}.get(offset,0)
    def write_reg(self,offset,value):
        self.writes.append((offset,value))
        if self.accept:self.config=value

def refused(operation,error):
    try:operation()
    except error:return
    raise AssertionError('operation was not refused')

def main():
    regs=Registers();backend=MulticoreMMIOBackend(regs)
    for ports,limit in [(1,1),(3,8),(1,8),(3,1)]:
        got=backend.configure_memory(active_ports=ports,outstanding=limit)
        assert got['active_ports']==ports and got['outstanding']==limit
        assert backend.capabilities()['memory_configuration']==got
    # Configuration audits belong outside the measured graph, not on every
    # launch. Existing error/timestamp reads are unchanged by this feature.
    backend.device_clock=False;regs.reads.clear()
    with patch.object(U50Backend,'launch_grid',return_value=SimpleNamespace(ok=True)):
        backend.launch_grid()
    assert not regs.reads, 'memory controls introduced per-launch register probes'
    for ports,limit in [(0,1),(2,8),(4,1),(3,0),(3,9),(1,True),(True,1),(1,1.5),(1,'1.0')]:
        before=list(regs.writes)
        refused(lambda:backend.configure_memory(active_ports=ports,outstanding=limit),ArgumentError)
        assert regs.writes==before
    before=list(regs.writes);regs.busy=1
    refused(lambda:backend.configure_memory(active_ports=1,outstanding=1),ArgumentError)
    assert regs.writes==before
    regs.busy=0;regs.accept=False
    refused(lambda:backend.configure_memory(active_ports=1,outstanding=8),DeviceLost)
    regs.accept=True;regs.has_controls=False;regs.cap=0
    assert backend.memory_configuration() is None
    refused(lambda:backend.configure_memory(active_ports=1,outstanding=1),ArgumentError)
    regs.has_controls=True
    for cap in [0,0x5234304d00020308,0x5234304d00010208,0x5234304d00010300]:
        regs.cap=cap;refused(backend.memory_configuration,DeviceLost)
    regs.cap=0x5234304d00010308
    for word in [0,0x208,0x309,0x10308]:
        regs.config=word;refused(backend.memory_configuration,DeviceLost)
    print('MEMORY CONTROL HOST PASS valid counts, no-write refusals, busy, stale readback and malformed capabilities')
if __name__=='__main__':main()
