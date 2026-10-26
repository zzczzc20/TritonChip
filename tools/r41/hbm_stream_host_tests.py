#!/usr/bin/env python3
"""Host command, ownership, peer-lifetime and Session checks; no board claims."""
from pathlib import Path
import collections
import ctypes
import random
import struct
import sys
import threading
import unittest
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT/'python'),str(Path(__file__).resolve().parent)]
from tnca.backends.hbm_stream import StreamController, descriptors, MAGIC, CAPABILITY
from tnca.backends.multicore_mmio import MulticoreMMIOBackend
from tnca.backends.u50 import _XRTRegisters
from tnca.hbm_layout import StripedHBMLayout, STRIPED_BYTES
from tnca.runtime import Session, Buffer, _Allocator
from tnca.errors import ArgumentError, DeviceLost, BufferError, BufferBusy, CapabilityError, DependencyFailed
from hbm_layout_tests import Accessor


class RegisterBank:
    """Test-only byte memory and synchronous CU register fixture."""
    def __init__(self):
        self.memory = Accessor()
        self.events = []
        self.regs = {0:MAGIC,8:CAPABILITY,0x20:0,0x28:0}
        self.lanes = [[0,0,0] for _ in range(32)]
        self.counts = [(0,0,0) for _ in range(32)]
        self.selected = 0
        self.hold_busy = False
        self.bad_counter = False
        self.bad_readback = False

    def read(self, address):
        if address in (0x58,0x60,0x68):return self.lanes[self.selected][(address-0x58)//8]
        if address in (0xb0,0xb8,0xa8):return self.counts[self.selected][(0xb0,0xb8,0xa8).index(address)]
        return self.regs.get(address,0)

    def write(self, address, value):
        self.events.append(('cu-write',address,value))
        if self.bad_readback and address==0x40:return
        self.regs[address] = value
        if address==0x50:self.selected=value
        if address in (0x58,0x60,0x68):self.lanes[self.selected][(address-0x58)//8]=value
        if address!=0x18:return
        self.regs[0x20] = 1
        if self.hold_busy:return
        mode = self.regs[0x40] >> 32
        pattern = self.regs[0x40] & 0xffffffff
        total_reads=total_writes=result=0
        for pc,(source,destination,lines) in enumerate(self.lanes):
            if not self.regs[0x48] & (1<<pc):continue
            n = lines*64
            nr,nw = (0 if mode==1 else lines),(0 if mode==2 else lines)
            data = b'' if mode==1 else self.memory.bo_read((pc<<28)+source,n)
            if mode==1:data=pattern.to_bytes(4,'little')*(n//4)
            if mode!=2:self.memory.bo_write((pc<<28)+destination,data)
            else:result=(result+sum(x[0] for x in struct.iter_unpack('<I',data))) & 0xffffffff
            self.counts[pc]=(nr,nw,max(nr,nw)+40+pc)
            total_reads+=nr;total_writes+=nw
        self.regs.update({0x20:2,0x28:self.regs[0x10],0x30:0,
                          0x38:max(p[2] for p in self.counts)+35,
                          0x78:total_reads+int(self.bad_counter),0x80:total_writes,0x70:result})


def backend(bank):
    events = bank.events
    class Core:
        words = (STRIPED_BYTES+131072)//8
        busy = False
        def read_reg(self,address):
            events.append(('core-read',address))
            return {0x48:self.busy,0xa8:0,0xb0:STRIPED_BYTES,0xb8:(32<<32)|82}[address]
        def alive(self):return True
    device = MulticoreMMIOBackend(Core())
    device.opened=True;device.local_limit=131072;device._ownership_cap=True
    device._stream=StreamController(bank,timeout=.01,poll_interval=0)
    def request(op,*args,**kwargs):
        events.append(('request',op,args))
        return SimpleNamespace(ok=True)
    device.request=request
    return device


def session(device):
    result = Session.__new__(Session)
    result.closed=result.poisoned=False;result.id=1;result.epoch=0;result.dev=device
    result._lock=threading.RLock();result._events=collections.OrderedDict();result._next_job=0
    result._next_buffer=0;result._buffers={};result._heap=_Allocator(131072,STRIPED_BYTES+131072)
    return result


class HostTests(unittest.TestCase):
    def test_analytic_pc_ranges_match_gather_groups(self):
        randomizer=random.Random(41)
        for _ in range(100):
            size=randomizer.randrange(0,200000)
            offset=randomizer.randrange(0,STRIPED_BYTES-size)
            actual=StripedHBMLayout.pc_ranges(offset,size)
            expected=sorted((start>>28,start,n) for start,n,_ in StripedHBMLayout.groups(offset,size))
            self.assertEqual(actual,expected)
        self.assertEqual(StripedHBMLayout.pc_ranges(0,STRIPED_BYTES),[(pc,pc<<28,1<<28) for pc in range(32)])

    def test_real_driver_copy_fill_and_wrapping_sum(self):
        bank=RegisterBank();driver=StreamController(bank,poll_interval=0)
        layout=StripedHBMLayout(0,STRIPED_BYTES)
        n=(1<<17)+64;data=random.Random(41).randbytes(n)
        layout.write(bank.memory,0,data)
        handoffs=[]
        copied=driver.execute(0,0,1<<20,n,handoff=lambda *r:handoffs.append(r))
        self.assertEqual(layout.read(bank.memory,1<<20,n),data)
        self.assertEqual((copied['read_bytes'],copied['write_bytes']),(n,n))
        self.assertEqual(len(copied['lanes']),32)
        self.assertEqual(handoffs,[(0,n),(1<<20,n)])
        pattern=0xfedcba98
        driver.execute(1,0,1<<20,n,pattern=pattern,handoff=lambda *r:None)
        self.assertEqual(layout.read(bank.memory,1<<20,n),pattern.to_bytes(4,'little')*(n//4))
        reduced=driver.execute(2,1<<20,0,n,handoff=lambda *r:None)
        self.assertEqual(reduced['result_sum'],pattern*(n//4)&0xffffffff)
        self.assertEqual(reduced['sequence'],3)

    def test_invalid_descriptors_have_no_ownership_or_cu_writes(self):
        bank=RegisterBank();driver=StreamController(bank)
        for args in [(0,0,64,64),(0,0,0,64),(0,0,1<<20,63),(2,1,0,64),(2,STRIPED_BYTES-64,0,128),
                     (3,0,0,64),(2,0,0,-64),(2,False,0,64)]:
            bank.events.clear()
            with self.subTest(args=args),self.assertRaises(ArgumentError):
                driver.execute(*args,handoff=lambda *r:self.fail('invalid command handed ownership off'))
            self.assertEqual(bank.events,[])
        with self.assertRaises(ArgumentError):
            driver.execute(1,0,0,64,pattern=-1,handoff=lambda *r:self.fail('bad pattern handed ownership off'))
        self.assertEqual(bank.events,[])

    def test_fault_timeout_and_counter_mismatch_cannot_retry_or_forge_success(self):
        for reason in ('fault','timeout','counter'):
            bank=RegisterBank();driver=StreamController(bank,timeout=.002,poll_interval=0)
            if reason=='fault':bank.regs[0x20]=4
            if reason=='timeout':bank.hold_busy=True
            if reason=='counter':bank.bad_counter=True
            with self.subTest(reason=reason),self.assertRaises(DeviceLost):
                driver.execute(1,0,0,64,handoff=lambda *r:None)
            self.assertIsNone(driver.last_result)
            before=len(bank.events)
            with self.assertRaises(DeviceLost):driver.execute(1,0,0,64,handoff=lambda *r:None)
            self.assertEqual(len(bank.events),before)

    def test_configuration_refusal_never_rings_doorbell(self):
        bank=RegisterBank();driver=StreamController(bank);bank.bad_readback=True
        with self.assertRaises(DeviceLost):driver.execute(1,0,0,64,handoff=lambda *r:None)
        self.assertFalse(any(e[1]==0x18 for e in bank.events))

    def test_backend_serializes_core_drain_and_both_range_handoffs(self):
        bank=RegisterBank();device=backend(bank);bank.events.clear()
        result=device.stream_operation(0,131072,131072+(1<<20),1<<17)
        requests=[event for event in bank.events if event[0]=='request']
        self.assertEqual(requests,[('request',4,()),('request',18,(131072,1<<17)),('request',18,(131072+(1<<20),1<<17))])
        first_write=next(i for i,e in enumerate(bank.events) if e[0]=='cu-write')
        self.assertEqual(len([e for e in bank.events[:first_write] if e[0]=='request']),3)
        self.assertEqual(result['read_bytes'],1<<17)

    def test_backend_reserved_ranges_and_bad_pattern_refuse_before_device_io(self):
        bank=RegisterBank();device=backend(bank);bank.events.clear()
        with self.assertRaises(ArgumentError):device.stream_operation(1,0,0x7fff0000,64)
        with self.assertRaises(ArgumentError):device.stream_operation(1,0,131072,64,pattern=1<<32)
        self.assertEqual(bank.events,[])

    def test_session_events_alignment_buffers_and_poison(self):
        bank=RegisterBank();device=backend(bank);owner=session(device)
        source=owner.allocate(1<<20);destination=owner.allocate(1<<20)
        self.assertEqual(source.addr%(1<<17),0);self.assertEqual(destination.addr%(1<<17),0)
        result=owner.stream_fill(source,0x12345678)
        self.assertTrue(owner.latest_event.ok)
        self.assertEqual(owner.latest_event.device_cycles,result['cycles'])
        self.assertIsNone(owner.latest_event.device_timing)
        self.assertTrue(owner.try_stream_copy(source,destination,1<<20))
        self.assertEqual(owner.stream_reduce_u32(destination)['result_sum'],0x12345678*((1<<20)//4)&0xffffffff)
        bank.events.clear();count=len(owner._events)
        with self.assertRaises(ArgumentError):owner.stream_fill(destination,-1)
        with self.assertRaises(BufferError):owner.stream_copy(source,destination,destination.size+64)
        self.assertEqual(len(owner._events),count);self.assertEqual(bank.events,[])
        source.leases=1
        with self.assertRaises(BufferBusy):owner.stream_copy(source,destination,64)
        source.leases=0
        bank.bad_counter=True
        with self.assertRaises(DeviceLost):owner.stream_fill(destination,0)
        self.assertTrue(owner.poisoned);self.assertFalse(destination.valid);self.assertFalse(owner.latest_event.ok)
        self.assertFalse(device.alive())

    def test_graph_capture_and_unsupported_native_copy_keep_original_path(self):
        bank=RegisterBank();owner=session(backend(bank))
        source=owner.allocate(1<<20);destination=owner.allocate(1<<20)
        bank.events.clear()
        with patch.dict(sys.modules,{'tnca.torch_device.graph':SimpleNamespace(active_capture=lambda:object())}):
            with self.assertRaises(CapabilityError):owner.stream_fill(destination,0)
        self.assertFalse(owner.try_stream_copy(source,destination,65))
        owner.dev._stream=None
        self.assertFalse(owner.try_stream_copy(source,destination,64))
        self.assertEqual(bank.events,[])

    def test_peer_context_reuses_handle_and_closes_before_parent(self):
        calls=[]
        class Library:
            def xclIPName2Index(self,handle,name):calls.append(('lookup',handle,name));return 9
            def xclOpenContext(self,handle,uuid,ip,shared):calls.append(('open-context',handle,ip));return 0
            def xclCloseContext(self,handle,uuid,ip):calls.append(('close-context',handle,ip));return 0
            def xclClose(self,handle):calls.append(('close-handle',handle))
        parent=_XRTRegisters();parent._handle=123;parent._lib=Library();parent._ip=4
        parent._uuid=(ctypes.c_ubyte*16)(*range(16))
        with patch('tnca.backends.u50._loaded_xclbin_uuid',return_value=bytes(range(16))):
            peer=parent.open_peer('tnca_stream:tnca_stream_1')
        self.assertEqual(peer._handle,123);self.assertFalse(peer._owns_handle)
        parent.close();parent.close();peer.close()
        self.assertEqual([c for c in calls if c[0].startswith('close')],
                         [('close-context',123,9),('close-context',123,4),('close-handle',123)])

    def test_peer_refusal_keeps_parent_context(self):
        parent=_XRTRegisters();parent._handle=123;parent._ip=4
        parent._uuid=(ctypes.c_ubyte*16)(*range(16))
        calls=[]
        parent._lib=SimpleNamespace(xclIPName2Index=lambda *a:9,xclOpenContext=lambda *a:1)
        with patch('tnca.backends.u50._loaded_xclbin_uuid',return_value=bytes(16)):
            with self.assertRaises(DeviceLost):parent.open_peer('stream')
        with patch('tnca.backends.u50._loaded_xclbin_uuid',return_value=bytes(range(16))):
            with self.assertRaises(DependencyFailed):parent.open_peer('stream')
        self.assertEqual(parent._handle,123);self.assertEqual(parent._ip,4);self.assertEqual(parent._peers,[])


if __name__=='__main__':unittest.main()
