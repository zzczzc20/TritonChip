#!/usr/bin/env python3
"""Reject signed sizes before reaching a device or the unsigned XRT ABI."""
from pathlib import Path
import sys
import threading
from types import SimpleNamespace
import unittest
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'python')]
from tnca.runtime import Session,Buffer
from tnca.backends.u50 import _XRTRegisters
from tnca.errors import BufferError,ArgumentError


class BoundarySpy:
    def __init__(self):self.calls=[]
    def download(self,addr,nbytes):
        self.calls.append((addr,nbytes))
        if nbytes<0:raise AssertionError('negative length reached device boundary')
        return SimpleNamespace(ok=True),bytes(nbytes)
    def xclSyncBO(self,*args):
        self.calls.append(('sync',args[-2],args[-1]))
        if args[-2]<0:raise AssertionError('negative size would enter unsigned XRT ABI')
        return 0
    def xclReadBO(self,*args):self.calls.append(('read',args[-2],args[-1]));return 0


class CopyRangeTests(unittest.TestCase):
    def test_session_refuses_invalid_ranges_without_device_io(self):
        s=Session.__new__(Session);s.closed=s.poisoned=False;s.id=1;s.epoch=0
        s._lock=threading.RLock();s.dev=BoundarySpy();b=Buffer(1,131072,64,1,0)
        for offset,nbytes in [(0,-1),(64,-1),(0,-(1<<63)),(-1,1),(63,2),(65,0),(0,1<<64)]:
            s.dev.calls.clear()
            with self.subTest(offset=offset,nbytes=nbytes):
                with self.assertRaises(BufferError):s.download(b,nbytes,offset)
                self.assertEqual(s.dev.calls,[])
        s.dev.calls.clear()
        self.assertEqual(s.download(b,1,63),b'\x00')
        self.assertEqual(s.dev.calls,[(131135,1)])

    def test_xrt_refuses_invalid_ranges_before_any_c_call(self):
        x=_XRTRegisters.__new__(_XRTRegisters);x._bo_size=4<<30
        x._lib=BoundarySpy();x._handle=0;x._bo=1
        for offset,nbytes in [(0,-1),(4<<30,-1),(0,-(1<<63)),(-1,1),((4<<30)-1,2),(0,1<<64)]:
            x._lib.calls.clear()
            with self.subTest(offset=offset,nbytes=nbytes):
                with self.assertRaises(ArgumentError):x.bo_read(offset,nbytes)
                self.assertEqual(x._lib.calls,[])
        x._lib.calls.clear()
        self.assertEqual(x.bo_read((4<<30)-1,1),b'\x00')
        self.assertEqual(x._lib.calls,[('sync',1,(4<<30)-1),('read',1,(4<<30)-1)])


if __name__=='__main__':unittest.main()
