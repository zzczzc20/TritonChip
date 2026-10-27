#!/usr/bin/env python3
"""Storage destruction after a device failure must never open a new session."""
import ast
import json
import os
from pathlib import Path
import sys
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'python'),str(ROOT/'compiler/python'),str(ROOT/'tools/asm')]
from tnca.torch_device import DeviceMemory
from tnca.runtime import Buffer


def sessions_class():
    # Exercise the actual lazy session-cache definition without requiring a
    # Triton installation or contacting a device for a destructor regression.
    path=ROOT/'python/tnca/triton_backend/driver.py';module=ast.parse(path.read_text())
    node=next(n for n in module.body if isinstance(n,ast.ClassDef) and n.name=='_Sessions')
    namespace=dict(threading=threading,json=json,os=os,
        backend_choice=lambda:('fpga-u280-multicore',json.loads(os.environ.get('TNCA_BACKEND_ARGS','{}'))))
    exec(compile(ast.Module(body=[node],type_ignores=[]),str(path),'exec'),namespace)
    return namespace['_Sessions']


class StorageOwnerTests(unittest.TestCase):
    def memory(self, owner):
        memory=DeviceMemory();buf=Buffer(1,0x20000,64,1,0)
        owner._lock=threading.RLock()
        owner._heap=SimpleNamespace(free_list=[],limit=1<<20,next=0)
        owner.allocate=lambda size,name:buf
        memory.session=lambda index=0:owner
        self.assertEqual(memory.alloc(64),(buf.addr,1))
        def forbidden(*a):raise AssertionError('destructor attempted to open/resolve a new session')
        memory.session=forbidden
        return memory,buf

    def test_destructor_uses_original_owner_even_when_retirement_is_refused(self):
        called=[]
        class Owner:
            def retire(self,buf):
                called.append(buf)
                raise RuntimeError('poisoned session refuses retire')
        memory,buf=self.memory(Owner());memory.free(buf.addr,1)
        self.assertEqual(called,[buf]);self.assertEqual(memory.live,0)
        self.assertEqual(memory.allocations,{});self.assertEqual(memory.owners,{})
        memory.free(buf.addr,1)  # Idempotent even after the original failure.

    def test_old_destructor_cannot_retire_reused_address(self):
        retired=[];memory,old=self.memory(SimpleNamespace(retire=retired.append))
        fresh=Buffer(2,old.addr,64,2,0);memory.buffers[old.addr]=fresh
        memory.free(old.addr,1)
        self.assertIs(memory.buffers[old.addr],fresh);self.assertEqual(retired,[old])

    def test_poisoned_cached_session_does_not_open_another_device(self):
        cls=sessions_class();old=SimpleNamespace(closed=False,poisoned=True)
        key=('profile','fpga-u280-multicore','{}',None)
        cls.open[key]=old
        with patch.dict(os.environ,{},clear=True),patch('tnca.runtime.Session',side_effect=AssertionError('new device opened')):
            self.assertIs(cls.get('profile'),old)

    def test_backend_timeout_is_not_passed_twice(self):
        cls=sessions_class();cls.open={};self.addCleanup(cls.open.clear)
        env={'TNCA_BACKEND_ARGS':'{"host_timeout": 17, "deadline_cycles": 23}'}
        with patch.dict(os.environ,env,clear=True),patch('tnca.runtime.Session') as session:
            session.return_value=SimpleNamespace(closed=False)
            self.assertIs(cls.get('profile'),session.return_value)
        session.assert_called_once_with(profile='profile',counters=False,
            backend='fpga-u280-multicore',host_timeout=17,deadline_cycles=23)


if __name__=='__main__':unittest.main()
