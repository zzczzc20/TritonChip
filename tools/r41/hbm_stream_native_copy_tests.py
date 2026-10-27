#!/usr/bin/env python3
"""Real native Torch copy dispatch through a register/byte-memory fixture.

This tests the software path, not FPGA execution or measured bandwidth.
"""
from pathlib import Path
import os
import sys
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'python'),str(ROOT),str(ROOT/'compiler/python'),str(ROOT/'tools/asm'),str(Path(__file__).resolve().parent)]
os.environ.setdefault('MAX_JOBS','2')
import torch
import triton
import tnca.torch_device as D
from tnca.torch_device import copy as C
from tnca.triton_backend.driver import launch_log, _bench_local
from tnca.hbm_layout import StripedHBMLayout, STRIPED_BYTES
from hbm_stream_host_tests import RegisterBank,backend,session
from types import SimpleNamespace


def main():
    torch.set_num_threads(2)
    bank=RegisterBank();owner=session(backend(bank))
    owner.profile='core-vrf-sync-v5-x8';owner.backend_name='register-byte-fixture'
    layout=StripedHBMLayout(0,STRIPED_BYTES)
    transfers=[]
    def upload(address,data):
        transfers.append(('upload',address,len(data)))
        layout.write(bank.memory,address-owner.dev.local_limit,data)
        return SimpleNamespace(ok=True)
    def download(address,nbytes):
        transfers.append(('download',address,nbytes))
        return SimpleNamespace(ok=True),layout.read(bank.memory,address-owner.dev.local_limit,nbytes)
    owner.dev.upload=upload;owner.dev.download=download
    with patch.object(D.MEMORY,'session',return_value=owner):
        D.register()
        host=torch.arange((32<<20)//4,dtype=torch.int32)
        source=host.to('tnca');destination=torch.empty_like(source)
        assert source.data_ptr()%(1<<17)==destination.data_ptr()%(1<<17)==0
        transfers.clear();bank.events.clear()
        # A fallback would enter Triton code here; the fast path must not.
        with patch.object(C,'_copy_kernel',side_effect=AssertionError('unexpected kernel fallback')):
            with launch_log() as launches:
                destination.copy_(source)
        assert not transfers,transfers
        assert len(launches)==1 and launches[0]['operation_kind']=='hbm-stream'
        assert launches[0]['device_cycles']==owner.latest_event.device_cycles>0
        assert launches[0]['upload_bytes']==launches[0]['download_bytes']==0
        assert launches[0]['artifact'] is None and launches[0]['device_timing'] is None
        assert owner.latest_event.ok and owner.last_stream_result['read_bytes']==32<<20
        assert torch.equal(destination.cpu(),host)
        print('NATIVE TORCH 32 MiB COPY PASS: real tensors, stream event/log, no host transfers inside copy',flush=True)

        # Storage resize copy reaches the same production path.
        transfers.clear()
        C.copy_storage(D.MEMORY,destination.data_ptr(),source.data_ptr(),32<<20)
        assert not transfers
        assert torch.equal(destination.cpu(),host)
        print('NATIVE STORAGE COPY PASS',flush=True)

        transfers.clear();before=len(bank.events)
        with patch('tnca.torch_device.graph.active_capture',return_value=object()):
            assert not C._try_stream_copy(D.MEMORY,destination.data_ptr(),source.data_ptr(),32<<20)
        with patch.dict(os.environ,{'TNCA_LAUNCH':'skip'}):
            assert not C._try_stream_copy(D.MEMORY,destination.data_ptr(),source.data_ptr(),32<<20)
        with patch.object(_bench_local,'compile_only',True,create=True):
            assert not C._try_stream_copy(D.MEMORY,destination.data_ptr(),source.data_ptr(),32<<20)
        assert not C._try_stream_copy(D.MEMORY,destination.data_ptr()+64,source.data_ptr(),32<<20)
        assert not C._try_stream_copy(D.MEMORY,destination.data_ptr(),source.data_ptr(),64)
        assert len(bank.events)==before and not transfers
        print('NATIVE CAPTURE/COMPILE-ONLY/SHAPE FALLBACK GUARDS PASS',flush=True)
        del source,destination
    print('NATIVE STREAM SOFTWARE INTEGRATION PASS (register fixture, not board evidence)',flush=True)


if __name__=='__main__':main()
