#!/usr/bin/env python3
"""A refused ownership transfer must never reach BO read/write."""
from pathlib import Path
import sys
from unittest.mock import Mock
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'python'))
from tnca.backends.multicore_mmio import MulticoreMMIOBackend,OP_CACHE_HANDOFF
from tnca.transport import Reply
from tnca.errors import DeviceLost


def main():
    passed=0
    for operation in ('read','write'):
        for status,fault in ((1,False),(3,False),(0,True)):
            link=Mock();link.words=(0x20000+4096)//8
            dev=MulticoreMMIOBackend(link,hbm_dma=True)
            dev.local_limit=0x20000;dev.memory_bytes=0x21000
            dev.request=Mock(return_value=Reply(1,status,0,0))
            dev.axi_status=Mock(return_value={'error':fault,'stray':False})
            try:
                if operation=='write':dev._put_payload(0x20000,[1])
                else:dev._get_payload(0x20000,1)
            except DeviceLost:pass
            else:raise AssertionError('failed handoff was accepted')
            link.hbm_read.assert_not_called();link.hbm_write.assert_not_called()
            dev.request.assert_called_once_with(OP_CACHE_HANDOFF,0x20000,8)
            passed+=1
    print(f'DMA HANDOFF GUARDS PASS ({passed}/{passed})',flush=True)


if __name__=='__main__':main()
