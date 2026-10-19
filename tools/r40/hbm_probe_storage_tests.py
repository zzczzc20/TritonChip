#!/usr/bin/env python3
"""Exercise real first-fit allocation/reserve rules for exact BO-edge probes."""
from pathlib import Path
import sys
import unittest
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'python'),str(ROOT/'compiler/python'),str(ROOT/'tools/asm'),str(ROOT/'tools/r40')]
from tnca.runtime import _Allocator, DEVICE_RESERVED_RANGES
from hbm_address import topology_probe_storage, topology_probe_fill_sizes


class ProbeStorageTests(unittest.TestCase):
    def test_actual_allocator_keeps_frame_space_and_reaches_high_address(self):
        for size in (1<<30,4<<30):
            with self.subTest(size=size):
                local=131072;pcsize=256<<20
                window=dict(base=0,size=size,allocation=dict(topology_sha256='fixture',physical_banks=[
                    dict(pc=i,allocated_base=i*pcsize,allocated_size=pcsize) for i in range(size//pcsize)]))
                layout=topology_probe_storage(window);heap=_Allocator(local,local+size,reserved=DEVICE_RESERVED_RANGES)
                control=None;views=[]
                def tensor(nbytes):
                    free=sum(n for _,n in heap.free_list)+heap.limit-heap.next
                    self.assertGreaterEqual(free-nbytes,32768,'production launch reserve would refuse this tensor')
                    return heap.alloc(nbytes)
                for i,e in enumerate(layout):
                    top=heap.next
                    if i==len(layout)-1:heap.release(*control);control=None
                    gap=local+e['storage_start']-top
                    self.assertGreaterEqual(gap,0)
                    for fill_bytes in topology_probe_fill_sizes(top,local+e['storage_start'],heap.reserved):
                        tensor(fill_bytes)
                    addr,reserved=tensor(e['storage_bytes'])
                    views.append(addr+e['view_offset'])
                    self.assertEqual(views[-1],local+e['offset'])
                    if i==0:control=heap.alloc(65536)
                self.assertEqual(views[0],local)
                self.assertEqual(views[-1]+512,local+size)
                self.assertEqual(sum(n for _,n in heap.free_list)+heap.limit-heap.next,65536)
                frame=heap.alloc(32768);heap.release(*frame)
                if size==4<<30:self.assertGreaterEqual(views[-1],1<<32)

    def test_old_contiguous_filler_mutation_moves_the_physical_probe(self):
        lo,hi=DEVICE_RESERVED_RANGES[0]
        start=lo-4096;target=hi+8192
        heap=_Allocator(start,target+65536,reserved=DEVICE_RESERVED_RANGES)
        heap.alloc(target-start)
        moved,_=heap.alloc(512)
        self.assertNotEqual(moved,target,'old filler must reproduce the actual board assertion')
        heap=_Allocator(start,target+65536,reserved=DEVICE_RESERVED_RANGES)
        for size in topology_probe_fill_sizes(start,target,heap.reserved):heap.alloc(size)
        self.assertEqual(heap.alloc(512)[0],target)
        with self.assertRaises(ValueError):topology_probe_fill_sizes(start,lo+64,heap.reserved)

    def test_invalid_frame_and_overlapping_tail_are_refused(self):
        for frame in (0,-64,1):
            with self.assertRaises(ValueError):topology_probe_storage({},frame)
        w=dict(base=0,size=1024,allocation=dict(topology_sha256='fixture',physical_banks=[
            dict(pc=0,allocated_base=0,allocated_size=1024)]))
        with self.assertRaises(ValueError):topology_probe_storage(w)


if __name__=='__main__':unittest.main()
