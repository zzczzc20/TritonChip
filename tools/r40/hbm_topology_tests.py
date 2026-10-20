#!/usr/bin/env python3
"""Physical PC names/ranges come from image metadata, not address division."""
from pathlib import Path
import struct
import sys
import unittest
from unittest.mock import patch
sys.path[:0]=[str(Path(__file__).resolve().parents[2]/'python'),str(Path(__file__).resolve().parent)]
from tnca.backends import u50
from tnca.errors import DependencyFailed
from hbm_address import topology_probe_offsets


def topology(rows):
    return struct.pack('<i4x',len(rows))+b''.join(struct.pack('<BB6xQQ16s',6,1,kib,base,tag.encode()) for tag,base,kib in rows)


class TopologyTests(unittest.TestCase):
    def decode(self,rows,base,size):
        with patch.object(u50,'_xclbin_sections',return_value={u50._MEM_TOPOLOGY:topology(rows)}):
            return u50._xclbin_hbm_allocation('fixture',base,size)

    def test_opaque_pc_ids_and_partial_nonuniform_physical_ranges(self):
        base=1<<40
        result=self.decode([('HBM[7]',base,2),('HBM[23]',base+2048,3)],base+64,2048)
        self.assertEqual([b['pc'] for b in result['physical_banks']],[7,23])
        self.assertEqual([b['allocated_size'] for b in result['physical_banks']],[1984,64])
        self.assertEqual(len(result['topology_sha256']),64)

    def test_holes_overlaps_duplicates_and_unknown_names_are_uncertified(self):
        for rows in [[('HBM[7]',0,1),('HBM[23]',2048,1)],
                     [('HBM[7]',0,2),('HBM[23]',1024,2)],
                     [('HBM[7]',0,1),('HBM[7]',1024,1)],
                     [('memory',0,2)]]:
            with self.subTest(rows=rows),self.assertRaises(DependencyFailed):self.decode(rows,0,2048)

    def test_missing_and_truncated_topologies_are_refused(self):
        for sections in [{},{u50._MEM_TOPOLOGY:b''},{u50._MEM_TOPOLOGY:struct.pack('<i4x',3)}]:
            with patch.object(u50,'_xclbin_sections',return_value=sections),self.assertRaises(DependencyFailed):
                u50._xclbin_hbm_allocation('fixture',0,1024)

    def test_invalid_physical_ranges_are_refused(self):
        for base,size in [(-1,512),(0,0),(True,512),(1<<64,512)]:
            with self.assertRaises(DependencyFailed):self.decode([],base,size)

    def test_clamping_records_both_requested_and_exposed_capacity(self):
        link=u50.XDMALink.__new__(u50.XDMALink)
        link.xclbin='fixture';link.ip_name='kernel'
        class Accessor:
            def bo_open(self,size,bank):return 0
        link._accessor=Accessor()
        bank=dict(index=32,size=2048,base=0,tag='HBM[0:31]')
        with patch.object(u50,'_xclbin_bank',return_value=bank),patch.object(u50,'_xclbin_sections',return_value={u50._MEM_TOPOLOGY:topology([('HBM[7]',0,1),('HBM[23]',1024,1)])}):
            self.assertEqual(link.hbm_open(4096),(0,2048))
        self.assertEqual(link.hbm_allocation['requested_bytes'],4096)
        self.assertEqual(link.hbm_allocation['exposed_bytes'],2048)
        self.assertEqual([b['pc'] for b in link.hbm_allocation['physical_banks']],[7,23])

    def test_edge_and_cross_pc_probes_use_actual_boundaries(self):
        result=self.decode([('HBM[7]',1<<40,4),('HBM[23]',(1<<40)+4096,8)],1<<40,12288)
        window=dict(base=1<<40,size=12288,allocation=result)
        self.assertEqual(topology_probe_offsets(window),[0,4096-256,12288-512])
        window['allocation']['mapping_error']='unavailable'
        with self.assertRaises(ValueError):topology_probe_offsets(window)


if __name__=='__main__':unittest.main()
