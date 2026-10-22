import copy
import unittest

from dram_address_map import normalize, plan, translate


def sample():
    # Deliberately offset BO, partial first/last banks, non-arithmetic PC IDs.
    return dict(base=320,size=384,local_limit=128,allocation=dict(exposed_bytes=384,topology_sha256='a'*64,
        physical_banks=[dict(pc=7,tag='HBM[7]',used=True,base=256,size=256,allocated_base=320,allocated_size=192),
                        dict(pc=2,tag='HBM[2]',used=True,base=512,size=256,allocated_base=512,allocated_size=192)]))


class AddressMapTests(unittest.TestCase):
    def test_actual_bo_translation_and_pc_boundaries(self):
        mapping=normalize(sample())
        self.assertEqual(translate(mapping,128),dict(pc=7,offset=64,physical_address=320))
        self.assertEqual(translate(mapping,256),dict(pc=7,offset=192,physical_address=448))
        self.assertEqual(translate(mapping,320),dict(pc=2,offset=0,physical_address=512))
        self.assertEqual(translate(mapping,448),dict(pc=2,offset=128,physical_address=640))
        for address in (0,64,449,512):
            with self.assertRaises(ValueError):translate(mapping,address)

    def test_bad_coverage_and_uncertified_tags_refused(self):
        changes=[lambda w:w['allocation']['physical_banks'][1].update(allocated_base=576),
                 lambda w:w['allocation']['physical_banks'][1].update(allocated_base=448),
                 lambda w:w['allocation']['physical_banks'][0].update(tag='HBM[0]'),
                 lambda w:w['allocation']['physical_banks'][0].update(used=False),
                 lambda w:w['allocation'].update(exposed_bytes=256),
                 lambda w:w['allocation'].update(mapping_error='uncertified'),
                 lambda w:w.update(base=True),lambda w:w.update(base=(1<<64)-64),
                 lambda w:w['allocation']['physical_banks'][1].update(pc=7,tag='HBM[7]')]
        for change in changes:
            window=copy.deepcopy(sample());change(window)
            with self.subTest(window=window),self.assertRaises(ValueError):normalize(window)


def recorded_group():
    # A recorded 512-byte BO over PC4-PC5 of a used MBG[4:7] group (4 x 256 bytes).
    return dict(base=1024,size=512,local_limit=128,allocation=dict(exposed_bytes=512,topology_sha256='b'*64,
        connected_memory=dict(tag='MBG[4:7]',used=1,base=1024,size=1024),
        physical_banks=[dict(pc=4,tag='HBM[4]',used=True,base=1024,size=256,allocated_base=1024,allocated_size=256),
                        dict(pc=5,tag='HBM[5]',used=True,base=1280,size=256,allocated_base=1280,allocated_size=256)]))


class PlannedMapTests(unittest.TestCase):
    def test_planned_bo_extends_the_recorded_group_rule(self):
        recorded,mapping=plan(recorded_group(),1024)
        self.assertEqual(recorded['bo_size'],512)
        self.assertEqual([b['pc'] for b in mapping['banks']],[4,5,6,7])
        self.assertEqual((mapping['local_limit'],mapping['bo_base'],mapping['bo_size'],mapping['topology_sha256']),(128,1024,1024,None))
        self.assertEqual(translate(mapping,128+512),dict(pc=6,offset=0,physical_address=1536))
        self.assertEqual(translate(mapping,128+1024-64),dict(pc=7,offset=192,physical_address=2048-64))
        for address,nbytes in ((128+1024,1),(128+1024-63,64),(64,64)):
            with self.assertRaises(ValueError):translate(mapping,address,nbytes)
        _,partial=plan(recorded_group(),640)
        self.assertEqual(partial['banks'][-1],dict(pc=6,base=1536,size=256,allocated_base=1536,allocated_size=128))

    def test_planned_bo_refusals(self):
        changes=[lambda w:w['allocation']['connected_memory'].update(tag='HBM[4]'),
                 lambda w:w['allocation']['connected_memory'].update(used=0),
                 lambda w:w['allocation']['connected_memory'].update(size=768),
                 lambda w:w['allocation']['physical_banks'][1].update(pc=6,tag='HBM[6]'),
                 lambda w:w['allocation'].update(mapping_error='uncertified')]
        for change in changes:
            window=copy.deepcopy(recorded_group());change(window)
            with self.subTest(window=window),self.assertRaises(ValueError):plan(window,1024)
        for size in (0,1056,1032):
            with self.subTest(size=size),self.assertRaises(ValueError):plan(recorded_group(),size)


if __name__=='__main__':
    unittest.main()
