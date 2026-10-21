import copy
import unittest

from dram_address_map import normalize, translate


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


if __name__=='__main__':
    unittest.main()
