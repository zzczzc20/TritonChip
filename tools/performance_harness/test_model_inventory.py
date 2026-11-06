"""Reject malformed checkpoint headers before deriving a capacity estimate."""
import unittest
from model_inventory import validate_header


def tensor(dtype='BF16', shape=None, offsets=None):
    return dict(dtype=dtype, shape=[2, 3] if shape is None else shape,
                data_offsets=[0, 12] if offsets is None else offsets)


class HeaderTests(unittest.TestCase):
    def test_distinct_storage_and_exact_alias(self):
        result = validate_header(dict(a=tensor(), tied=tensor(),
                                      b=tensor('F32', [2], [12, 20])), payload_bytes=20)
        self.assertEqual(result, dict(tensors=3, distinct_storages=2, distinct_bytes=20))

    def test_bad_dtype_shape_offsets_and_partial_aliases(self):
        cases = [dict(a=tensor(shape=[2, 4])),
                 dict(a=tensor(dtype='UNKNOWN')),
                 dict(a=tensor(shape=[True, 6])),
                 dict(a=tensor(offsets=[-2, 10])),
                 dict(a=tensor(), b=tensor(offsets=[6, 18])),
                 dict(a=tensor(), b=tensor('F32', [3], [0, 12]))]
        for header in cases:
            with self.subTest(header=header), self.assertRaises(ValueError):
                validate_header(header)

    def test_payload_size_is_an_independent_bound(self):
        for size in (11, 13):
            with self.subTest(size=size), self.assertRaises(ValueError):
                validate_header(dict(a=tensor()), payload_bytes=size)

    def test_empty_tensor_does_not_create_storage(self):
        self.assertEqual(validate_header(dict(a=tensor(shape=[0, 3], offsets=[0, 0])),
                                         payload_bytes=0)['distinct_bytes'], 0)


if __name__ == '__main__':
    unittest.main()
