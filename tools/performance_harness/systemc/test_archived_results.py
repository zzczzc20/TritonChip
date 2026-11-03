"""Archived model results remain readable without changing their contents."""
import gzip
import json
from pathlib import Path
import tempfile
import unittest

import predictions


class ArchivedResults(unittest.TestCase):
    def test_compressed_fallback_and_plain_file_precedence(self):
        with tempfile.TemporaryDirectory() as tmp:
            raw = Path(tmp) / 'results.json'
            compressed = raw.with_name(raw.name + '.gz')
            payload = {'points': [{'key': 'x', 'value': 7}]}
            compressed.write_bytes(gzip.compress(json.dumps(payload).encode(), mtime=0))
            actual, value = predictions.read_results(raw)
            self.assertEqual((actual, value), (compressed, payload))
            self.assertEqual(predictions.read_results(compressed), (compressed, payload))
            raw.write_text('{"points": []}', encoding='utf-8')
            self.assertEqual(predictions.read_results(raw), (raw, {'points': []}))


if __name__ == '__main__':
    unittest.main()
