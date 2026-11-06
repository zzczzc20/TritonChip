"""Archived model results remain readable without changing their contents."""
import gzip
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock
import sys

import directional
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

    def test_directional_report_records_actual_compressed_dataset(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); raw=root/'dataset.json'; compressed=root/'dataset.json.gz'
            compressed.write_bytes(gzip.compress(b'{"rows": []}', mtime=0))
            predictions_path=root/'predictions.jsonl'; predictions_path.write_text('')
            out=root/'report.json'
            argv=['directional','--dataset',str(raw),'--predictions',str(predictions_path),'--out',str(out)]
            with mock.patch.object(sys,'argv',argv):
                directional.main()
            report=json.loads(out.read_text())
            self.assertIn(str(compressed),report['inputs'])
            self.assertNotIn(str(raw),report['inputs'])



if __name__ == '__main__':
    unittest.main()
