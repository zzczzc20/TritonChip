"""Read recorded JSON from its original path or a lossless gzip archive."""
import gzip
import json
from pathlib import Path


def read_json(path):
    """Return (actual source path, value), preferring an existing plain file."""
    actual = Path(path)
    if not actual.is_file() and actual.suffix != '.gz':
        compressed = actual.with_name(actual.name + '.gz')
        if compressed.is_file():
            actual = compressed
    if actual.suffix == '.gz':
        with gzip.open(actual, 'rt', encoding='utf-8') as stream:
            return actual, json.load(stream)
    return actual, json.loads(actual.read_text(encoding='utf-8'))
