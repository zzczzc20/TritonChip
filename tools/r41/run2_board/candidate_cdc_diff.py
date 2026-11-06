#!/usr/bin/env python3
"""Compare candidate CDC endpoints to the reviewed baseline and run-2.

This is a review aid, never a CDC disposition or authorization to load.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def endpoints(text):
    found = []
    for line in text.splitlines():
        match = re.match(r'\s*\d+\s+(CDC-\d+)\s+Critical\s', line)
        if match:
            fields = line.split()
            found.append((match.group(1), fields[-2], fields[-1]))
    if not found or len(found) != len(set(found)):
        raise ValueError('critical CDC endpoints are absent or duplicated')
    return set(found)


def compare(candidate, original, baseline, image):
    raw = json.loads(Path(baseline).read_text())['qualified_clock']['cdc_review_scope']['reviewed_critical']
    expected = {(r['rule'], r['source'], r['destination']) for r in raw}
    if len(raw) != len(expected):
        raise ValueError('baseline contains duplicate endpoints')
    current = endpoints(Path(candidate).read_text())
    old = endpoints(Path(original).read_text())
    def diff(reference):
        return dict(reference_count=len(reference), added=sorted(current-reference),
                    absent=sorted(reference-current))
    return dict(schema='tnca-candidate-cdc-comparison/1', status='needs-review',
                image_sha256=sha(image), cdc_report_sha256=sha(candidate),
                original_report_sha256=sha(original), baseline_sha256=sha(baseline),
                critical_count=len(current), against_reviewed_baseline=diff(expected),
                against_original_run2=diff(old))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('candidate', 'original', 'baseline', 'image', 'out'):
        p.add_argument('--'+name, type=Path, required=True)
    a = p.parse_args()
    data = compare(a.candidate, a.original, a.baseline, a.image)
    a.out.write_text(json.dumps(data, indent=2)+'\n')
    print('CDC comparison prepared; individual review is still required')


if __name__ == '__main__':
    main()
