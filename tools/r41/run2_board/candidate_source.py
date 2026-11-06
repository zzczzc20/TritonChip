#!/usr/bin/env python3
"""Verify the corrected frozen image source and bind it to a candidate hash."""
import argparse
import json
from pathlib import Path
import auto
import source_review

ARCHIVE_SHA = 'd64c78c9edb6c22b86a6a31cb8f19347cc87d2db20de5900896b3e95b54d3521'
COMMIT = '8d237223b9ab9cf6afc4228d6bc7149bf4a96086'


def review(archive, root, image, expected_image_sha):
    if auto.sha(image) != expected_image_sha:
        raise ValueError('candidate image hash changed')
    # Reuse the frozen archive's complete source-file comparison. This binds
    # the recorded candidate source to the image named by the build ledger;
    # it does not independently reproduce the Vivado build.
    source_review.ARCHIVE_SHA = ARCHIVE_SHA
    source_review.SOURCE_COMMIT = COMMIT
    old = auto.IMAGE_SHA
    try:
        auto.IMAGE_SHA = expected_image_sha
        result = source_review.review(archive, root)
    finally:
        auto.IMAGE_SHA = old
    result['scope'] = 'frozen corrected source; image association requires the retained Vivado build ledger'
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--archive', type=Path, required=True)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--image', type=Path, required=True)
    p.add_argument('--image-sha256', required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    result = review(a.archive, a.root, a.image, a.image_sha256)
    a.out.write_text(json.dumps(result, indent=2)+'\n')


if __name__ == '__main__':
    main()
