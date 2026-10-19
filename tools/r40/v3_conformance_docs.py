#!/usr/bin/env python3
"""Package one completed host-conformance run for device_conformance_acceptance.py.

    v3_conformance_docs.py --run-dir DIR --stage NAME --host-class cuda-host|no-cuda-host \
        --backend B --launch-controller-sha256 SHA --live-source-tree TREE --out DOC.json

DIR is a copy of the controller directory (controller.py, state.json and the
stage's NAME.log). Nothing is summarized: the raw controller, state and log go
into the document with their hashes, in the format the auditor reviews.
"""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'tools/r40'))
import contract_v2


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-dir', type=Path, required=True)
    p.add_argument('--stage', required=True)
    p.add_argument('--host-class', choices=('cuda-host', 'no-cuda-host'), required=True)
    p.add_argument('--backend', required=True)
    p.add_argument('--launch-controller-sha256', required=True)
    p.add_argument('--live-source-tree', required=True, help='software_tree of the source the run used, observed now')
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    controller = (a.run_dir/'controller.py').read_text()
    log = (a.run_dir/(a.stage+'.log')).read_text()
    state = json.loads((a.run_dir/'state.json').read_text())
    doc = dict(host_class=a.host_class, backend=a.backend, software_tree=state['software_tree'],
               live_source_tree=a.live_source_tree,
               test_harness_sha256=hashlib.sha256((ROOT/'tools/r40/device_tests.py').read_bytes()).hexdigest(),
               controller=controller, controller_sha256=hashlib.sha256(controller.encode()).hexdigest(),
               launch_controller_sha256=a.launch_controller_sha256, controller_state=state,
               stage_name=a.stage, log=log, log_sha256=hashlib.sha256(log.encode()).hexdigest(),
               contract_sha256=contract_v2.LOCK_SHA256,
               observed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(doc, indent=1)+'\n')
    print(a.out)


if __name__ == '__main__':
    main()
