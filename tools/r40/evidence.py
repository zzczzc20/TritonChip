"""Gate evidence for tools/r40/acceptance.py: one JSON file per work package.

Each R40 tool records the gate it decides in sim/build/r40/<WP>.json under
`gates[<gate id>]`, with a status (acceptance.STATUSES), the checks that
support it and any detail, without disturbing the gates other tools wrote
there. acceptance.py refuses a passing status without checks, so a tool
cannot pass a gate by saying so.
"""
import datetime
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def path_for(gate_id):
    root = ROOT / 'sim' / 'build' / 'r40'
    version = os.environ.get('R40_CONTRACT', 'v1')
    if version not in ('v1', 'v2'):
        raise ValueError('R40_CONTRACT must be v1 or v2')
    if version == 'v2':
        root = root / 'v2'
    return root / f"{gate_id.split('.', 1)[0]}.json"


def record(gate_id, status, checks, *, detail='', path=None, **extra):
    """Write one gate's evidence; returns the path written."""
    p = Path(path) if path else path_for(gate_id)
    if os.environ.get('R40_CONTRACT') == 'v2':
        from contract_v2 import verify, LOCK_SHA256
        verify()
        extra.update(contract_version=2, contract_sha256=LOCK_SHA256)
    p.parent.mkdir(parents=True, exist_ok=True)
    try:
        doc = json.loads(p.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        doc = {}
    doc.setdefault('gates', {})[gate_id] = dict(
        status=status, checks=[dict(name=c['name'], ok=bool(c['ok']),
                                    **{k: v for k, v in c.items() if k not in ('name', 'ok')})
                               for c in checks],
        detail=detail,
        written=datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds'),
        **extra)
    p.write_text(json.dumps(doc, indent=1, default=str) + '\n')
    return p
