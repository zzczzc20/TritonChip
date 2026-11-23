#!/usr/bin/env python3
"""Model-versus-board validation table (domain: SystemC/TLM model prediction against board).

    python3 tools/r42/kfuse_speed_validate.py MODEL.json [MODEL2.json ...] [--out OUT.json] [--board ab-int2|b41-run3-base]

Board: evidence/r42/board/ab-int2/records.tgz (R42 software of 5327118c, run-3 image 791b8af2, 100 MHz, serial-grid device
cycles, median of the recorded samples). Model: collected by kfuse_speed_collect.py. For every point present in the model
files the board cycles, the model cycles and their ratio are printed; per family and per class (GEMM-shaped / other) the
geometric mean ratio and the median and maximum absolute error. Points the model did not run are not listed.
"""
import argparse
import json
import math
import statistics
import tarfile
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
GEMM_FAMILIES = {'matmul', 'linear', 'mlp', 'gemv', 'bmm', 'gpt2_block', 'llama_prefill', 'llama_decode',
                 'attention_prefill', 'attention_decode'}


def board_records(which='ab-int2'):
    path = ROOT / f'evidence/r42/board/{which}/records.tgz'
    out = {}
    with tarfile.open(path) as t:
        for m in t.getmembers():
            if m.name.endswith('/original/formal.jsonl') or m.name == 'queue/ab-int2/original/formal.jsonl' \
                    or m.name.endswith('original/formal.jsonl'):
                for line in t.extractfile(m).read().decode().splitlines():
                    r = json.loads(line)
                    out[r['point_id']] = r
    return out


def family_of(pid):
    return pid.split('/')[0].split('.', 1)[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('models', nargs='+')
    ap.add_argument('--out')
    ap.add_argument('--board', default='ab-int2')
    a = ap.parse_args()
    board = board_records(a.board)
    model = {}
    for f in a.models:
        d = json.loads(Path(f).read_text())
        model.update(d['points'])
    rows = []
    for pid, m in sorted(model.items()):
        b = board.get(pid)
        if not b or m.get('status') != 'pass' or not m.get('device_cycles'):
            rows.append(dict(point=pid, status=m.get('status'), board=b['time']['device'] if b else None, model=None))
            continue
        bc = b['time']['device']
        rows.append(dict(point=pid, family=family_of(pid), tier=pid.split('/')[1], board=bc, model=m['device_cycles'],
                         ratio=m['device_cycles'] / bc, status='pass',
                         launches_board=b['time']['per_launch'], launches_model=m.get('per_launch')))
    ok = [r for r in rows if r.get('ratio')]
    print(f'{"point":<62}{"board":>14}{"model":>14}{"model/board":>12}')
    for r in rows:
        if r.get('ratio'):
            print(f'{r["point"]:<62}{r["board"]:>14,}{r["model"]:>14,}{r["ratio"]:>12.3f}')
        else:
            print(f'{r["point"]:<62}{"":>14}{"-":>14}   {r["status"]}')

    def stats(rs):
        e = [abs(r['ratio'] - 1) for r in rs]
        g = math.exp(sum(math.log(r['ratio']) for r in rs) / len(rs))
        return dict(n=len(rs), geo_ratio=g, median_abs_err=statistics.median(e), max_abs_err=max(e),
                    within20=sum(x <= 0.20 for x in e), within10=sum(x <= 0.10 for x in e))
    fam = defaultdict(list)
    for r in ok:
        fam[r['family']].append(r)
    summary = dict(all=stats(ok) if ok else None,
                   gemm_shaped=stats([r for r in ok if r['family'] in GEMM_FAMILIES]) if ok else None,
                   other=stats([r for r in ok if r['family'] not in GEMM_FAMILIES]) if any(
                       r['family'] not in GEMM_FAMILIES for r in ok) else None,
                   by_family={k: stats(v) for k, v in sorted(fam.items())})
    print()
    for k in ('all', 'gemm_shaped', 'other'):
        s = summary[k]
        if s:
            print(f'{k:<12} n={s["n"]:>3} geo(model/board)={s["geo_ratio"]:.3f} median|err|={s["median_abs_err"]:.3f} '
                  f'max|err|={s["max_abs_err"]:.3f} within10%={s["within10"]} within20%={s["within20"]}')
    for k, s in summary['by_family'].items():
        print(f'  {k:<18} n={s["n"]:>2} geo={s["geo_ratio"]:.3f} med|err|={s["median_abs_err"]:.3f} max={s["max_abs_err"]:.3f}')
    if a.out:
        Path(a.out).write_text(json.dumps(dict(summary=summary, rows=rows), indent=1))


if __name__ == '__main__':
    main()
