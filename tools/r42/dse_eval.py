#!/usr/bin/env python3
"""Per-point model cycles of one run bundle and the 145-point prediction (domain: SystemC/TLM model applied as ratios to board cycles).

A bundle is the set of collected files  col/<tag>_<set>.json  with sets tsm (77 tiny/small/medium of the 113), hv (heavy points run
directly), px (reduced-grid proxies), xd (xl points run directly), xp (xl proxies). `tag` = <tree>_<config>, tree val|head.

model_cycles(bundle, pid) -> (cycles, kind):
  direct  the point itself ran in the model
  proxy   a reduced-grid proxy ran; its cycles are scaled by the demand ratio (macs or bytes) point/proxy (dse_proxies.PROXY_FOR)
  wave    FP8 Hub-kernel point: T = programs/(4*cores) waves x (a + b*K/128), a and b fitted on the xp proxies (m=1 and m=48 classes)
  parts   xl composite: sum over parts of planning share x factor of the part (shares: kfuse_speed xl_inputs.json, analytic)
Only ratios of model cycles between bundles reach the report (the model bias cancels): a proxy's scale and the planning shares are
common to both bundles.
"""
import json
import math
import sys
from functools import lru_cache
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dse_common as C   # noqa: E402

COL = C.ROOT / 'evidence/r42/dse/col'
SETS = ('tsm', 'hv', 'px', 'xd', 'xp')
BLOCK = 128
HERE = Path(__file__).resolve().parent


@lru_cache(None)
def _inputs():
    for p in (C.ROOT / 'evidence/r42/kfuse_speed/xl_inputs.json', Path('/home/ubuntu/TritonChip/evidence/r42/kfuse_speed/xl_inputs.json')):
        if p.exists():
            return {q['id']: q for q in json.loads(p.read_text())['points']}
    raise FileNotFoundError('xl_inputs.json')


@lru_cache(None)
def proxy_for():
    p = C.ROOT / 'evidence/r42/dse/proxy_map.json'
    return json.loads(p.read_text())


class Bundle:
    def __init__(self, tag, cores=3):
        self.tag = tag
        self.cores = cores
        self.d = {}
        for s in SETS:
            f = COL / f'{tag}_{s}.json'
            if f.exists():
                for k, v in C.load_collected(f).items():
                    if v.get('status') == 'pass' and v.get('device_cycles'):
                        self.d[k] = v
        self._wave = {}

    # ---- FP8 wave model --------------------------------------------------------------------------------------------
    def wave(self, mclass):
        if mclass in self._wave:
            return self._wave[mclass]
        m = 1 if mclass == 1 else 48
        pts = []
        pw = -(-(12 * (1 if m == 1 else 3)) // (4 * self.cores))       # waves of the proxy (12 column blocks x row chunks)
        for k in (512, 1024):
            v = self.d.get(f'op.fp8_linear/xl/fp8-m{m}-k{k}-n1536')
            if v:
                pts.append((k // BLOCK, v['device_cycles'] / pw))
        f = None
        if len(pts) == 2:
            (k1, c1), (k2, c2) = pts
            b = (c2 - c1) / (k2 - k1)
            a = c1 - b * k1
            f = (a, b)
        self._wave[mclass] = f
        return f

    def fp8_cycles(self, m, n, k, mclass):
        f = self.wave(mclass)
        if not f:
            return None
        programs = -(-m // 16) * -(-n // BLOCK)
        waves = -(-programs // (4 * self.cores))
        return waves * (f[0] + f[1] * (k // BLOCK))

    # ---- point resolution ------------------------------------------------------------------------------------------
    def direct(self, pid):
        v = self.d.get(pid)
        return v['device_cycles'] if v else None

    def proxy(self, pid, board):
        pf = proxy_for().get(pid)
        if not pf:
            return None
        v = self.d.get(pf['proxy'])
        if not v:
            return None
        key = 'macs' if pf['kind'] == 'macs' else 'bytes'
        full = board[pid]['demand']
        def dem(x):
            return x['macs'] if key == 'macs' else x['read_bytes'] + x['write_bytes']
        pd = v.get('demand')
        if not pd or not dem(pd):
            return None
        return v['device_cycles'] * dem(full) / dem(pd)

    def fp8_point(self, pid):
        inp = _inputs().get(pid)
        if not inp or not inp['fp8_shapes']:
            return None
        mc = 1 if inp['m_class'] == 1 else 16
        tot = 0.0
        for (m, n, k) in inp['fp8_shapes']:
            c = self.fp8_cycles(m, n, k, mc)
            if c is None:
                return None
            tot += c
        return tot


def model_cycles(bd, pid, board):
    """(cycles, kind) of a point in a bundle, or (None, None)"""
    c = bd.direct(pid)
    if c is not None:
        return c, 'direct'
    c = bd.proxy(pid, board)
    if c is not None:
        return c, 'proxy'
    if board[pid]['family'] == 'fp8_linear':
        c = bd.fp8_point(pid)
        if c is not None:
            return c, 'wave'
    return None, None


PART_STANDINS = {   # xl composite part -> the xl points whose ratio stands for it
    'attention_decode': ['op.attention_decode/xl/fp16-b1-h16-kv8-l2048-d128', 'op.attention_decode/xl/fp16-b1-h16-kv8-l4096-d128',
                         'op.attention_decode/xl/fp16-b1-h16-kv8-l8192-d128'],
    'attention_prefill': ['op.attention_prefill/xl/fp16-b1-h16-kv8-s1024-d128', 'op.attention_prefill/xl/fp16-b1-h16-kv8-s2048-d128'],
    'elementwise': ['op.rmsnorm_residual/xl/fp16-r2048-d2048', 'op.rope/xl/fp16-b1-h16-s2048-d128', 'op.swiglu/xl/fp16-r2048-i6144'],
    'head': ['op.gemv/xl/fp16-k2048-n151936'],
}


def part_ratio(part, point_ratio, composite_id):
    """ratio (cfg/val-base) of a non-FP8 part of a composite, from the ratios of its stand-in points; None when none is available"""
    if part == 'attention':
        key = 'attention_decode' if 'decode' in composite_id else 'attention_prefill'
    else:
        key = part
    rs = [point_ratio[q] for q in PART_STANDINS.get(key, []) if q in point_ratio]
    return geo(rs) if rs else None


def composite_ratio(bd_cfg, bd_base, pid, point_ratio):
    """xl composite: planning-share weighted ratio of its parts (shares: analytic, kfuse_speed xl_inputs.json)"""
    inp = _inputs()[pid]
    tot0 = tot1 = 0.0
    used = {}
    for q in inp['parts']:
        part, cyc = q['part'], q['cycles']
        if part == 'fp8_projections':
            c0, c1 = bd_base.fp8_point(pid), bd_cfg.fp8_point(pid)
            f = c1 / c0 if c0 and c1 else None
        else:
            f = part_ratio(part, point_ratio, pid)
        used[part] = f
        tot0 += cyc
        tot1 += cyc * (1.0 if f is None else f)
    return tot1 / tot0, used


def geo(xs):
    return C.geo(xs)


def evaluate(head_cfg_tag, val_base_tag, head_base_tag, board, cores_cfg=3, clock_mhz=100.0):
    """Per-point table of one configuration (ratio of model cycles, head software on cfg / validation software on the base model).

    ratio_i = m(head, cfg) / m(val software, base model): the model's prediction of the cycle ratio of this configuration with the
    current software head against the round-2 software on today's hardware. predicted board cycles = board_r2 x ratio (the model's
    absolute bias cancels). Points the model could not run take the geometric-mean ratio of their family (kind 'family'), else 1.0.
    """
    bc = Bundle(head_cfg_tag, cores_cfg)
    bv = Bundle(val_base_tag, 3)
    bh = Bundle(head_base_tag, 3)
    rows = {}
    for pid, b in board.items():
        c1, k1 = model_cycles(bc, pid, board)
        c0, k0 = model_cycles(bv, pid, board)
        ch, kh = model_cycles(bh, pid, board)
        rows[pid] = dict(point=pid, family=b['family'], tier=b['tier'], xl=b['xl'], b41=b['b41'], r2=b['r2'], kind=k1 if (c1 and c0) else None,
                         m_val=c0, m_head_base=ch, m_cfg=c1)
        if c1 and c0:
            rows[pid]['ratio'] = c1 / c0
            rows[pid]['hw_ratio'] = c1 / ch if ch else None        # hardware effect alone: cfg / head software on the base model
            rows[pid]['sw_ratio'] = ch / c0 if ch else None        # software effect alone (head against round 2) on today's hardware
            if b['family'] in ('fp8_linear', 'fp8_act_quant') or True:
                pass
    ratios = {p: r['ratio'] for p, r in rows.items() if 'ratio' in r}
    hw = {p: r['hw_ratio'] for p, r in rows.items() if r.get('hw_ratio')}
    sw = {p: r['sw_ratio'] for p, r in rows.items() if r.get('sw_ratio')}
    # xl composites
    for pid, b in board.items():
        if pid in _inputs() and b['family'].startswith('fp8_layer') or b['family'].startswith('fp8_decoder'):
            f, used = composite_ratio(bc, bv, pid, ratios)
            fh, _ = composite_ratio(bh, bv, pid, {p: r['sw_ratio'] for p, r in rows.items() if r.get('sw_ratio')})
            rows[pid].update(ratio=f, kind='parts', parts=used, sw_ratio=fh, hw_ratio=f / fh if fh else None)
            ratios[pid] = f
    # families inherit for points without a stand-in
    byfam = {}
    for p, r in rows.items():
        if 'ratio' in r:
            byfam.setdefault(r['family'], []).append(r)
    for p, r in rows.items():
        if 'ratio' not in r:
            rs = byfam.get(r['family'])
            same = [x for x in rs if x['tier'] == r['tier']] if rs else []
            use = same or rs
            if use:
                r['ratio'] = geo([x['ratio'] for x in use]); r['kind'] = 'family'
                hws = [x['hw_ratio'] for x in use if x.get('hw_ratio')]
                sws = [x['sw_ratio'] for x in use if x.get('sw_ratio')]
                r['hw_ratio'] = geo(hws) if hws else 1.0
                r['sw_ratio'] = geo(sws) if sws else 1.0
            else:
                r['ratio'] = r['hw_ratio'] = r['sw_ratio'] = 1.0; r['kind'] = 'held'
    for r in rows.values():
        r['pred_cycles'] = r['r2'] * r['ratio']
        r['pred_wall_ratio_vs_b41'] = (r['b41'] / 100.0) / (r['pred_cycles'] / clock_mhz)    # b41 at 100 MHz over predicted time at clock_mhz
    return rows


def summarize(rows):
    out = dict(S145=geo([r['pred_wall_ratio_vs_b41'] for r in rows.values()]), n=len(rows))
    for key in ('tier', 'family'):
        g = {}
        for r in rows.values():
            g.setdefault(r[key], []).append(r['pred_wall_ratio_vs_b41'])
        out['by_' + key] = {k: geo(v) for k, v in sorted(g.items())}
    kinds = {}
    for r in rows.values():
        kinds[r['kind']] = kinds.get(r['kind'], 0) + 1
    out['kinds'] = kinds
    return out
