#!/usr/bin/env python3
"""P1+ T1: the R3.7 cases launched the way Triton users launch, `kernel[grid](...)`.

Same original TileBench JIT functions, same inputs and independent references
as tools/r37/aot_run.py; the difference is the entry point. Here Triton's own
JIT specialises, compiles (through the tnca backend plugin, which calls the
lowering compile_aot uses) and launches (through Session). Arguments are torch
CPU tensors; descriptors are triton.tools.tensor_descriptor.TensorDescriptor.

Checked per case: every output element against the reference, every input and
the unwritten part of every output unchanged, and the guard bytes after each
buffer intact. The device cycles of each launch are recorded next to the
frozen AOT baseline, because JIT specialisation (an integer argument equal to
1 becomes a constant, divisibility attributes) can legitimately change the
code -- a changed count is reported, not hidden and not failed.

Run with the torch environment:
    ~/src/torchenv/bin/python tools/r38/triton_native_run.py [--only OP ...]
"""
import argparse
import copy
import json
from pathlib import Path
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
import harness as H                                         # noqa: E402

import torch                                                # noqa: E402
import triton                                               # noqa: E402

E = H.E
TORCH = {'fp32': torch.float32, 'fp16': torch.float16, 'bf16': torch.bfloat16, 'i32': torch.int32,
         'i64': torch.int64, 'i8': torch.int8, 'e4m3': torch.float8_e4m3fn, 'e5m2': torch.float8_e5m2}
GUARD = 8


def host_buffer(dtype, values, outputs, name, preserve):
    size = E.DTYPES[dtype][0]
    data = bytearray(b''.join(w.to_bytes(8, 'little') for w in E.pack(values, dtype)))[:len(values) * size]
    for buf, offset, expect, _, _ in outputs:
        if name == buf and not preserve:
            data[offset * size:(offset + len(expect)) * size] = b'\xcd' * (len(expect) * size)
    raw = torch.frombuffer(bytearray(bytes(data) + b'\xa5' * GUARD), dtype=torch.uint8).clone()
    return raw, raw[:len(values) * size].view(TORCH[dtype]), bytes(data) + b'\xa5' * GUARD


def run_case(name, case):
    from tnca.triton_backend import TensorDescriptor
    fn, signature, constexprs = case.kernel_spec
    outputs = [(case.out, case.out_offset, case.expect, case.rtol, case.atol)]
    outputs += [(b, o, v, 0., 0.) for b, o, v in case.extra_outputs]
    grid = tuple(p + 1 for p in case.program_id)
    if case.kernel == 'batched_matmul':
        a, bt = case.buffers['a'][1], case.buffers['bt'][1]
        expected = [sum(a[z*256+m*16+k]*bt[z*256+n*16+k] for k in range(16))
                    for z in range(2) for m in range(16) for n in range(16)]
        outputs = [(case.out, 0, expected, case.rtol, case.atol)]
    raws, views, initial = {}, {}, {}
    for bname, (dtype, values) in case.buffers.items():
        raws[bname], views[bname], initial[bname] = host_buffer(dtype, values, outputs, bname, case.preserve_output)
    args = {}
    for pname in fn.arg_names:
        if pname in constexprs:
            args[pname] = constexprs[pname]
        elif pname in case.scalars:
            args[pname] = case.scalars[pname]
        elif pname in case.descriptors:
            d = case.descriptors[pname]
            args[pname] = TensorDescriptor(views[d['buffer']], d['shape'], d['strides'], d['block'])
        elif pname in views:
            off = case.offsets.get(pname, 0)
            args[pname] = views[pname][off:] if off else views[pname]
        else:
            raise KeyError(f'{name}: no value for parameter {pname!r}')
    kernel = fn
    kernel[grid](**args)
    from tnca.triton_backend import driver as D
    captured, count = {}, 0
    for bname, raw in raws.items():
        data = raw.numpy().tobytes()
        dtype, values = case.buffers[bname]
        size, _, decode = E.DTYPES[dtype]
        unchanged = bytearray(b'\1' * len(data))
        for buf, offset, expect, rtol, atol in outputs:
            if bname != buf:
                continue
            start, end = offset * size, (offset + len(expect)) * size
            got = [decode(int.from_bytes(data[i:i+size], 'little')) for i in range(start, end, size)]
            ok, detail = E._compare_values(got, expect, rtol, atol)
            if not ok:
                raise AssertionError(f'{buf}: {detail}')
            unchanged[start:end] = bytes(end - start)
            count += len(expect)
        if any(c and x != y for c, x, y in zip(unchanged, data, initial[bname])):
            raise AssertionError(f'{bname}: input, output tail or guard overwritten')
        captured[bname] = (dtype, [decode(int.from_bytes(data[i:i+size], 'little'))
                                   for i in range(0, len(values) * size, size)])
    launcher = D.TNCALauncher.last
    return captured, dict(detail=f'{count} elements correct; inputs, tails and guards intact',
                          grid=grid, cycles=getattr(launcher, 'last_cycles', None))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--only', nargs='+')
    ap.add_argument('--json', type=Path, default=ROOT / 'sim' / 'build' / 'r38' / 'torch' / 'triton_native.json')
    args = ap.parse_args()
    args.json.parent.mkdir(parents=True, exist_ok=True)
    frozen = json.loads((ROOT / 'spec' / 'r38' / 'baseline_cycles.json').read_text())['cycles']
    cases = H.cases()
    rows, captured = [], {}
    wanted = None
    if args.only:
        wanted = {n for n, c in cases.items() if n in args.only or c.kernel in args.only}
        changed = True
        while changed:
            changed = False
            for n in list(wanted):
                for feed in cases[n].feeds.values():
                    if feed[0] not in wanted:
                        wanted.add(feed[0]); changed = True
    for name, original in cases.items():
        if wanted is not None and name not in wanted:
            continue
        case = copy.copy(original)
        case.buffers = copy.deepcopy(original.buffers)
        row = dict(case=name, operator=case.kernel, ok=False)
        t = time.monotonic()
        try:
            for buf, feed in case.feeds.items():
                dtype, data = captured[feed[0]][feed[1]]
                if len(feed) == 3:
                    nr, nc = feed[2]
                    data = [data[r * nc + c] for c in range(nc) for r in range(nr)]
                case.buffers[buf] = (dtype, data)
            captured[name], detail = run_case(name, case)
            row.update(ok=True, **detail)
        except Exception as e:                            # noqa: BLE001
            row['detail'] = f'{type(e).__name__}: {e}'[:400]
        row['baseline_cycles'] = frozen.get(name)
        row['seconds'] = round(time.monotonic() - t, 2)
        rows.append(row)
        print(f'{"PASS" if row["ok"] else "FAIL"} {name} {row.get("cycles")} (aot {row["baseline_cycles"]}) '
              f'{"" if row["ok"] else row["detail"][:200]}', flush=True)
    same = sum(1 for r in rows if r['ok'] and r.get('cycles') == r['baseline_cycles'])
    args.json.write_text(json.dumps(dict(rows=rows, passed=sum(r['ok'] for r in rows), total=len(rows),
                                         same_cycles_as_aot=same), indent=1) + '\n')
    print(f'triton-native: {sum(r["ok"] for r in rows)}/{len(rows)} pass; '
          f'{same} with cycles identical to the AOT baseline')
    return 0 if rows and all(r['ok'] for r in rows) else 1


if __name__ == '__main__':
    sys.exit(main())
