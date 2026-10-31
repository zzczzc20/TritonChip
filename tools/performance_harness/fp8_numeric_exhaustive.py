#!/usr/bin/env python3
"""Exhaustive FP8 conversion semantics: the TNCA conversion vs torch CPU.

    $TORCH_PYTHON tools/performance_harness/fp8_numeric_exhaustive.py --out FILE.json [--threads 4]

For EVERY input -- all 2**32 binary32 patterns, all 2**16 BF16 and FP16
patterns, all 2**8 patterns of each FP8 format -- three results are
compared bit for bit:

  torch     torch CPU's own conversion (Tensor.to on CPU tensors)
  hw        the TNCA conversion instruction (ISA sec 8.4, `vf.cvt`), as the
            native model implements it: tools/performance_harness/systemc/
            include/tnca_numeric.hpp `widen`/`narrow`, the functions the
            model executes and which are checked bit for bit against the
            RTL. A 16-bit -> FP8 conversion is two instructions (Inductor
            widens a 16-bit load to binary32 first), and so is FP8 ->
            BF16/FP16 (Inductor converts to binary32 and the store narrows).
  compiled  hw plus the NaN rule the compiler emits for a conversion with
            an FP8 end (tnca_cc.frontend `_fp8_nan_fix`), restated here
            operation for operation

The claim checked: `hw` differs from `torch` only where the input is a
NaN, and `compiled` equals `torch` everywhere. The compiled kernels
themselves are run on the model and the RTL by fp8_cases.py.
"""
import argparse
import ctypes
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

HERE = Path(__file__).resolve().parent
HEADER = HERE/'systemc/include/tnca_numeric.hpp'
SHIM = r'''
#include "tnca_numeric.hpp"
#include <cstddef>
#include <cstdint>
extern "C" void convert(unsigned src, unsigned dst, const std::uint32_t* x, std::uint32_t* y, std::size_t n) {
    for (std::size_t i = 0; i < n; ++i) y[i] = tnca_numeric::narrow(dst, tnca_numeric::widen(src, x[i]));
}
'''
F32, BF16, F16, E5M2, E4M3 = 0, 1, 2, 3, 4          # ISA sec 8.4 format codes


def build(tmp):
    src = Path(tmp)/'shim.cpp'
    lib = Path(tmp)/'shim.so'
    src.write_text(SHIM)
    subprocess.run(['g++', '-O2', '-std=c++17', '-shared', '-fPIC', f'-I{HEADER.parent}',
                    str(src), '-o', str(lib)], check=True)
    so = ctypes.CDLL(str(lib))
    so.convert.argtypes = [ctypes.c_uint, ctypes.c_uint, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t]
    return so


def hw(so, src, dst, x):
    """x: uint32 numpy array of source bit patterns (low bits)."""
    import numpy as np
    x = np.ascontiguousarray(x, dtype=np.uint32)
    y = np.empty_like(x)
    so.convert(src, dst, x.ctypes.data, y.ctypes.data, x.size)
    return y


def two_step(so, src, mid, dst, x):
    return hw(so, mid, dst, hw(so, src, mid, x))


# ---- the compiler's NaN rule (tnca_cc.frontend._fp8_nan_fix), in numpy ----
def rule_narrow(y, root, root_bits):
    """FP8 result `y`; `root` the value the conversion started from."""
    import numpy as np
    sign = root if root_bits == 8 else (root >> np.uint32(root_bits - 8))
    return y | (sign & np.uint32(0x80))


def rule_widen(y, root, fmt, dst):
    """`y` the instruction's dst result for the FP8 `root`."""
    import numpy as np
    root = root.astype(np.uint32)
    e4m3 = fmt == E4M3
    if dst == F32:
        ordered = ~np.isnan(y.view(np.float32))
        top = (root << np.uint32(24))
        if e4m3:
            nan = top | np.uint32(0x7FF00000)
        else:
            nan = (top.view(np.int32) >> 3).view(np.uint32) | np.uint32(0x7FC00000)
    else:
        wide = hw_widen_cache[fmt]
        ordered = ~np.isnan(wide.view(np.float32))
        if dst == F16:
            nan = (root << np.uint32(8)) | np.uint32(0x7F80 if e4m3 else 0x7E00)
        else:
            nan = np.full_like(y, 0x7FC0)
    return np.where(ordered, y, nan)


hw_widen_cache = {}


def classify(torch_bits, hw_bits, compiled_bits, input_is_nan):
    import numpy as np
    raw = torch_bits != hw_bits
    return dict(hw_mismatches=int(raw.sum()), hw_mismatches_non_nan_input=int((raw & ~input_is_nan).sum()),
                compiled_mismatches=int((torch_bits != compiled_bits).sum()))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--threads', type=int, default=4)
    ap.add_argument('--chunk-log2', type=int, default=24)
    args = ap.parse_args()
    import numpy as np
    import torch
    torch.set_num_threads(args.threads)
    E4, E5 = torch.float8_e4m3fn, torch.float8_e5m2
    fmts = {E4: E4M3, E5: E5M2}
    names = {E4: 'e4m3', E5: 'e5m2', torch.float32: 'f32', torch.bfloat16: 'bf16', torch.float16: 'f16'}
    report = dict(schema='tnca-fp8-exhaustive/1', torch=str(torch.__version__),
                  cpu_capability=torch.backends.cpu.get_cpu_capability(), header=str(HEADER.relative_to(HERE.parents[1])),
                  results={})
    started = time.monotonic()
    with tempfile.TemporaryDirectory() as tmp:
        so = build(tmp)
        # FP8 -> binary32 / BF16 / FP16: all 256 patterns.
        for fp8 in (E4, E5):
            codes = torch.arange(256, dtype=torch.uint8)
            x = codes.numpy().astype(np.uint32)
            nan_in = codes.view(fp8).float().isnan().numpy()
            hw_widen_cache[fmts[fp8]] = hw(so, fmts[fp8], F32, x)
            for dst, code in ((torch.float32, F32), (torch.bfloat16, BF16), (torch.float16, F16)):
                ref = codes.view(fp8).to(dst)
                tb = ref.view(torch.int32 if dst == torch.float32 else torch.int16).numpy().astype(np.int64) \
                    & (0xFFFFFFFF if dst == torch.float32 else 0xFFFF)
                h = hw(so, fmts[fp8], F32, x) if dst == torch.float32 else two_step(so, fmts[fp8], F32, code, x)
                c = rule_widen(h, x, fmts[fp8], code)
                report['results'][f'{names[fp8]}->{names[dst]}'] = dict(
                    inputs=256, **classify(tb.astype(np.uint32), h, c, nan_in))
        # BF16 / FP16 -> FP8 through binary32: all 65536 patterns.
        for src, code in ((torch.bfloat16, BF16), (torch.float16, F16)):
            pats = torch.arange(65536, dtype=torch.int32).to(torch.int16)
            x = pats.numpy().astype(np.uint32) & 0xFFFF
            nan_in = pats.view(src).isnan().numpy()
            for fp8 in (E4, E5):
                tb = pats.view(src).to(fp8).view(torch.uint8).numpy().astype(np.uint32)
                h = two_step(so, code, F32, fmts[fp8], x)
                c = rule_narrow(h, x, 16)
                report['results'][f'{names[src]}->{names[fp8]}'] = dict(
                    inputs=65536, **classify(tb, h, c, nan_in))
        # binary32 -> FP8: all 2**32 patterns, in chunks.
        n = 1 << args.chunk_log2
        totals = {fp8: dict(inputs=0, hw_mismatches=0, hw_mismatches_non_nan_input=0, compiled_mismatches=0)
                  for fp8 in (E4, E5)}
        examples = {fp8: [] for fp8 in (E4, E5)}
        for start in range(0, 1 << 32, n):
            x = np.arange(start, start + n, dtype=np.uint64).astype(np.uint32)
            xt = torch.from_numpy(x.view(np.int32)).view(torch.float32)
            nan_in = np.isnan(x.view(np.float32))
            for fp8 in (E4, E5):
                tb = xt.to(fp8).view(torch.uint8).numpy().astype(np.uint32)
                h = hw(so, F32, fmts[fp8], x)
                c = rule_narrow(h, x, 32)
                r = classify(tb, h, c, nan_in)
                t = totals[fp8]
                t['inputs'] += n
                for k in ('hw_mismatches', 'hw_mismatches_non_nan_input', 'compiled_mismatches'):
                    t[k] += r[k]
                if (r['compiled_mismatches'] or r['hw_mismatches_non_nan_input']) and len(examples[fp8]) < 8:
                    bad = np.nonzero((tb != c) | ((tb != h) & ~nan_in))[0][:8 - len(examples[fp8])]
                    examples[fp8] += [dict(input=f'0x{int(x[i]):08x}', torch=f'0x{int(tb[i]):02x}',
                                           hw=f'0x{int(h[i]):02x}', compiled=f'0x{int(c[i]):02x}') for i in bad]
            if (start // n) % 32 == 31:
                print(f'binary32 {start + n:#x} done, {time.monotonic() - started:.0f}s', flush=True)
        for fp8 in (E4, E5):
            report['results'][f'f32->{names[fp8]}'] = dict(totals[fp8], examples=examples[fp8])
    report['wall_seconds'] = round(time.monotonic() - started, 1)
    report['verdict'] = dict(
        hw_differs_only_on_nan_inputs=all(r['hw_mismatches_non_nan_input'] == 0 for r in report['results'].values()),
        compiled_equals_torch_everywhere=all(r['compiled_mismatches'] == 0 for r in report['results'].values()))
    args.out.write_text(json.dumps(report, indent=1) + '\n')
    for k, r in report['results'].items():
        print(f"{k:12s} inputs {r['inputs']:>10d}  hw mismatches {r['hw_mismatches']:>8d} "
              f"(non-NaN inputs {r['hw_mismatches_non_nan_input']})  compiled mismatches {r['compiled_mismatches']}")
    print(json.dumps(report['verdict']))
    return 0 if all(report['verdict'].values()) else 1


if __name__ == '__main__':
    sys.exit(main())
