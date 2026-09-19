#!/usr/bin/env python3
"""Exhaustive FP16 and BF16 widening vectors: all 65,536 of each."""
import argparse
import struct


def f16_to_f32_bits(h):
    s = (h >> 15) & 1
    e = (h >> 10) & 0x1f
    f = h & 0x3ff
    if e == 0x1f:
        return 0x7fc00000 if f else ((s << 31) | (0xff << 23))
    if e == 0:
        if f == 0:
            return s << 31
        # subnormal: value = f * 2^-24
        lz = 10 - f.bit_length()
        exp = 112 - lz
        frac = ((f << (lz + 1)) & 0x3ff) << 13
        return (s << 31) | (exp << 23) | frac
    return (s << 31) | ((e + 112) << 23) | (f << 13)


def bf16_to_f32_bits(b):
    return (b << 16) & 0xffffffff


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('-o', '--output', required=True)
    args = ap.parse_args()
    lines = []
    for v in range(1 << 16):
        lines.append(f"{v:04x} 0 {f16_to_f32_bits(v):08x}")
        lines.append(f"{v:04x} 1 {bf16_to_f32_bits(v):08x}")
    with open(args.output, 'w') as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"{len(lines)} vectors -> {args.output}")


if __name__ == '__main__':
    main()
