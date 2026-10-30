#!/usr/bin/env python3
"""Check the routed platform IRQ source's LUT-level sticky condition.

This establishes a local state invariant, not a minimum pulse width, a CDC
sign-off, or an authorization to operate the card.
"""
import hashlib
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[3]
SEGMENTS = ROOT/'evidence/r41/hbm_bw/g4/run3/route_prepackage/platform_irq_segments.txt'
LOGIC = ROOT/'evidence/r41/hbm_bw/g4/run2/link_1f0a5aff/original_cdc_logic.txt'
LOGIC_SHA = 'c2018fea38f9eac440dbf444192db5eb3828fc326fd9c3ea4e39d8f0deacf682'


def main():
    raw = LOGIC.read_bytes()
    if hashlib.sha256(raw).hexdigest() != LOGIC_SHA:
        raise ValueError('original/candidate identical logic trace changed')
    trace = SEGMENTS.read_text()
    if "DRIVER_INIT 64'h5554FFFF55545554" not in trace:
        raise ValueError('unexpected source D LUT')
    inputs = {}
    for line in trace.splitlines():
        m = re.search(r'mi_r_error_i_i_1/I([0-5]) SEGMENTS (.*)', line)
        if m:
            inputs[int(m.group(1))] = m.group(2)
    if '/mi_r_error' not in inputs.get(5, '') or 'mi_r_error_i_i_6_n_0' not in inputs.get(4, ''):
        raise ValueError('source feedback or clear-control net differs')
    text = raw.decode()
    match = re.search(r'cell ([^\n]*axi_firewall_ctrl_user/inst/gen_mi\.mi_r_error_i_i_6) ref LUT4\nINIT 16\'h0008\n((?:pin [^\n]+\n){5})', text)
    if not match:
        raise ValueError('clear-control LUT changed')
    actual = {}
    for line in match.group(2).splitlines():
        m = re.search(r'/I([0-3]) net (.*)', line)
        if m:
            actual[int(m.group(1))] = m.group(2).rsplit('/',1)[-1]
    if actual != {0:'r_state[1]',1:'r_state[0]',2:'unblock',3:'r_state[2]'}:
        raise ValueError('clear-control inputs differ')
    lut = 0x5554FFFF55545554
    if any(not ((lut >> (32|low)) & 1) for low in range(16)):
        raise ValueError('asserted error does not hold when clear-control is zero')
    clear_lut = 0x0008
    if [n for n in range(16) if clear_lut >> n & 1] != [3]:
        raise ValueError('clear-control truth table differs')
    print('PASS: asserted Q holds while clear-control=0 and reset is inactive;')
    print('clear-control requires r_state[2:0]=011 and unblock=0.')
    print('This does not prove minimum assertion duration or glitch-free aggregation.')


if __name__ == '__main__':
    main()
