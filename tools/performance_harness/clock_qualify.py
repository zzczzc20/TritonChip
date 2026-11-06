#!/usr/bin/env python3
"""Qualified operating clock of the R41 Stage 2 run-1 image (100 MHz).

    clock_qualify.py capture  --image X --uuid U --out board_clock.json [--sysfs ROOT]
    clock_qualify.py assemble --image X --audit DIR --capture board_clock.json --out qualified_clock.json
    clock_qualify.py verify   --evidence qualified_clock.json --image X [--sysfs ROOT] --out verify.json

`capture` only reads sysfs (loaded UUID, clock counter, ICAP clock readback):
it never opens, programs or resets the card. It refuses unless the loaded
UUID is the image's and every counter sample agrees.

`assemble` derives every field of tnca-qualified-operating-clock/1 from raw
files, never from typed-in numbers: the post-route audit of the routed
checkpoint (flow/fpga/audit_r40_operating_clock.tcl ... 100 placement:
timing.rpt, route_status.rpt, cdc.rpt, actual_cores.rpt, analysis.tsv) and the
capture. The CDC review is accepted only when the critical CDC findings are
exactly the 56 platform findings reviewed for attempt 3
(evidence/r40/v2/convergence/attempt03/qualified_clock90_after_load.json);
anything else is refused for a new review. The result is re-read through
python/tnca/operating_clock.qualified_clock, the check every board record makes.

`verify` re-reads the counter now and re-checks the evidence (identity step).
"""
import argparse
import datetime
import hashlib
import json
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'python'))

IMAGE_SHA256 = '30e79769c128668ac8f1d0d9017f6ea8d22b8335b8fee18cbe374f4bdef4e5de'
QUALIFICATION = 'r41s2-run1-100mhz'
CLOCK_HZ = 100_000_000
DEVICE, MGMT = '0000:04:00.1', '0000:04:00.0'
A3_REVIEW = ROOT / 'evidence/r40/v2/convergence/attempt03/qualified_clock90_after_load.json'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def utc():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def canonical_uuid(text):
    raw = str(text).strip().lower().replace('-', '')
    if len(raw) != 32 or any(c not in '0123456789abcdef' for c in raw):
        raise ValueError(f'not a uuid: {text!r}')
    return f'{raw[:8]}-{raw[8:12]}-{raw[12:16]}-{raw[16:20]}-{raw[20:]}'


def capture(sysfs, image, uuid, samples=3, interval=2.0):
    base, mgmt = Path(sysfs) / DEVICE, Path(sysfs) / MGMT
    loaded = canonical_uuid((base / 'xclbinuuid').read_text())
    if loaded != canonical_uuid(uuid):
        raise SystemExit(f'loaded image {loaded} is not {uuid}; nothing captured')
    counters = sorted(mgmt.glob('clock_freq_counter*/clock_counter_freqs'))
    if not counters:
        raise SystemExit('no clock counter under the management function')
    rows = []
    for i in range(samples):
        raw = counters[0].read_text()
        rows.append(dict(raw=raw, frequencies_mhz=[int(v) for v in raw.split()]))
        if i + 1 < samples:
            time.sleep(interval)
    first = {r['frequencies_mhz'][0] for r in rows}
    if len(first) != 1:
        raise SystemExit(f'clock counter samples disagree: {sorted(first)}')
    icap = {}
    for path in sorted(list(mgmt.glob('icap*/clock_freqs')) + list(base.glob('icap*/clock_freqs'))):
        icap[str(path)] = path.read_text()
    if canonical_uuid((base / 'xclbinuuid').read_text()) != loaded:
        raise SystemExit('loaded image changed during the capture')
    mhz = first.pop()
    return dict(schema='tnca-loaded-clock-capture/1', observed_utc=utc(), image_sha256=sha(image),
                loaded_xclbin_uuid=loaded, device=DEVICE, counter_path=str(counters[0]),
                counter_samples=rows, icap_readback=icap, clock_hz=mhz * 1_000_000,
                physical_core_clock_divider=1, clock_source='measured-board-counter',
                measurement_resolution_hz=1_000_000,
                scope='DATA clock counter index 0 (v++ clock ID 0, DATA_CLK); undivided core clock; '
                      'integer MHz readout; read-only sysfs, captured after program.sh loaded the image')


def design_timing(text):
    lines = text.splitlines()
    for i, line in enumerate(lines):
        if line.strip().startswith('| Design Timing Summary'):
            for row in lines[i + 1:i + 12]:
                cells = row.split()
                if len(cells) == 12 and re.fullmatch(r'-?\d+\.\d+', cells[0]):
                    v = [float(c) if '.' in c else int(c) for c in cells]
                    return dict(wns=v[0], tns=v[1], tns_failing=v[2], whs=v[4], ths=v[5], ths_failing=v[6],
                                wpws=v[8], tpws=v[9], tpws_failing=v[10])
    raise ValueError('no Design Timing Summary row in timing.rpt')


def route_status(text):
    def field(name):
        m = re.search(r'# of ' + name + r'\.*\s*:\s*(\d+)', text)
        if not m:
            raise ValueError(f'route_status.rpt lacks {name}')
        return int(m.group(1))
    return dict(routable=field('routable nets'), fully_routed=field('fully routed nets'),
                errors=field('nets with routing errors'))


def cdc_critical(text):
    rows = set()
    for line in text.splitlines():
        m = re.match(r'\s*\d+\s+(CDC-\d+)\s+Critical\s', line)
        if m:
            cells = line.split()
            rows.add((m.group(1), cells[-2], cells[-1]))
    return rows


def actual_cores(text):
    slrs, failing = {}, None
    for line in text.splitlines():
        m = re.match(r'core (\d+) .*actual_slrs \{(SLR\d)\} expected (SLR\d) pass 1$', line.strip())
        if m and m.group(2) == m.group(3):
            slrs[m.group(1)] = int(m.group(2)[3:])
        if line.startswith('failing_cores'):
            failing = line.split(None, 1)[1].strip()
    if failing != '{}':
        raise ValueError(f'placement audit reports failing cores {failing}')
    return slrs


def assemble(image, audit, board_clock):
    audit = Path(audit)
    image_sha = sha(image)
    if image_sha != IMAGE_SHA256:
        raise SystemExit(f'{image} is not the run-1 image ({image_sha})')
    capture_row = json.loads(Path(board_clock).read_text())
    problems = []
    if capture_row.get('image_sha256') != image_sha:
        problems.append('capture belongs to another image')
    if capture_row.get('clock_hz') != CLOCK_HZ:
        problems.append(f'measured clock {capture_row.get("clock_hz")} Hz, not {CLOCK_HZ}')
    analysis = dict(l.split('\t', 1) for l in (audit / 'analysis.tsv').read_text().splitlines() if '\t' in l)
    if abs(float(analysis.get('analysis_period_ns', 'nan')) - 10.0) > 0.00051 or analysis.get('requested_mhz') != '100':
        problems.append(f'audit was not at the routed 10 ns period ({analysis})')
    t = design_timing((audit / 'timing.rpt').read_text())
    if t['wns'] < 0 or t['whs'] < 0 or t['wpws'] < 0 or t['tns_failing'] or t['ths_failing'] or t['tpws_failing']:
        problems.append(f'post-route timing not met: {t}')
    r = route_status((audit / 'route_status.rpt').read_text())
    if r['errors'] or r['fully_routed'] != r['routable']:
        problems.append(f'routing incomplete: {r}')
    a3 = json.loads(A3_REVIEW.read_text())['qualified_clock']['cdc_review_scope']
    reviewed = {(x['rule'], x['source'], x['destination']) for x in a3['reviewed_critical']}
    found = cdc_critical((audit / 'cdc.rpt').read_text())
    if found != reviewed:
        problems.append(f'critical CDC findings differ from the attempt-3 review: '
                        f'{len(found - reviewed)} new, {len(reviewed - found)} gone; a new review is required')
    slrs = actual_cores((audit / 'actual_cores.rpt').read_text())
    if slrs != {'0': 0, '1': 1, '2': 2}:
        problems.append(f'physical cores not in their SLRs: {slrs}')
    if problems:
        raise SystemExit('refused: ' + '; '.join(problems))
    files = {name: dict(path=str((audit / f).resolve()), sha256=sha(audit / f)) for name, f in
             (('timing', 'timing.rpt'), ('route', 'route_status.rpt'), ('cdc', 'cdc.rpt'),
              ('placement', 'actual_cores.rpt'))}
    files['board_clock'] = dict(path=str(Path(board_clock).resolve()), sha256=sha(board_clock))
    return dict(schema='tnca-qualified-operating-clock/1', image_sha256=image_sha, qualification=QUALIFICATION,
                clock_hz=capture_row['clock_hz'], setup_wns_ns=t['wns'], hold_whs_ns=t['whs'],
                pulse_width_wpws_ns=t['wpws'], route_legal=True, cdc_review='passed',
                cdc_review_scope=dict(status='critical findings identical to the attempt-3 platform-scoped review',
                                      reviewed_source=str(A3_REVIEW.relative_to(ROOT)),
                                      reviewed_source_sha256=sha(A3_REVIEW), critical=len(found)),
                clock_source='measured-board-counter', physical_core_slrs=slrs, route=r, timing=t,
                audit_analysis=analysis, clock_capture=capture_row, evidence_files=files,
                scope='R41 S2-01 100 MHz candidate (run 1/5): routed at 100 MHz, measured 100 MHz on the card; '
                      'not the final Stage 2 delivery')


def check(evidence, image):
    from tnca.operating_clock import qualified_clock
    return qualified_clock(evidence, sha(image))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    c = sub.add_parser('capture')
    c.add_argument('--image', type=Path, required=True)
    c.add_argument('--uuid', required=True)
    c.add_argument('--sysfs', type=Path, default=Path('/sys/bus/pci/devices'))
    c.add_argument('--interval', type=float, default=2.0)
    c.add_argument('--out', type=Path, required=True)
    a = sub.add_parser('assemble')
    a.add_argument('--image', type=Path, required=True)
    a.add_argument('--audit', type=Path, required=True)
    a.add_argument('--capture', type=Path, required=True)
    a.add_argument('--out', type=Path, required=True)
    v = sub.add_parser('verify')
    v.add_argument('--evidence', type=Path, required=True)
    v.add_argument('--image', type=Path, required=True)
    v.add_argument('--uuid', required=True)
    v.add_argument('--sysfs', type=Path, default=Path('/sys/bus/pci/devices'))
    v.add_argument('--xclbinutil', default='xclbinutil')
    v.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    if args.cmd == 'capture':
        row = capture(args.sysfs, args.image, args.uuid, interval=args.interval)
        if args.out.exists():
            raise SystemExit(f'{args.out} exists; captures are never overwritten')
        args.out.write_text(json.dumps(row, indent=1) + '\n')
        print(f'captured {row["clock_hz"]} Hz on {row["loaded_xclbin_uuid"]}')
    elif args.cmd == 'assemble':
        row = assemble(args.image, args.audit, args.capture)
        args.out.write_text(json.dumps(row, indent=1) + '\n')
        got = check(args.out, args.image)
        print(f'qualified {got["clock_hz"]} Hz ({got["clock_qualification"]}) '
              f'evidence {got["operating_clock_evidence_sha256"]}')
    else:
        got = check(args.evidence, args.image)
        now = capture(args.sysfs, args.image, args.uuid, interval=1.0)
        info = subprocess.run([args.xclbinutil, '--info', '--input', str(args.image)],
                              capture_output=True, text=True)
        m = re.search(r'UUID \(xclbin\):\s*(\S+)', info.stdout)
        data_clk = re.search(r'Name:\s*DATA_CLK\s*\n\s*Index:.*\n\s*Type:.*\n\s*Frequency:\s*(\d+) MHz', info.stdout) \
            or re.search(r'DATA_CLK[\s\S]{0,120}?Frequency:\s*(\d+) MHz', info.stdout)
        row = dict(schema='tnca-r41s2-clock-verify/1', observed_utc=utc(), image=str(args.image),
                   image_sha256=sha(args.image), qualified=dict(got, operating_clock_evidence=None),
                   evidence=str(args.evidence), capture_now=now,
                   xclbinutil_uuid=m.group(1) if m else None,
                   xclbinutil_data_clk_mhz=int(data_clk.group(1)) if data_clk else None)
        problems = []
        if row['xclbinutil_uuid'] is None or canonical_uuid(row['xclbinutil_uuid']) != canonical_uuid(args.uuid):
            problems.append(f'xclbinutil UUID {row["xclbinutil_uuid"]} is not {args.uuid}')
        if row['xclbinutil_data_clk_mhz'] != 100:
            problems.append(f'xclbin DATA_CLK {row["xclbinutil_data_clk_mhz"]} MHz')
        if now['clock_hz'] != got['clock_hz'] or now['clock_hz'] != CLOCK_HZ:
            problems.append(f'counter now {now["clock_hz"]} Hz')
        row.update(status='pass' if not problems else 'fail', problems=problems)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(row, indent=1, default=str) + '\n')
        print(row['status'], problems)
        sys.exit(1 if problems else 0)


if __name__ == '__main__':
    main()
