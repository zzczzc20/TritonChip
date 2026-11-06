#!/usr/bin/env python3
"""Record the individually reviewed platform-scoped candidate CDC findings.

This disposition is conditional on the existing vendor platform recovery and
reset assumptions. It does not make the raw Critical severities disappear,
prove glitch-free asynchronous aggregation, waive timing, or authorize load.
"""
import gzip
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[3]
E = ROOT/'evidence/r41/hbm_bw/g4/run3/route_prepackage'
BASELINE = ROOT/'evidence/r40/v2/convergence/attempt03/qualified_clock90_after_load.json'
IMAGE_SHA = '791b8af28f2583e4b29cd4bb191d44e8692bed41b7132c493d86818cb0c7ee09'
CDC_SHA = '8c98866634215f258f21c6baf2b1a3c848425ae65b50289a7ccaf4679dcae529'
EXPECTED = {
    BASELINE:'853126752a292593e261ae8ec78c255afb9364666353a20e0ee22196f214d9e8',
    E/'platform_irq_trace.txt':'2b13c1d5191c0693b057ca5cce743357900d1a8c394b51d0096bea17f66f2f59',
    E/'platform_irq_segments.txt':'ff70d990058bae3cabb246ad2cacc60e27ce015ec9ea7893d8dde54e70de9976',
    E/'platform_irq_logic.txt':'c2018fea38f9eac440dbf444192db5eb3828fc326fd9c3ea4e39d8f0deacf682',
}
PREFIX = 'level0_i/blp/blp_i/PLP/'
NEW = ('CDC-10', PREFIX+'ss_fic/inst/user_fanout_firewall_and_regslices/axi_firewall_ctrl_user/inst/gen_mi.mi_r_error_i_reg/C',
       PREFIX+'ss_dma/inst/axi_intc_aggregated_mgmt/U0/INTC_CORE_I/INTR_DETECT_GEN[0].ASYNC_GEN.intr_ff_reg[0]/D')
OLD = ('CDC-10', PREFIX+'ss_fic/inst/data_firewalls_and_regslices/axi_firewall_data_c2h/inst/gen_si.si_r_error_i_reg/C', NEW[2])


def main():
    for path,expected in EXPECTED.items():
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError('review evidence changed: '+str(path))
    cdc = gzip.decompress((E/'cdc.rpt.gz').read_bytes())
    if hashlib.sha256(cdc).hexdigest() != CDC_SHA:
        raise ValueError('candidate CDC report changed')
    subprocess.run([sys.executable, str(ROOT/'tools/r41/run2_postlink/check_irq_sticky.py')], check=True)
    trace = (E/'platform_irq_trace.txt').read_text()
    if (any(x not in trace for x in ('source pin_count 1','destination pin_count 1','pblock_blp'))
            or not re.search(r'^ASYNC_REG\s+bool\s+\S+\s+1\s*$',trace,re.M)):
        raise ValueError('platform source or synchronizer trace incomplete')
    actual = []
    for line in cdc.decode().splitlines():
        m = re.match(r'\s*\d+\s+(CDC-\d+)\s+Critical\s', line)
        if m:
            p = line.split()
            actual.append((m.group(1),p[-2],p[-1]))
    rows = json.loads(BASELINE.read_text())['qualified_clock']['cdc_review_scope']['reviewed_critical']
    prior = {(r['rule'],r['source'],r['destination']):r for r in rows}
    if len(actual)!=56 or len(set(actual))!=56 or len(prior)!=56:
        raise ValueError('critical findings are missing or duplicated')
    if set(actual)-set(prior)!={NEW} or set(prior)-set(actual)!={OLD}:
        raise ValueError('CDC endpoint differences require a new review')
    reviewed = []
    for rule,source,destination in actual:
        key = (rule,source,destination)
        if key == NEW:
            reason = ('Vendor MI read-error level into the same management IRQ synchronizer as the prior SI error. '
                      'The implemented Q-feedback LUT retains an asserted error until the verified unblock-state control or reset; '
                      'the destination has a two-stage synchronizer and the unchanged platform cone aggregates sticky errors with OR. '
                      'Conditional on the vendor recovery protocol capturing/handling each error before unblock and coordinated resets. '
                      'A minimum pulse width and glitch-free simultaneous assertion/clear are not independently proven; '
                      'the original Critical severity remains visible.')
        else:
            reason = prior[key]['review']
        reviewed.append(dict(rule=rule,source=source,destination=destination,severity='Critical',
                             disposition='accepted-platform-scoped',review=reason))
    result = dict(schema='tnca-candidate-cdc-review/1',kind='cdc-review',status='pass',
                  qualification='accepted-for-conditional-operation-with-scoped-platform-findings',
                  image_sha256=IMAGE_SHA,cdc_report_sha256=CDC_SHA,raw_critical=56,
                  review_scope='Unmodified vendor platform only; no TNCA custom endpoint. '
                               'Existing platform recovery/reset assumptions apply; raw severity retained.',
                  limitations=['No independent minimum-pulse or simultaneous-clear glitch proof for the IRQ aggregation.',
                               'Do not unblock an error before it is captured and handled; coordinate platform resets.',
                               'This is a scoped vendor-IP contract review, not a blanket CDC waiver.'],
                  sources=['https://docs.amd.com/r/en-US/pg293-axi-firewall/Port-Descriptions',
                           'https://docs.amd.com/r/en-US/pg293-axi-firewall/Recovery',
                           'https://docs.amd.com/v/u/en-US/pg099-axi-intc'],
                  evidence_sha256={str(p.relative_to(ROOT)):v for p,v in EXPECTED.items()},
                  reviewed_critical=reviewed)
    out = E/'candidate_cdc_review.json'
    with out.open('x') as f:
        f.write(json.dumps(result,indent=2)+'\n')
    print('56 individually reviewed critical rows; conditional platform scope; no image loaded')


if __name__ == '__main__':
    main()
