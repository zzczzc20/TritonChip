"""Rebuild the complete routed census and bound Stage 2 regional risk.

The F01-F10 cohort is read from the existing execution document. This audit
does not invent tasks, implement pipelines, launch Vivado or promote timing.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import sys
import tarfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'tools/r40'))
from group_failing_paths import parse_tsv, group_rows


OWNERS = {
    'F01': ['rtl/core/tnca_core.v', 'rtl/core/tnca_operand_collector.v'],
    'F02': ['rtl/core/tnca_core.v', 'rtl/core/tnca_scalar_rf.v', 'rtl/core/tnca_barrier.v'],
    'F03': ['rtl/core/tnca_vector_fpu.v', 'rtl/core/tnca_fp32_fma.v'],
    'F04': ['rtl/core/tnca_core.v', 'rtl/core/tnca_scalar_rf.v'],
    'F05': ['rtl/core/tnca_core.v', 'rtl/core/tnca_vrf_banked.v', 'rtl/core/tnca_rf_bank.v', 'rtl/core/tnca_operand_collector.v'],
    'F06': ['rtl/tnca_grid_dispatch.v'],
    'F07': ['rtl/nuca/tnca_nuca_slice.v'],
    'F08': ['xls/tnca_fma.x', 'tools/xls/config.json', 'rtl/core/tnca_fp32_fma.v', 'rtl/gen/tnca_fma_xls.v'],
    'F09': ['rtl/core/tnca_fp32_fma.v', 'rtl/core/tnca_matrix.v', 'rtl/core/tnca_acc_rf.v'],
    'F10': ['rtl/core/tnca_lsu.v', 'rtl/core/tnca_lsu_pair.v', 'rtl/noc/tnca_core_ni.v', 'rtl/noc/tnca_router.v'],
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_slr(text):
    result = {}
    for line in text.splitlines():
        fields = [part.strip() for part in line.split('|')[1:-1]]
        if len(fields) != 7 or fields[0] not in ('CLB', 'CLB LUTs', 'CLB Registers'):
            continue
        result[fields[0]] = {f'SLR{i}': dict(used=int(fields[i+1]), percent=float(fields[i+4])) for i in range(3)}
    if len(result) != 3 or 'Design State : Routed' not in text or 'xcu280-fsvh2892-2L-e' not in text:
        raise ValueError('missing or incompatible routed SLR resource report')
    return result


def validate_slr(resources, reference):
    for site, key in (('CLB', 'clb_percent'), ('CLB LUTs', 'lut_percent'), ('CLB Registers', 'ff_percent')):
        for slr, expected in reference[key].items():
            if abs(resources[site][slr]['percent']-expected) > .015:
                raise ValueError('raw SLR resource report differs from frozen A3 reference')


def check_coverage(families, units, rows, frozen):
    end = next(r for r in frozen['frontier'] if r['target_mhz'] == 100)
    if (len(rows) != end['endpoints'] or len(families) != end['families'] or len(units) != end['unit_pairs'] or
            sum(r['endpoints'] for r in families) != len(rows) or
            abs(sum(r['slack'] for r in rows)-frozen['census']['residual_after_cone_whatif']['tns_ns']) > .002):
        raise ValueError('incomplete or changed residual census population')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root', type=Path, required=True)
    ap.add_argument('--slr-report', type=Path, required=True)
    ap.add_argument('--kernel-report', type=Path, required=True)
    ap.add_argument('--ooc-audit', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    root, out = args.root.resolve(), args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    census = root/'evidence/r40/v3/final_image_candidate/census/census.tgz'
    with tarfile.open(census, 'r:gz') as archive:
        (out/'residual.tsv').write_bytes(archive.extractfile('residual_failing_endpoints.tsv').read())
    rows = parse_tsv(out/'residual.tsv')
    families, units = group_rows(rows)
    frozen_path = root/'evidence/r40/v3/final_image_candidate/census/frontier.json'
    frozen = json.loads(frozen_path.read_text())
    check_coverage(families, units, rows, frozen)
    frontier, discrepancies = [], []
    for expected in frozen['frontier']:
        mhz = expected['target_mhz']
        groups = [g for g in families if g['worst_slack_ns'] < 10-1000/mhz]
        row = dict(target_mhz=mhz, families=len(groups), endpoints=sum(g['endpoints'] for g in groups),
                   unit_pairs=len({(g['source_unit'], g['destination_unit']) for g in groups}))
        frontier.append(row)
        if any(row[key] != expected[key] for key in ('families', 'endpoints', 'unit_pairs')):
            discrepancies.append(dict(derived=row, frozen=expected))
    plan_path = root/'evidence/r40/v3/final_implementation_plan/plan.json'
    baseline = json.loads(plan_path.read_text())['baseline']
    source_reference = json.loads((root/'spec/r41/stage1/historical_ooc_sources.json').read_text())
    if sha(args.slr_report) != source_reference['a3_routed_slr_report_sha256']:
        raise ValueError('raw routed SLR resource report identity changed')
    regions = parse_slr(args.slr_report.read_text())
    validate_slr(regions, baseline['utilization_routed'])
    ooc = json.loads(args.ooc_audit.read_text())
    if ooc['status'] != 'historical-source-equivalence-pass':
        raise ValueError('M1-M4 source equivalence must be verified before reusing OOC')
    key = 'evidence/r40/v3/final_image_candidate/kernel_audit/kernel_audit.tgz'
    if sha(args.kernel_report) != ooc['archives'][key]['reports']['kernel/utilization.rpt']['sha256']:
        raise ValueError('kernel synthesis resource report changed')
    execution = root/'doc/r41_execution_tasks.md'
    candidates = []
    for line in execution.read_text().splitlines():
        match = re.match(r'^\| (F\d\d) \| (.*?) \| (.*?) \|$', line)
        if not match:
            continue
        identifier, structure, verification = match.groups()
        owners = OWNERS[identifier]
        if any(not (root/path).is_file() for path in owners):
            raise ValueError('candidate refers to a missing responsible source module')
        candidates.append(dict(id=identifier, structure=structure, verification=verification,
            responsible_sources={path: sha(root/path) for path in owners},
            state='proposed, not implemented or physically qualified',
            reference_candidate='658c811' if identifier == 'F07' else None))
    if {r['id'] for r in candidates} != set(OWNERS):
        raise ValueError('the fixed cohort differs from the existing execution document')
    for source, name in ((args.slr_report, 'a3-slr-util-routed.rpt'),
                         (args.kernel_report, 'm1-m4-kernel-utilization.rpt')):
        (out/name).write_bytes(source.read_bytes())
    report = dict(schema='tnca-r41-frequency-preflight/1', status='completed-with-stage2-blockers',
        source_identity=dict(rtl_tree=ooc['rtl_tree'], ooc_reference=ooc['reference_commit']),
        source_sha256=dict(census=sha(census), frontier=sha(frozen_path), existing_plan=sha(plan_path),
                           existing_execution_tasks=sha(execution), slr_report=sha(args.slr_report),
                           kernel_resource_report=sha(args.kernel_report)),
        census_scope='A3 routed DCP what-if exclusion of M1-M3 cone; not a routed census of new RTL',
        complete_residual=dict(endpoints=len(rows), families=len(families), unit_pairs=len(units),
                               tns_ns=round(sum(r['slack'] for r in rows), 3)),
        frontier=frontier, frontier_discrepancies=discrepancies,
        family_groups=families, unit_groups=units, fixed_candidate_cohort=candidates,
        historical_routed_regions=regions,
        region_scope='A3 routed placement as a risk bound; M1-M4 synthesis resource report is unplaced',
        stage2_target_mhz=100, stage1_new_bitstream_budget=0,
        stage2_ready=False,
        stage2_blockers=['F01-F10 implementations, module screens and integrated functional verification remain required',
                        'the named cohort is not proof that every residual family is cut; the complete per-family population remains a required screening input',
                        'SLR0 historical CLB packing is 95.24 percent although LUT/FF usage is lower; added FFs do not prove placement feasibility',
                        'C1 source and functional evidence exist but its physical screen is still unperformed',
                        'candidate timing/area/SLR/SLL deltas and regenerated timing census must be measured before full implementation',
                        'actual routed 100 MHz setup/hold, route legality, CDC and board clock remain Stage 2 requirements'],
        signoff='diagnostic preparation only; no physical-frequency capability or stage acceptance promoted')
    (out/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({key: report[key] for key in ('status', 'complete_residual', 'frontier_discrepancies', 'stage2_ready')}, indent=2))


if __name__ == '__main__':
    main()
