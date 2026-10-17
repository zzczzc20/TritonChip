#!/usr/bin/env python3
"""Check diagnostic storage accounting without opening a Vivado checkpoint."""
import csv
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / 'flow/fpga/diagnose_r40_checkpoint.tcl'


class StorageAccounting(unittest.TestCase):
    def report(self, rows, out):
        # Only the pure Tcl reporting procedures precede this argument guard.
        procedures = SCRIPT.read_text().split('\nif {[llength $argv] ni {2 3}}', 1)[0]
        data = '\n'.join('{' + ' '.join(map(str, row)) + '}' for row in rows)
        body = (procedures + '\nset rows {\n' + data + '\n}\n' +
                'if {[catch {r40_tag_storage_report $rows {' + str(out/'cells.tsv') +
                '} {' + str(out/'summary.tsv') + '}} message]} ' +
                '{puts stderr $message; exit 1}\n')
        return subprocess.run(['tclsh'], input=body, text=True,
                              capture_output=True, timeout=15)

    def test_macro_children_are_not_added_to_storage_roots(self):
        # Hierarchical RAM ratios observed in the retained three-cache OOC
        # screen: 96 RAM32M16 roots, 1,344 RAMD32 and 192 RAMS32 children.
        # From evidence/r40/v2/convergence/attempt02/preflight.json. Keep this
        # small regression fixture usable in source exports without evidence/.
        observed = {'RAM32M16': 96, 'RAMD32': 1344, 'RAMS32': 192}
        rows = []
        for core in range(3):
            cache = f'inst/u_system/g_tile[{core}].u_tile/u_slice'
            for way in range(8):
                for bank in range(4):
                    name = f'{cache}/g_sync_tags.g_way[{way}].bank_reg_{bank}'
                    rows.append((name, 'RAM32M16', 0))
                    rows.extend((f'{name}/RAMD{i}', 'RAMD32', 1) for i in range(14))
                    rows.extend((f'{name}/RAMS{i}', 'RAMS32', 1) for i in range(2))
                    rows.append((f'{name}/decode', 'LUT1', 1))
        rows.append(rows[0])  # Repeated canonical cells must not inflate counts.
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            result = self.report(rows, out)
            self.assertEqual(result.returncode, 0, result.stderr)
            with (out/'summary.tsv').open() as stream:
                summary = list(csv.DictReader(stream, delimiter='\t'))
            totals = {}
            for row in summary:
                key = row['scope'], row['primitive']
                totals[key] = totals.get(key, 0) + int(row['unique_cells_DO_NOT_SUM_scopes'])
            self.assertEqual(totals['storage_root', 'RAM32M16'], observed['RAM32M16'])
            self.assertEqual(totals['storage_child', 'RAMD32'], observed['RAMD32'])
            self.assertEqual(totals['storage_child', 'RAMS32'], observed['RAMS32'])
            self.assertEqual(len({row['cache'] for row in summary}), 3)
            with (out/'cells.tsv').open() as stream:
                inventory = list(csv.DictReader(stream, delimiter='\t'))
            for row in inventory:
                if row['scope'] == 'storage_child':
                    self.assertTrue(row['storage_parent'].endswith(tuple(f'bank_reg_{i}' for i in range(4))))

    def test_exploded_ram_and_register_fallback_remain_visible(self):
        rows = [('g_sync_tags.g_way[0].bank_reg', 'RAMD32', 1),
                ('g_sync_tags.g_way[1].bank_reg[0]', 'FDRE', 1),
                ('g_sync_tags.g_way[1].bank_reg_mux', 'LUT6', 1)]
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            result = self.report(rows, out)
            self.assertEqual(result.returncode, 0, result.stderr)
            with (out/'cells.tsv').open() as stream:
                inventory = list(csv.DictReader(stream, delimiter='\t'))
            self.assertEqual({(row['scope'], row['primitive']) for row in inventory},
                             {('storage_root', 'RAMD32'), ('storage_flop', 'FDRE'), ('helper', 'LUT6')})
            self.assertTrue(all(row['cache'] == '<top>' for row in inventory))

    def test_missing_or_conflicting_storage_is_refused(self):
        fixtures = [[], [('g_sync_tags.bank_reg_mux', 'LUT6', 1)],
                    [('g_sync_tags.bank_reg', 'FDRE', 1),
                     ('g_sync_tags.bank_reg', 'RAMD32', 1)]]
        for rows in fixtures:
            with self.subTest(rows=rows), tempfile.TemporaryDirectory() as tmp:
                out = Path(tmp)
                self.assertNotEqual(self.report(rows, out).returncode, 0)
                self.assertFalse((out/'cells.tsv').exists())


class FinalBoundaryReports(unittest.TestCase):
    def run_reports(self, out, missing_pair=False, missing_cut=False):
        checkpoint = out.parent/'fixture.dcp'
        checkpoint.touch()
        # Exercise the production report orchestration under Tcl. A real
        # retained module DCP separately checks the Vivado API and filters.
        body = r'''
proc set_param {args} {}
proc open_checkpoint {args} {}
proc get_cells {args} {
    global missing_pair
    set pattern [lindex $args end]
    foreach name {read_tag_reg victim_owner_q_reg request_owner_q_reg victim_pair_q_reg selected_way_q_reg} {
        if {[string first $name $pattern]>=0} {
            if {$missing_pair && $name eq "victim_pair_q_reg"} {return {}}
            return [list $name]
        }
    }
    return {g_sync_tags.bank_reg}
}
proc get_property {property cells} {
    switch $property {
        NAME {return $cells}
        REF_NAME {if {[lindex $cells 0] eq "g_sync_tags.bank_reg"} {return RAM32M16}; return FDRE}
        IS_PRIMITIVE {return 0}
    }
}
proc report_timing {args} {
    global missing_cut
    set f [open [lindex $args end] w]
    if {[string match */owner_to_selected.rpt [lindex $args end]] ||
        ($missing_cut && [string match */owner_to_pair.rpt [lindex $args end]])} {
        puts $f {No timing paths found.}
    } else {
        puts $f {Slack (MET) : 0.500ns}
        puts $f {  Source: source/Q}
        puts $f {  Destination: destination/D}
        puts $f {  Data Path Delay: 9.000ns (logic 2.000ns (22%) route 7.000ns (78%))}
        puts $f {  Logic Levels: 18}
    }
    close $f
}
'''
        body += '\nset missing_pair ' + str(int(missing_pair))
        body += '\nset missing_cut ' + str(int(missing_cut))
        body += '\nset argv [list {' + str(checkpoint) + '} {' + str(out) + '} final-cache-cuts]\n'
        body += 'if {[catch {source {' + str(SCRIPT) + '}} message]} {puts stderr $message; exit 1}\n'
        return subprocess.run(['tclsh'], input=body, text=True,
                              capture_output=True, timeout=15)

    def test_final_reports_keep_registered_segments_and_absent_direct_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)/'reports'
            result = self.run_reports(out)
            self.assertEqual(result.returncode, 0, result.stderr)
            with (out/'register_boundaries.tsv').open() as stream:
                kinds = {row['boundary'] for row in csv.DictReader(stream, delimiter='\t')}
            self.assertEqual(kinds, {'tag', 'victim_owner', 'request_owner', 'victim_pair', 'selected_way'})
            for name in ('owner_to_pair', 'pair_to_selected', 'selected_output'):
                with (out/(name+'_ranked.tsv')).open() as stream:
                    rows = list(csv.DictReader(stream, delimiter='\t'))
                self.assertEqual(rows[0]['logic_levels'], '18')
            self.assertIn('No timing paths', (out/'owner_to_selected.rpt').read_text())
            self.assertFalse((out/'owner_to_selected_ranked.tsv').exists())
            # Existing reports, including a failed extraction, are immutable.
            self.assertNotEqual(self.run_reports(out).returncode, 0)

    def test_missing_pair_boundary_is_not_success(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)/'reports'
            result = self.run_reports(out, missing_pair=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('pair-minimum', result.stderr)
            self.assertFalse((out/'owner_to_pair.rpt').exists())

    def test_missing_required_segment_preserves_failed_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)/'reports'
            result = self.run_reports(out, missing_cut=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('no ranked timing paths for owner_to_pair', result.stderr)
            self.assertIn('No timing paths', (out/'owner_to_pair.rpt').read_text())


if __name__ == '__main__':
    unittest.main()
