#!/usr/bin/env python3
"""Exercise the real placement collector's refusal paths with Tcl query stubs.

These are collector tests, not physical placement evidence.
"""
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT/'flow/fpga/audit_r40_placed_cores.tcl'


@unittest.skipUnless(shutil.which('tclsh'), 'Tcl interpreter unavailable')
class ActualPlacementCollectorTests(unittest.TestCase):
    def collect(self, mode, prior=False):
        with tempfile.TemporaryDirectory() as temp:
            report = Path(temp)/'cores.rpt'
            if prior:
                report.write_text('prior evidence\n')
            wrapper = Path(temp)/'query_stubs.tcl'
            wrapper.write_text(r'''
proc open_checkpoint {path} {}
proc get_cells {args} {
    set out [list]
    for {set core 0} {$core < 3} {incr core} {
        if {$::mode eq "missing" && $core == 2} {continue}
        foreach kind {logic constant macro} {
            lappend out [format {top/u_system/g_tile[%d].u_tile/%s} $core $kind]
        }
    }
    if {[string match local_* $::mode]} {
        for {set replica 0} {$replica<3} {incr replica} {
            if {$::mode eq "local_missing" && $replica==2} {continue}
            foreach kind {logic macro} {
                lappend out [format {top/u_system/g_local_tmu.u_tmu/g_replica[%d].local_tmu/%s} $replica $kind]
            }
        }
    }
    return $out
}
proc get_sites {args} {
    set out [list]
    foreach object [lindex $args end] {
        if {[string match */constant $object]} {continue}
        if {$::mode eq "unplaced" && [string first {g_tile[2]} $object]>=0} {continue}
        if {$::mode eq "ram_unplaced" && [string first {g_tile[2]} $object]>=0 && [string match */macro $object]} {continue}
        if {$::mode eq "local_unplaced" && [string first {g_replica[2]} $object]>=0} {continue}
        lappend out "SITE:$object"
    }
    return $out
}
proc get_property {property objects} {
    set out [list]
    foreach object $objects {
        switch $property {
            NAME {lappend out $object}
            REF_NAME {
                if {[string match */constant $object]} {lappend out GND
                } elseif {[string match */macro $object]} {lappend out RAM256X1S
                } else {lappend out LUT6}
            }
            IS_PLACED {
                # Observed Vivado 2023.2 RAM macro: physical site exists,
                # IS_PLACED is empty and direct cell->SLR is unavailable.
                if {[string match */macro $object]} {lappend out {}
                } else {lappend out [expr {!($::mode eq "unplaced" && [string first {g_tile[2]} $object] >= 0)}]}
            }
            default {error "unexpected property $property"}
        }
    }
    return $out
}
proc get_slrs {flag objects} {
    if {$flag ne "-of_objects"} {error "expected an actual object query"}
    set regions [list]
    foreach object $objects {
        if {![string match SITE:* $object]} {error "expected actual mapped sites"}
        if {![regexp {g_tile\[([0-2])\]} $object whole core]} {
            regexp {g_replica\[([0-2])\]} $object whole core
            if {$::mode eq "local_wrong_slr" && $core==1} {set core 0}
        }
        if {$::mode eq "all_on_one_slr"} {set core 0}
        lappend regions SLR$core
    }
    return [lsort -unique $regions]
}
''' + f'\nset mode {{{mode}}}\nset argv [list mock_routed.dcp {{{report}}}]\nsource {{{SCRIPT}}}\n')
            result = subprocess.run(['tclsh', str(wrapper)], text=True, capture_output=True)
            return result, report.read_text() if report.exists() else ''

    def test_actual_three_slrs_and_unplaced_constants(self):
        result, report = self.collect('three_slrs')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('R40_ACTUAL_CORE_PLACEMENT_PASS', result.stdout)
        self.assertEqual(report.count('excluded_constants 1'), 3)

    def test_wrong_locations_missing_core_and_unplaced_logic_refused(self):
        for mode in ('all_on_one_slr', 'missing', 'unplaced', 'ram_unplaced'):
            with self.subTest(mode=mode):
                result, report = self.collect(mode)
                self.assertNotEqual(result.returncode, 0)
                self.assertNotIn('R40_ACTUAL_CORE_PLACEMENT_PASS', result.stdout)
                self.assertIn('failing_cores', report)

    def test_located_ram_macros_and_local_state_use_actual_sites(self):
        result, report = self.collect('local_copies')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(report.count('placed 2 unplaced 0'), 6)
        self.assertIn('local_tmu_replica_count 3', report)

    def test_misplaced_incomplete_and_unplaced_local_copies_refused(self):
        for mode in ('local_wrong_slr', 'local_missing', 'local_unplaced'):
            with self.subTest(mode=mode):
                result, report = self.collect(mode)
                self.assertNotEqual(result.returncode, 0)
                self.assertNotIn('R40_ACTUAL_CORE_PLACEMENT_PASS', result.stdout)
                self.assertIn('failing_cores', report)
                self.assertIn('local_tmu', report.split('failing_cores')[-1])

    def test_prior_evidence_cannot_be_overwritten(self):
        result, report = self.collect('three_slrs', prior=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(report, 'prior evidence\n')


if __name__ == '__main__':
    unittest.main()
