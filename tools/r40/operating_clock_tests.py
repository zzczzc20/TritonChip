#!/usr/bin/env python3
"""Clock re-analysis guards; no implementation or board operation."""
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[2]
SCRIPT=ROOT/'flow/fpga/audit_r40_operating_clock.tcl'
STUB=r'''
set period 10.0
proc set_param {args} {}
proc open_checkpoint {args} {}
proc get_clocks {args} {return clock}
proc get_pins {args} {return pin}
proc get_property {args} {return $::period}
proc create_generated_clock {args} {
    set divide [lindex $args [expr {[lsearch $args -divide_by]+1}]]
    set multiply [lindex $args [expr {[lsearch $args -multiply_by]+1}]]
    set ::period [expr {10.0*$divide/$multiply}]
}
foreach op {report_clocks report_route_status report_timing_summary report_timing report_clock_interaction report_cdc check_timing report_drc} {
    proc $op {args} {set f [open [lindex $args end] w];puts $f recorded;close $f}
}
foreach op {synth_design opt_design place_design phys_opt_design route_design write_bitstream write_checkpoint} {
    proc $op {args} {error "forbidden implementation operation"}
}
'''


class ClockAnalysis(unittest.TestCase):
    def invoke(self,d,mhz):
        d=Path(d);cp=d/'routed.dcp';cp.touch()
        driver=d/'driver.tcl'
        driver.write_text(STUB+f'\nset argv [list {{{cp}}} {{{d/"audit"}}} {mhz}]\n'+
                          f'if {{[catch {{source {{{SCRIPT}}}}} why]}} {{puts stderr $why;exit 1}}\n')
        return subprocess.run(['tclsh',str(driver)],capture_output=True,text=True)

    def test_91mhz_is_analysis_only(self):
        with tempfile.TemporaryDirectory() as d:
            result=self.invoke(d,91)
            self.assertEqual(result.returncode,0,result.stderr)
            self.assertIn('analysis_period_ns\t10.989',(Path(d)/'audit/analysis.tsv').read_text())
            self.assertIn('R40_OPERATING_CLOCK_ANALYSIS_DONE',result.stdout)

    def test_invalid_frequency_refused(self):
        for frequency in (0,101,'nan'):
            with tempfile.TemporaryDirectory() as d:
                self.assertNotEqual(self.invoke(d,frequency).returncode,0)
                self.assertFalse((Path(d)/'audit').exists())

    def test_existing_evidence_preserved(self):
        with tempfile.TemporaryDirectory() as d:
            (Path(d)/'audit').mkdir();marker=Path(d)/'audit/marker';marker.write_text('retain')
            self.assertNotEqual(self.invoke(d,91).returncode,0)
            self.assertEqual(marker.read_text(),'retain')


class ClockEvidence(unittest.TestCase):
    def test_bound_evidence_and_refusals(self):
        import hashlib,json
        from operating_clock import qualified_clock
        with tempfile.TemporaryDirectory() as d:
            d=Path(d);raw=d/'raw.txt';raw.write_text('test fixture only')
            files={k:dict(path='raw.txt',sha256=hashlib.sha256(raw.read_bytes()).hexdigest())
                   for k in ('timing','route','cdc','board_clock')}
            data=dict(schema='tnca-qualified-operating-clock/1',image_sha256='a'*64,
                      qualification='conditional-r40v2-20261018',clock_hz=91_000_000,
                      setup_wns_ns=0.01,hold_whs_ns=0.001,route_legal=True,cdc_review='passed',
                      clock_source='measured-board-counter',evidence_files=files)
            path=d/'clock.json';path.write_text(json.dumps(data))
            self.assertEqual(qualified_clock(path,'a'*64)['clock_hz'],91_000_000)
            with self.assertRaises(ValueError):qualified_clock(path,'b'*64)
            for key,value in (('clock_hz',100_000_000),('setup_wns_ns',-0.001),
                              ('hold_whs_ns',float('nan')),('cdc_review','not-measured'),
                              ('clock_source','metadata')):
                bad=dict(data);bad[key]=value;path.write_text(json.dumps(bad))
                with self.assertRaises(ValueError):qualified_clock(path,'a'*64)
            path.write_text(json.dumps(data));raw.write_text('changed')
            with self.assertRaises(ValueError):qualified_clock(path,'a'*64)


if __name__=='__main__':unittest.main()
