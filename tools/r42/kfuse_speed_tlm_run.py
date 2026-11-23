#!/usr/bin/env python3
"""Run ArchBench points through the SystemC/TLM model with the compiler's DEFAULT knobs (domain: model prediction).

    python tools/r42/kfuse_speed_tlm_run.py [run_graphs.py arguments]

A thin wrapper of tools/performance_harness/systemc/run_graphs.py: it clears the `B40N` knob dictionary that
run_graphs.py sets for the delivered R40 configuration (TNCA_MAX_BLOCK, TNCA_SDPA, TNCA_PIPELINE, ...), so that the
compiler of the checkout runs with its own defaults, as the board A/B records of evidence/r42/board do (their knob
list is only TNCA_ACTIVE_CORES, TNCA_TRANSFER_POLICY=resident, TNCA_TORCH_DEVICE=native, the clock evidence).
Nothing else of run_graphs.py is changed.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SC = ROOT / 'tools/performance_harness/systemc'
sys.path[:0] = [str(SC), str(ROOT / 'tools/performance_harness')]
import run_graphs   # noqa: E402

run_graphs.B40N.clear()
if __name__ == '__main__':
    sys.argv[0] = str(SC / 'run_graphs.py')
    run_graphs.main()
