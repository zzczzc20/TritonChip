# Harness validation fixtures

`test_u50_flow_guard.py` is a vendor-free smoke test for the legacy U50
adapter boundary.  It verifies that:

* the wrapped shell entrypoint is syntactically valid;
* an empty build directory is rejected before Vitis/Vivado is sourced or
  launched; and
* the 150 MHz platform clock and divide-by-four 37.5 MHz device-clock
  contract remain visible in the inputs the harness records.

It is intentionally separate from the long FPGA build and can run on a
developer laptop:

```sh
python3 tools/tritonchip_fpga_harness/tests/test_u50_flow_guard.py
```

The test does not claim that a `.xo`, `.xclbin`, routed STA report, or board
measurement exists.  Those claims require the harness worker on the U50 host
and are represented by the implementation harness's mock and live evidence
tests.

`test_board_stage.py` runs the post-link half -- `submit` with a `board`
block, `submit --skip-board` followed by `measure`, and every failure verdict
-- in a throwaway git repository, with each vendor step replaced by a
manifest-supplied command.  Four of its tests instead read the checkout the
harness is committed in: the profiles' scripts exist, the default conformance
and ArchBench commands use options those tools accept, the example manifest
names the device profile the RTL defaults build, and the board results
recorded under `spec/r39/` summarize as a clean `measured-board` run.  They
skip in a checkout that does not carry those files.

The complete offline suite, including mock package/link lifecycle and the
zero-exit-code-without-xclbin regression, is:

```sh
python3 -m unittest discover -s tools/tritonchip_fpga_harness/tests -p 'test*.py'
```

It runs without Vivado, Vitis, a platform installation, or an FPGA license.
