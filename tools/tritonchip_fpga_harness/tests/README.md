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

The complete offline suite, including mock package/link lifecycle and the
zero-exit-code-without-xclbin regression, is:

```sh
python3 -m unittest discover -s tools/tritonchip_fpga_harness/tests -p 'test*.py'
```

It runs without Vivado, Vitis, a platform installation, or an FPGA license.
