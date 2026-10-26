# Extended host-wait retry

The unchanged capacity child ran successfully with TNCA_HOST_TIMEOUT=172800
and the outer subprocess timeout=172800. Hardware command deadlines and
numerical checks were unchanged. The driver log records returncode 0 and correct
pointwise/reduction results for three 128 KiB tensors above a held 24 MiB pad.

The original 108/109 HBM run remains failed and preserved. This isolated retry
is not a rerun of the full gate. It supports investigating host waiting/runtime
rather than treating the original missing traceback as a proven arithmetic
failure. It does not establish the root cause of the earlier failure or the
separate instrumented SIGSEGV. Simulation performance remains an open issue.

Source tree: `~/tritonchip-r40-bar-full`, frozen 576b310 archive plus barrier48
and axis0-reduction changes. The exact retry driver is included.
