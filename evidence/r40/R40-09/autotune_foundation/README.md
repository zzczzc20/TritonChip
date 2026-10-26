# Autotune foundation — functional checkpoint, not accepted

The unchanged standard Torch pointwise graph runs on three physical RTL cores,
three independent memory outlets and two-stage links through the board binding.
Candidates select one or three active cores. The eight-core candidate exceeds
the actual core capacity and is refused before launch. Inputs and outputs are
native resident tensors. Selection uses device-clock endpoints, never simulator
wall time. This is not a frozen holdout or a board performance experiment.

`policy.log` covers resource/capability filtering, sixteen candidates per kernel,
shared thread admission, graph-budget overrun, full configuration cache identity,
common-clock span ranking and invalid timing rejection. `grid_clock.log` covers
the runtime physical-grid Event bridge, including unavailable old-backend clocks.

`native_initial.log` records the initial integration run and 16 common-clock tests
in the locked Torch environment. That native test preceded source-fingerprint
reporting and per-candidate output checks. Its synthetic identity labels must
not be used as formal cache provenance. The final batch log, when retained here,
records the source/RTL/harness digests and each candidate's actual endpoints.

Run from a source checkout with the locked Torch/Triton environment and Icarus:

```sh
export PYTHONPATH=python:compiler/python:tools/asm
export CUDA_VISIBLE_DEVICES=
make r40-autotune-checks TORCH_PYTHON="$HOME/tc_env/torchenv/bin/python"
```

The software budget reports an in-flight callback overrun and excludes that
sample; it does not forcibly cancel compiled device execution. No long-graph
budget exception is claimed. Thread admission shares one GraphBudget instance;
cross-process workers and durable cache coordination remain open. Automatic
Inductor candidate integration, the cost model, compile/tuning amortization,
holdout/regret, full suites and final board/model evidence remain incomplete.
