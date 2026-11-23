# kernels-community/finegrained-fp8, v4, `torch-cuda` build: an unchanged copy

The nine Python files and `metadata.json` of `build/torch-cuda` of https://huggingface.co/kernels-community/finegrained-fp8
(Hub branch `v4`, kernel commit `fe2bb66fee6c2a77eccf08a0508dc57428d488ed`, the build transformers 5.20.2 loads and
evidence/r41/fp8 validated), byte for byte, Apache License 2.0 (`LICENSE`; each file keeps its header). They are here
so that a host without a CUDA build of torch (the FPGA host, thor) and without the `kernels` package or the network
runs the same source. `archbench/fp8_hub.py` verifies every file against the sha256 pinned in `fp8_hub.FILES` before it
imports them, whichever way they are found; a file that differs is refused. Nothing in this directory is edited,
wrapped or re-implemented: the xl points call `matmul_2d` and `fp8_act_quant` and the standard path compiles them.
