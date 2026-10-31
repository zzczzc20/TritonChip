# X8 row-coordinate reuse: mixed TLM result

`doc/engineering_discipline.md` applies. The saved `candidate.patch.gz` is an experiment against source commit `8b5f7d5c`, not an integrated compiler change and not part of the frozen image. For `matrix_dp` targets it keeps already computed row coordinates in scalar values instead of packing and repeatedly extracting them from a vector; all legality checks, fallback gathers, transfer ordering and arithmetic remain present. The trade-off is longer scalar live ranges, so allocation failures and regressions must be checked before considering integration.

On GB10 the base and candidate compile-only jobs both completed for `1x256x256` and `16x2048x2048` Hub cases with profile `core-vrf-sync-v5-x8-g2c4-dp`. For the final dumped block-dynamic kernel, `tools/r41/loop_issue_count.py` counted 8,669 instructions in the base and 8,628 in the candidate, each with 32 matrix commands. This is an assembly instruction count, not measured speed or board evidence. Raw output `/home/zhongchun/r41-x8-loop-comparison.txt`, SHA-256 `37e8a7509dd5fc74f98c60fd1fe877f6428fab448de8259ac92a0ae8a528b48a`. Base assembly SHA-256 `2bdf31340614f2789d31c80749fe250b319192aa5f4bda751cc5b26a10bc13b6`; candidate assembly SHA-256 `f64f469c14fc18d3b4e61d671bf7873f9d85be15400d6e568c41772827575981`, under the corresponding `~/r41-x8-{base,candidate}-compile/w8a8_block_dynamic_fp8_matmul_kernel/` directories.

The existing registered `fp8_kernels_cases.py --group v4 --quick` set was run on the uncalibrated TLM model, separately for base and candidate (`r41-x8-base-tlm-retry-20261029.service` and `r41-x8-candidate-tlm-retry-20261029.service`). It uses non-default IEEE division and extended deadlines set by the harness; it is not a board or default-setting performance result. Compare every case's exact-reference result and device cycles, including tails, and perform the default and dp-profile non-regression audits before integration. Source directories, model configuration, invocation and output paths are recorded by the services.

Both first TLM launches failed before device execution because CUDA was visible and Triton selected two active drivers. Their preserved GB10 logs are `~/r41-x8-base-tlm-failure.log`, SHA-256 `41ad783782ad16caf669496bdd838a4a08d6c81b3e9835ea6eedd7305c6a2839`, and `~/r41-x8-candidate-tlm-failure.log`, SHA-256 `8a2a91d5dfc006a207ce7ba5edaec447a88d172416c3ffd9999f0a2f9d643852`; the retries set `CUDA_VISIBLE_DEVICES` empty. The retries completed as described below.

## Completed comparison — 2026-10-29 UTC

Both retries exited 0 and passed all 12 cases against the registered exact references. The following values are per-launch TLM device cycles, base → candidate; they are not board measurements.

| Case | Base → candidate cycles |
| --- | --- |
| quant 4x256 | 2079 → 2079 |
| quant 7x384 | 3798 → 3798 |
| quant 129x256 | 44507 → 44507 |
| block dynamic 1x256x256 | 167025 → 166941 |
| block dynamic 7x136x144 | 185079 → 176310 |
| block dynamic 16x256x256 | 195082 → 186679 |
| block dynamic 33x256x384 | 1080699 → 1032773 |
| block dynamic 129x200x328 | 4597990 → 4392886 |
| block static 7x136x144 | 118337 → 116213 |
| block static 129x200x328 | 3317371 → 3265801 |
| tensor dynamic 7x256x256 | quant 8367 → 8367; matmul 99972 → 100204 |
| tensor dynamic 129x200x328 | quant 369139 → 368110; matmul 2729037 → 2723793 |

The 7x256x256 tensor-dynamic matmul regressed by 232 cycles (about 0.23%), so the result is mixed and the candidate is not integrated. Further integration audits were not run. The base report is `~/r41-x8-base-tlm-retry/result/report.json` on GB10, SHA-256 `f040d37663b1652c6f0d62d4598e3d03b7eb42aaed6cb83eadd4f9fa1f0522cc`; the candidate report is `~/r41-x8-candidate-tlm-retry/result/report.json`, SHA-256 `5ce101fb5e9fd85852da88a32d9de34bfaf37bb03645831cd319858ebba44d12`.
