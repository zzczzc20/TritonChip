# Tiled pointwise mode 2 board gate (prepared only)

**Domain: manifest preparation and FPGA-host dry-run. No board measurement was made and no card was accessed.** These experiments use the non-default `TNCA_TILED_POINTWISE=2` option as evidence for the default-change gate, not as bandwidth-target results.

The registering source is commit `f3385ab683d658e77be092783ca34c953dfdd57e`. `tools/r41/membound_manifests.py --tiled2` generates six memory-bound slices and the `s2-original-suite` copy under `spec/r41/hbm_bw/membound_tiled2/`. Structural comparison against the baseline confirms the same 513 memory-bound points and 113 non-regression points, with the same image, point order, sampling, timing boundary and cache policy; the only run-knob addition is `TNCA_TILED_POINTWISE=2`. Every manifest declares its expected counter changes and is marked `non_default`.

The launcher is `evidence/r41/hbm_bw/launches/membound_tiled2.sh`. It runs the six slices followed by the 113-point suite through `acceptance_queue.py`; it never programs or resets the card and never resolves the PyTorch interpreter symlink.

## Verification

The registering commit was copied to `acf3019:~/r41_tiled2_f3385ab6` with `git archive`. On that archive:

- `membound_manifests.py --tiled2 --check` passed.
- All seven manifests passed `acceptance_queue.py --check-only`, each reporting `card_access=false`.
- All seven manifests passed `board_batch.py --dry-run`: 626 configurations and an estimated 7.748 board hours in total. This is an estimate, not a measured result.
- Structural, Python-syntax, shell-syntax and whitespace checks passed. The baseline generator check passed on GB10 and the baseline JSON files were not changed.

Earlier failures are retained in the raw logs: the local Python lacked PyTorch, the first GB10 sync lacked the estimator's evidence input, and cross-host floating-point evaluation initially made estimate metadata differ in its final digits. The tiled2 metadata is now emitted with reproducible precision; no point, knob, image or timing declaration was changed to resolve those failures.

Raw output is in `verification-logs.tgz`, including the failed attempts and the final successful logs.

## Lead launch command

Run only on the FPGA host as the lead, after the baseline queue has finished:

```sh
C=f3385ab6
git archive $C | $S/fp "mkdir -p ~/r41_tiled2_$C && tar x -C ~/r41_tiled2_$C"
$S/fp "sh ~/r41_tiled2_$C/evidence/r41/hbm_bw/launches/membound_tiled2.sh ~/r41_tiled2_$C ~/r41_tiled2_${C}_out"
```

Progress is written to `~/r41_tiled2_${C}_out/membound_tiled2.log`; per-point results remain in each experiment's `batch.json`.
