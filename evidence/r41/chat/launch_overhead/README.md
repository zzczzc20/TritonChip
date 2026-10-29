# R41 decode launch and code-store work

Binding: `doc/engineering_discipline.md`. Source commit: `baf2a8d2f86c6a617dd27980a8d55c3a11378640`.
All execution results below are from GB10 on 2026-10-29. No card was accessed. Raw logs and reports are in
`raw/launch-overhead.tgz`; the failed TLM attempts are retained beside the successful runtime tests.

## Changes

- `python/tnca/triton_backend/driver.py`: native resident launches now use the existing bounded asynchronous FIFO by
  default. `TNCA_SUBMIT=single` restores the old policy. Host-staged tensors retain synchronous execution. The device
  instruction stream is unchanged.
- The module loader also deduplicates byte-identical `.tnca` artifacts across temporary paths within a session epoch.
  Aliases share LRU state and are removed together on eviction.
- `tools/r40/code_cache_epoch_tests.py`: executes a copied artifact through the shared module and checks its result.

The archived TLM profile establishes a capacity limit rather than an LRU-policy bug: the final reduced-layer decode
step contains 17 distinct VL artifacts totalling 203,776 reserved code bytes and 42 distinct Qwen3.5 artifacts
totalling 365,568 bytes, against the 131,072-byte code region. This change does not claim that either working set is
now resident. Batching can overlap code placement with earlier device work, but this host's TLM does not model board
transport latency.

## Gate results

- `make -j2 compiler`: PASS, 157 compiler unit tests, 53 skipped; `compiler-ir` skipped because `tnca-opt` was not
  built. See `compiler.log`.
- `tools/performance_harness/default_change_audit.py run ... --jobs 4`, compared with
  `~/r41-fp8-audit/run-ctrl/report.json`: **113 unchanged, 0 changed, 0 not compared**. This is compile-only code and
  launch-structure equivalence, not a performance measurement. See `report.json` and `default-audit-compare.json`.
- `tools/r40/code_cache_epoch_tests.py`: PASS, seven RTL launches in a one-module code region, including the
  byte-identical artifact alias. See `code-cache.log`.
- `tools/r40/batch_tests.py --calls 20`: PASS (5/5), including bounded submission, streams, concurrent graphs and
  storage/lease audits. See `batch_tests.log`.
- `tools/r40/replay_tests.py --replays 2`: PASS (5/5), including input updates, replay without the Triton wrapper,
  guard invalidation and recovery epoch checks. See `replay_tests.log`.

## Blocked model measurements

The requested commands were started with default TNCA settings, `core-vrf-sync-v5-x8`, the recorded TLM binary,
1 VL layer / 4 Qwen3.5 layers, bucket 64, four tokens and `--session-check "Say hi."`. Both exited before loading the
model because `~/tc_env/torchenv/bin/python` has no `transformers` module. No device launch or model measurement
occurred; `vl-session-failed.log` and `q35-session-failed.log` preserve the failures. No package was installed.

Consequently there is no honest after-run launch-count, host-time, reload-count or bit-identity result to report.
Board impact is also unmeasured. A full working-set residency fix still requires smaller code or a larger code store;
artifact deduplication only removes exact duplicates.

## Reproduction and archive

Runtime tests were run after `source ~/tc_env/env.sh` with `~/tc_env/torchenv/bin/python`. The default audit candidate
is `report.json`, whose identity records the software-tree hash and run configuration. The TLM output directories are
`~/r41-k8launch-runs/vl-session` and `~/r41-k8launch-runs/q35-session` on GB10.

`raw/launch-overhead.tgz` contains the raw logs/reports. No credentials or helper-script contents are included.
