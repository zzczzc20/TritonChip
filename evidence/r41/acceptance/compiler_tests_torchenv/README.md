# Compiler tests under the GB10 torch interpreter

Source: `b6fcac0b8ce4952f98579f9a900b17b456c74a9`, with the Makefile change in this commit applied for the after runs. Host: GB10, isolated checkout `~/r41-codex-t9-compiler`, 2026-10-28 UTC. These are compiler tests, not board measurements; image hash and device clock do not apply. The raw logs are `logs.tgz` (SHA-256 `6a3fd94f10632339cb474de3718c7ea7f78e23ea9763970d7ea07084fc1358c5`). The build log includes the completed local `tnca-opt` build against `~/tc_env/root/usr/lib/llvm-20`.

After `source ~/tc_env/env.sh`, the before and after command was:

```sh
TNCA_LLVM=$HOME/tc_env/root/usr/lib/llvm-20 make compiler PYTHON=$HOME/tc_env/torchenv/bin/python
```

- `before.log`: exit 2, 16 failures out of 157 tests. The standalone import and failure trace identify `~/tritonchip/python/tnca`, an editable install from another tree, rather than the checkout under test. The compiler unit failure stopped make before the IR tests.
- `after.log`: exit 0; 157 unit tests reported, with 53 skipped because `sim/build/tb_core.vvp` was absent; 6 IR RUN cases passed. No tests were changed.
- `after-shadow.log`: same result with `PYTHONPATH=$HOME/tritonchip/python` supplied deliberately, checking that the checkout also precedes an existing path.
- `strict.log`: the same command with `R1_STRICT=1` failed on those 53 skips, as required by the strict gate; it names each skip. `core-build.log` records the subsequent build of `sim/build/tb_core.vvp` for a further strict run.

The harness exports the current checkout's `python` and `compiler/python` ahead of any inherited `PYTHONPATH` for `compiler` and `compiler-rtl`, including child processes. It does not alter the installed environment or test expectations.

A follow-up strict run with the simulator built was still in progress when this record was updated: GB10 shell PID 2086811, raw output `~/r41-codex-t9-results/strict-core.log`. Its command is the one above with `R1_STRICT=1`; the runner buffers unit results until the suite finishes. No result is claimed for that run.
