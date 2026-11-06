# The three servers and the shared U280 card

> **Binding:** [`doc/engineering_discipline.md`](../../doc/engineering_discipline.md) (evidence integrity, the standard path, no special-casing, no constructive exceptions, no test poisoning) applies to all work and takes precedence over this file.


Every R41 session reaches the servers through helper scripts, which keep the credentials outside the repository. Do not print them, copy them into any file, or commit them.

```sh
S=/tmp/claude-1000/-home-ubuntu-TritonChip/3dd1d4a8-6f0c-47d5-bf28-f88428fc463d/scratchpad
```

| Server | Command | Role |
|---|---|---|
| **GB10** (user `zhongchun`) | `$S/gb 'command'` | Build and simulation host: iverilog, verilator, yosys (`source ~/tc_env/env.sh`), the SystemC/TLM model, Torch (`~/tc_env/torchenv/bin/python`; never pip install into it), Hugging Face work in `~/flux-venv`. It is also the jump host to the FPGA host. aarch64, so XLS does not run there. |
| **FPGA host** `acf3019` (user `zzhouch`) | `$S/fp 'command'` | The shared Alveo U280 (card BDF `0000:04:00.1`), XRT 2.16, Vitis and Vivado 2023.2 in `/tools/Xilinx`, XLS in `~/tnca_xls/.xls/cur`. 24 cores, 62 GB shared with other users. Full `v++` links and every board run happen here. |
| **x86 host** (WSL2 on Windows, user `vivado`) | `$S/vx 'command'` | CPU-only Vivado and Vitis for synthesis, place and route and full links (16 cores, 31 GB RAM plus 32 GB swap). No card. Setup: `evidence/r41/stage2/x86_link/README.md`; `source ~/x86_link_env.sh` first. |

## File transfer
- To GB10: `rsync -a --delete --exclude .git --exclude /evidence --exclude /sim -e $S/gb_rsh <worktree>/ gb10:<dir>/`.
- To the x86 host: `rsync -e $S/vx_rsh <files> vivado@143.89.78.138:<dir>/`.
- To the FPGA host: stream through GB10 (`cat file | $S/fp 'cat > ~/file'`, or `scp` from GB10 to `zzhouch@acf3019.ece.ust.hk`). Fetch results back with `$S/fp 'tar czf - <paths>' > file`.
- The FPGA host cannot pull from GitHub, so sources go there as a `git archive` tarball.

## Limits on shared machines
- GB10 is shared: at most 6 cores and 16 GB of RAM per job, and keep 40 GB available (`free -g`). Never `pkill -f`; kill only your own processes by PID.
- Vivado on the FPGA host: run it through `~/r41s2_tools/vivado_slot.sh --min-gb 8 LOG -- command` (two slots and a memory guard). A full link needs at least 38 GB available (peak 31 GB) and about 9 hours.
- Vivado on the x86 host: at most two jobs at once, each with six threads or fewer.

## The shared U280 card
Other users (`eeyhui`, `cliufc`) run long jobs on the card. These rules are not negotiable:
- Never load, reset or open the card unless it has had no context, no outstanding exec and no client process for 15 consecutive minutes (`/sys/bus/pci/devices/0000:04:00.1/kdsstat`). Never override anyone's run.
- A **new** image needs explicit approval from the user. Images already approved: the 90 MHz B40 image (`a88ebdf5…`, UUID `1bada9db-78f3-5bc8-7ba6-cce062e7ade8`) and the 100 MHz run 1 image (`30e79769…`, UUID `49964b17-e0b4-5bd2-dd40-7fc37b1205d3`). The loaded image can be changed by others at any time; the harness checks the UUID and clock around every step and stops when they change.
- Load with `flow/fpga/program.sh --check` and then `--go`, after `export TNCA_BOARD=u280` and `source /tools/Xilinx/Vitis/2023.2/settings64.sh; source /opt/xilinx/xrt/setup.sh`. Without these, `program.sh` checks a U50 and refuses.
- Board work runs through `tools/performance_harness/experiment.py` manifests (`spec/r41/...`), launched detached on the FPGA host by a waiting script that enforces the 15-minute idle rule: `evidence/r41/stage1/launches/board_queue.sh` and `evidence/r41/stage2/board/launches/`. A manifest that fails keeps its failure; re-running it needs `--retry-failed`, and a preserved failed point also needs its entry removed from the batch `state.json` (keep a backup).

## Budget of full image runs
The author allows at most five full synthesis + place-and-route runs for Stage 2 in total. One is used (run 1, 100 MHz, 2026-10-22). Unit, kernel and tile screens are free. Ask the lead before starting one, and record it as n/5.
