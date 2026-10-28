# FPGA-host launch records for the P2/P4 and chat queues

These files are the record of who launched the R41 P2/P4 board queues and the full-model chat queue, and how. They answer audit findings R5 and R14 (`evidence/r41/hbm_bw/AUDIT.md`). They were copied read-only from the FPGA host `acf3019` (user `zzhouch`) on 2026-10-28. No card access was made to collect them.

| Archive | sha256 |
|---|---|
| `fpga_host_launch_records.tgz` | `d208c99d5ca232b1567f8963b2664a0ef94c0f28169a0fd27cd0405fc600851f` |
| `fpga_host_torchenv_wrapper.tgz` (`~/bin/torchenv-python`) | `3e23aeaeaf4929daf9ed5c80a896196d041e6521c6b16ad7d8c0c0f6675dde83` |

Host-side sha256 of the main launch files, taken on the host before copying:

| Host path | sha256 |
|---|---|
| `~/r41_p2p4_launch.log` | `c467480b9a01ba595907375bb8cb12db95d9f9c41236976de227e48d6b92156d` |
| `~/r41_p2p4_launch2.sh` | `83e48b1587922ec66b1b9eb9bfa3a7df876a3bfbf570f2a6336bcdb1c487356b` |
| `~/r41_p2p4_launch2.log` | `1c9fe5d07ea82354a8492f16f8a5670ff27f4875d2e53fd135eb646af1ef63a3` |
| `~/r41_chat_launch.log` | `33f54206989f15a9ad42950d99ec1da555b2b0cca65242debac00658402fa87e` |
| `~/bin/torchenv-python` | `83b3a36925333c62ae03884d78a0e787665cc19a8381854924ee1b7028cdb100` |

## What the records show

All times are UTC, as written in the logs. The lead (Claude) launched all of the runs below on the FPGA host; the source is the lead's rectification brief of 2026-10-28, which names `~/r41_p2p4_launch2.sh`, `spec/r41/chat/board_queue.sh` and these logs as the record. The scripts' own headers read "Lead-only". The images loaded are the approved run-1 100 MHz image (`30e79769…`, UUID `49964b17-…`), under the 15-minute idle rule enforced by each script.

1. **P2/P4 try 1** (`r41_p2p4_launch.sh`, `r41_p2p4_launch.log`). The card was idle for 15 minutes at 2026-10-27T09:19:31Z, holding the B40 image (`1bada9db-…`). The script loaded the run-1 image through `program.sh --check` and `--go`, and started the queue at 09:19:45Z. The queue exited with status 1 at 09:34:48Z. Cause: the interpreter `~/tc_env/torchenv/bin/python` resolved to the system Python without `torch` (`ModuleNotFoundError: No module named 'torch'` in the `s2-p2-locality` step; the failed report and stderr are in the archive under `r41_accept_p2_p4_7b10e7b/`). This failure was not archived before; it is preserved here.
2. **P2/P4 try 2** (`r41_p2p4_launch2.sh`, `r41_p2p4_launch2.log`). The only change from try 1 is the interpreter: the wrapper `~/bin/torchenv-python`, which runs `exec /home/zzhouch/tc_env/torchenv/bin/python "$@"` (archived in `fpga_host_torchenv_wrapper.tgz`). After 15 idle minutes the card already held the run-1 image (logged at 11:19:28Z), so no load was needed. The queue ran from 11:19:28Z and exited 0 at 11:51:37Z. Its output is `board-p2p4-7b10e7b-try2.tgz` (one directory up).
3. **Chat queue, 68b9191 source** (`r41_chat_launch.sh`, `r41_chat_launch.log`, `r41_chat_68b9191_out*`). Started at 2026-10-27T13:03:56Z on the run-1 image. The script does not load an image. The wrapper's log line reads "chat queue exit 0" at 13:18:58Z, but the queue output records a failure: `experiment.py` refused the manifest (`ValueError: a board experiment must declare its expected counter changes`), and `acceptance_queue.py` exited with a `CalledProcessError`. This run measured nothing. The repository already keeps it as `evidence/r41/chat/board-68b9191-preflight-failure.tgz`.
4. **Chat queue, 51aef19 source.** Launched on 2026-10-28 through `spec/r41/chat/board_queue.sh` (the source and queue checks, `r41_chat_51aef19_*_check.json`, were written at 07:19Z). Its log is `~/r41_chat_51aef19_out.log` on the host. At 08:12Z on 2026-10-28 the queue was still running (`chat.py --model vl --device tnca`), so no result is recorded here.

The lead's brief gives 2026-10-27 and 2026-10-28 as the launch dates. The logs show that both P2/P4 tries and the 68b9191 chat launch were on 2026-10-27 (UTC), and that the 51aef19 chat launch was on 2026-10-28.
