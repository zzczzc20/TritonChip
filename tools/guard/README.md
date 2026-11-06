# R41 guard

`doc/engineering_discipline.md` remains binding. Install the checked-in hooks
after any authorized cleanup commits have been pushed: `tools/guard/install.sh`.
The installer copies them to the common Git directory and sets an **absolute**
`core.hooksPath` there, so older worktrees without these files use the same hooks.
It checks every worktree for a conflicting worktree-local setting.

Commits need a product code or test change outside `tools/guard/`, plus one
`Significance:` trailer of at least eight words naming tokens/s, chat quality,
no-CPU-fallback, image bring-up, or PR closure. Evidence additions are limited
to 200,000 bytes, 2,000 lines and five new files per commit; changed `.tgz`,
`.tar.gz`, `.zip` and `.log` files are limited to 100,000 bytes. The pre-push
hook rechecks pushed commits and rejects non-fast-forwards. Raw evidence goes to
the host archive and `evidence/r41/RAW_INDEX.md`, with hashes and failures kept.

An author may write a nonempty nonce to `/home/ubuntu/.guard/author_override`.
The next commit consumes it; `override.log` records its use and binds the
exception to the exact completed commit. It cannot authorize a
second commit. `make guard-test` tests these rules in temporary repositories.

These are **client-side** controls: Git can bypass both hooks with `--no-verify`
or a different hooks path, and other processes running as the same Unix user
can write the token or change files. Server-side protected-branch checks and a
separate author credential are required for enforcement against that user.

## Watchdog

`python3 tools/guard/watchdog.py --dry-run --history origin/r41 --hours 30`
replays commit history; it cannot reconstruct uncommitted work or past test
runs. `--dry-run --sample` measures worktrees without sending terminal input.
The synthetic tests exercise the warning sequence and reset, including the
25-minute interruption. Review the dry run and report it before starting live
monitoring. Terminal 10's goal must already be paused during setup.

All live thresholds and targets are in `/home/ubuntu/.guard/targets.json`:

```json
{
  "enabled": true,
  "targets": [{"tmux": "10", "thread_id": "THE-EXACT-CODEX-THREAD-ID"}],
  "worktree_prefix": "/home/ubuntu/TritonChip",
  "measure_seconds": 120,
  "feedback_seconds": 600,
  "minimum_lines": 30,
  "warning_seconds": [600, 1200, 1500],
  "stop_seconds": 1800,
  "resume_grace_seconds": 900,
  "assessment_seconds": 3600,
  "assessment_timeout_seconds": 600
}
```

The sensor reads tracked and untracked source contents across matching
worktrees, excluding generated, blank and comment lines. It ignores repeated
content and counts net unique line additions or first removals; this is a
conservative heuristic, not proof of significance or of who made a change.
Evidence-only churn is the fraction of active two-minute samples containing
only evidence, documentation or specification changes. A completed test or
TLM comparison can count when it produces a new result bound to a recent code
change. Successful test commands with explicit output paths in watched evidence
are observed when the previous file content is known. For other results, run
`python3 tools/guard/watchdog.py --validate-result PATH -- COMMAND ...` so the
result and source content are recorded; unobserved results are not credited.

At 10 and 20 minutes without a productive window it queues warnings; at 25 it
sends Escape, waits for the turn to stop and sends the warning as a new input.
Each warning includes the remaining minutes, the last 30 minutes' code lines,
the evidence-only share and a request for immediate work and self-assessment.
It verifies visibility on the target screen and logs retries to `warning.log`;
it refuses to overwrite an occupied input box. A productive window resets the
clock and sends “已恢复”. At 30 minutes it sends Escape and `/goal pause`, checks
the goal state and writes `stop.log` and `ALERT.txt`. Only the author resumes;
resume receives 15 minutes of grace. Hourly answers are read from that thread
and written to `significance.log`; missing answers make that window unproductive.

The monitor verifies the exact tmux session and Codex thread ID before sending
keys. It never directly signals a board queue, link or other process. Escape
interrupts the Codex turn and may also cancel that turn's foreground tool call;
detached jobs must remain independent. If the sensor, thread identity or input
box cannot be verified, it alerts rather than guessing. Hook restoration uses
`origin/r41`, including a conflicting per-worktree Git configuration; it does
not edit checkout files. Same-user modification can also tamper with the
watchdog, its state and its supervisor; this is detection, not a security boundary.

After dry-run review, install the hooks, create the config, then run
`tools/guard/enable-supervisor.sh` and start the installed `supervisor.sh` with
`setsid nohup`. The supervisor uses a lock and cron restarts it at reboot and
every five minutes. To stop, set `enabled` to false, remove cron lines marked
`# tritonchip-guard`, and stop only the verified watchdog PID; no process-name
wildcards are used. The PID is in `/home/ubuntu/.guard/watchdog.pid`; check its
command line before stopping it. Logs and persistent state are in that directory.

`evidence_inventory.py --json` lists tracked evidence, R41 specs and large
files, including sparse worktrees, and reports textual references by category.
Ambiguous basenames and directory references are marked separately. Dynamic
references require review; a missing textual reference is not proof of safety.
