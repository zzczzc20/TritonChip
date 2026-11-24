#!/usr/bin/env python3
"""board_batch.py for the xl tier: the registered batch run with two per-configuration rules.

    xl_board_batch.py --manifest M --batch xl --root SRC --out DIR --python PY [board_batch.py arguments]

It takes board_batch.py of the measured tree (SRC/tools/performance_harness, unchanged) and runs its `main()` with a
subclass of `board_batch.Batch` that adds exactly two things, both registered in the manifest before any run:

1. **Per-point process timeout.** Every configuration of the batch carries `timeout_seconds`
   (`tools/r42/board_ab.py register --suite xl`: max(1200 s, 4 x the point's `estimate.seconds_pessimistic` in
   archbench/suites/xl.json, the larger of its two planning estimates)). board_batch.py applies one timeout, the frozen plan's `board_point_timeout_seconds`
   (28800 s), to every process; here that value is replaced for the duration of one run by the configuration's own,
   so a hung host or board process of a short point cannot hold the shared card for hours. A process that exceeds it
   is killed and recorded as a failed run (`board point budget Ns exhausted`), its raw output kept, as for any
   failure in board_batch.py. The timeout covers the whole ArchBench process (compile, CPU reference, upload,
   launches, check).
2. **Sampling cap.** The frozen device rule runs a point at <= 1 s device time a second time, in a new process, for
   3 timed samples. When the qualification process of the point took more than CAP_SECONDS of wall time, that second
   process is not started: the qualification record is the point's only formal record (1 timed sample), exactly what
   board_batch.py already does for a point above 1 s device time. The decision is read from the recorded wall time of
   the qualification run and written to state.json (`xl_sampling_cap`), so it is reproducible and a resume takes the
   same decision.

Nothing else differs from board_batch.py: the same identity, numerical and structural checks decide whether a run is
ok, `formal.jsonl` and `batch.json` are written by its own code, and a failed run (compile failure, capacity,
timeout, out-of-tolerance result) is preserved and reported, never retried here.
"""
import argparse
from pathlib import Path
import sys

CAP_SECONDS = 300


def batch_class(board_batch):
    class XLBatch(board_batch.Batch):
        def bind(self):
            super().bind()
            missing = [c['id'] for c in self.reg['configs'].values()
                       if type(c.get('timeout_seconds')) not in (int, float) or not c['timeout_seconds'] > 0]
            if missing:
                raise board_batch.Refused(f'configurations without a registered timeout_seconds: {missing[:3]}')

        def run_one(self, cfg, tag, warmup, timed):
            if tag.startswith('f'):
                q = self.state['runs'].get(f"{cfg['id'].replace('/', '__')}-q")
                if q and q['ok'] and q['wall_seconds'] > CAP_SECONDS:
                    self.state.setdefault('xl_sampling_cap', {})[cfg['id']] = dict(
                        rule=f'qualification process wall time above {CAP_SECONDS} s: the qualification record is '
                             'the only formal record (1 timed sample); the repeat process was not started',
                        qualification_wall_seconds=q['wall_seconds'], cap_seconds=CAP_SECONDS, skipped_tag=tag)
                    self.save()
                    return q
            budgets = self.reg['plan']['budgets']
            saved = budgets['board_point_timeout_seconds']
            budgets['board_point_timeout_seconds'] = cfg['timeout_seconds']
            try:
                return super().run_one(cfg, tag, warmup, timed)
            finally:
                budgets['board_point_timeout_seconds'] = saved

    return XLBatch


def main():
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument('--root', type=Path)
    known, _ = pre.parse_known_args()
    if known.root is None:
        raise SystemExit('--root is required')
    sys.path.insert(0, str(known.root.resolve()/'tools/performance_harness'))
    import board_batch
    board_batch.Batch = batch_class(board_batch)
    return board_batch.main()


if __name__ == '__main__':
    raise SystemExit(main())
