#!/usr/bin/env python3
"""Per-launch profile of the chat driver's decode steps (observation only).

Runs `chat.py` unchanged (same arguments, same compiled graphs, same default TNCA settings) and records, for every
kernel launch made inside a metered prefill or decode step:
  - the kernel name, grid, artifact path and the device cycles the runtime reports;
  - every tensor argument (dtype, shape, view bytes) and what it is: a model parameter or buffer (by name), a
    static-cache tensor (layer and kind), or an activation (anything else);
  - whether the launch had to place its code in the code store (a load or reload) and how many modules it evicted;
  - on the SystemC/TLM backend, the difference of every model counter across the launch. The counters are read
    with the model's read-only `C` command while the device is idle, between launches, so they add host time and
    no device cycles.

Nothing here changes what the device executes: the hooks wrap the runtime's launcher, the session's grid dispatch
and chat.py's step meter, and only read. The output is OUT/profile/launches.json (one row per launch) beside
chat.py's own OUT/report.json. tools/models/decode_profile_report.py turns it into the per-class tables.

  python tools/models/decode_profile.py --model vl --device tnca --layers 1 --prompt "Say hi." \
      --new-tokens 4 --bucket 64 --out OUT
"""
import contextlib
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import chat  # noqa: E402

ROWS = []
STATE = dict(step=-1, session=None, ptrs={})


def _tensor_map(session):
    """storage data_ptr -> (role, name) for the model's parameters and buffers and the static cache."""
    import torch
    out = {}
    if session is None:
        return out
    for name, t in list(session.model.named_parameters()) + list(session.model.named_buffers()):
        if torch.is_tensor(t) and t.device.type == "tnca":
            out.setdefault(t.untyped_storage().data_ptr(), ("weight", name))
    for i, layer in enumerate(getattr(session.cache, "layers", [])):
        for kind in ("keys", "values"):
            t = getattr(layer, kind, None)
            if torch.is_tensor(t) and t.device.type == "tnca":
                out[t.untyped_storage().data_ptr()] = ("kv", f"layer{i}.{kind}")
        for kind in ("conv_states", "recurrent_states"):
            states = getattr(layer, kind, None)
            items = states.items() if isinstance(states, dict) else ([(0, states)] if torch.is_tensor(states) else [])
            for k, t in items:
                if torch.is_tensor(t) and t.device.type == "tnca":
                    out[t.untyped_storage().data_ptr()] = ("state", f"layer{i}.{kind}.{k}")
    return out


def _args(values):
    import torch
    rows = []
    for v in values:
        if isinstance(v, torch.Tensor):
            ptr = v.untyped_storage().data_ptr()
            role, name = STATE["ptrs"].get(ptr, ("act", None))
            rows.append(dict(role=role, name=name, dtype=str(v.dtype).replace("torch.", ""), shape=list(v.shape),
                             stride=list(v.stride()), view_bytes=v.numel() * v.element_size(),
                             storage_bytes=v.untyped_storage().nbytes(), storage=ptr))
        elif isinstance(v, (int, float, bool)):
            rows.append(dict(role="scalar", value=v))
    return rows


def _delta(before, after):
    return {k: after[k] - before.get(k, 0) for k in after
            if isinstance(after[k], (int, float)) and after[k] != before.get(k, 0)}


def install():
    from tnca import runtime
    from tnca.triton_backend import driver

    # The session runs tasks on its worker thread and the launcher returns once the task's event is terminal, so
    # the counters are read on that thread: before the task runs, and in _finish before the event turns terminal.
    # The difference rides on the event, which the launcher keeps as last_event.
    original_run = runtime.Session._run_one
    original_finish = runtime.Session._finish

    def run_one(self, task):
        snap = getattr(self.dev, "counters_snapshot", None)
        if snap is not None and STATE["step"] >= 0:
            task["_prof_before"] = snap()
        return original_run(self, task)

    def finish(self, task, state, error=None):
        before = task.pop("_prof_before", None)
        if before is not None:
            task["event"]._prof_counters = _delta(before, self.dev.counters_snapshot())
        return original_finish(self, task, state, error)

    runtime.Session._run_one = run_one
    runtime.Session._finish = finish

    original_call = driver.TNCALauncher.__call__

    def call(self, gridX, gridY, gridZ, stream, function, packed_metadata, launch_metadata, enter, leave, *args):
        if STATE["step"] < 0:
            return original_call(self, gridX, gridY, gridZ, stream, function, packed_metadata, launch_metadata,
                                 enter, leave, *args)
        resident = any(key[2] == function for key in driver._loaded)
        evicted = driver.evicted
        original_call(self, gridX, gridY, gridZ, stream, function, packed_metadata, launch_metadata, enter, leave,
                      *args)
        counters = getattr(self.last_event, "_prof_counters", None)
        ROWS.append(dict(step=STATE["step"], index=len(ROWS), kernel=self.metadata.name,
                         grid=[max(gridX, 1), max(gridY, 1), max(gridZ, 1)], artifact=function,
                         device_cycles=self.last_cycles, code_load=not resident,
                         code_bytes=next((m.code_size for k, m in driver._loaded.items() if k[2] == function), None),
                         evicted=driver.evicted - evicted, args=_args(args),
                         counters=counters))

    driver.TNCALauncher.__call__ = call

    original_session = chat.Session.__init__

    def session_init(self, *a, **k):
        original_session(self, *a, **k)
        if self.device == "tnca":
            STATE["session"] = self

    chat.Session.__init__ = session_init

    original_meter = chat.transfer_meter

    @contextlib.contextmanager
    def meter(enabled):
        if enabled:
            STATE["step"] += 1
            STATE["ptrs"] = _tensor_map(STATE["session"])
        try:
            with original_meter(enabled) as m:
                yield m
        finally:
            if enabled:
                STATE["ptrs"] = _tensor_map(STATE["session"])   # linear-attention states appear on first use

    chat.transfer_meter = meter


def main(argv=None):
    args = chat.parser().parse_args(argv)
    if args.device != "tnca" or args.interactive or args.script or args.session_check:
        raise SystemExit("decode_profile profiles one-shot tnca generation only")
    chat.L.use_tnca(args.tnca_tree)
    install()
    code = 1
    try:
        code = chat.main(argv)
    finally:
        if args.out:
            out = args.out / "profile"
            out.mkdir(parents=True, exist_ok=True)
            (out / "launches.json").write_text(json.dumps(dict(
                schema="tnca-decode-profile/1", argv=list(argv or sys.argv[1:]), exit=code,
                step_kinds="step 0 is the prefill call; steps 1.. are decode calls in order",
                launches=ROWS), indent=1) + "\n")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
