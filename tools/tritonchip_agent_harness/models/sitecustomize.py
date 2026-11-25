"""The performance models' hook, mounted read-only at runner.MODELS_DIR in a run whose task asks for the models.

A command that runs a model puts this directory first on PYTHONPATH (`PYTHONPATH=$TNCA_MODEL_HOOK`) and sets
TNCA_BACKEND=systemc-tlm-grid (or systemc-tlm): every Python process it starts then registers the backend of
the model in TNCA_MODEL_DIR -- by default the agent's own tools/performance_harness/systemc, edits and all --
as the models' calibration runs did (2026-11-08). With any other backend it does nothing. It is never on the
container's PYTHONPATH by itself, so the interpreter's own sitecustomize stays in force everywhere else.
"""
import os
import sys

if os.environ.get("TNCA_BACKEND", "").startswith("systemc-tlm"):
    _here = os.environ.get("TNCA_MODEL_DIR", "/work/tools/performance_harness/systemc")
    if _here not in sys.path:
        sys.path.append(_here)
    try:
        import tlm_backend
        tlm_backend.register()
    except Exception as exc:  # noqa: BLE001 -- the command fails on its own backend lookup, with this said first
        sys.stderr.write(f"tnca models: the model backend in {_here} did not register: "
                         f"{type(exc).__name__}: {exc}\n")
