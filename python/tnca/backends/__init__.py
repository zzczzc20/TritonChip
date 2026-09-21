"""Execution backend implementations.

One per way of reaching a device. The runtime does not import any of
these directly: it asks `tnca.backend.open_backend` for a name, which is
the boundary an external branch extends at without touching the runtime
(doc/phase3_6_tasks_and_acceptance.md section 4.2).
"""
