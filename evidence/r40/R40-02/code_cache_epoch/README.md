# Code pressure keeps the session and epoch

The CPU migration driver previously replaced a Session when its code region
filled. Two common-clock CPU llama_prefill controls failed because a graph's
launch timestamps then belonged to different sessions. Native resident code
already used module eviction rather than discarding its tensor storage.

The common driver now drains outstanding launches, unloads least-recently-used
modules and reloads into the same Session for both paths. Cache keys use the
monotonic Session ID, avoiding reuse of Python object IDs. A single oversized
kernel remains an explicit code-capacity refusal.

`rtl.log` executes six alternating kernels through the actual RTL with a region
that holds exactly one module. Numerical writes, a live data handle, session ID
and epoch stay correct through eviction. This is a correctness diagnostic, not
Torch performance evidence. The two failed board CPU controls are being rerun
with the patched driver; their result is pending.
