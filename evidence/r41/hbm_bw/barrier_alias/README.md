# Barrier index isolation

`doc/engineering_discipline.md` is binding. Domain: RTL simulation on GB10, not board. Source baseline `5e09a2dd5579134ef2299b9e646c659906ec611d` plus the directed test in this commit. No board image or measured clock applies.

The ISA specifies the barrier ID as a 16-bit token field (§9A.2) and identifies illegal fields as `ILLEGAL_INSTRUCTION` (§10). RSLICE assigns each context its own barrier slice (`doc/r39_tasks_and_acceptance.md`, R39-05). The ISA does not separately state a trap code for an index outside that slice. Existing RTL ORs the context slice into an unchecked immediate and truncates it to `BAR_W`, so the immediate can alias another context's barrier.

`python3 tools/r41/barrier_alias_tests.py` builds a four-context core twice, with 32 and 8 barriers, and checks the last valid and first invalid index in a four-way slice for `mb.init`, `mb.arm`, `mb.arrive`, and `mb.inval`. Before the fix, all eight valid cases passed and all eight out-of-range cases failed the expected illegal-instruction trap. Raw output including the failures: `before.tgz`, SHA-256 `379c414f216685092337d59d6114e2c11455cbf33966e7bd5cbb3d7aa342398a`.

The RTL fix is work in progress: it has passed elaboration only. Directed regression, token isolation and integration gates are pending; do not treat this revision as validated for an image.
