# Delivery identity amendment: 80dd → bc5a

`doc/r40_v3/identity_amendment.json` moves the v3 delivery product tree from
`files:80dd1a82…` to `files:bc5a778e…`. The only product file that changes is
`python/tnca/inductor.py`: kernels that generate a scan or sort keep the
standard 16 lanes when `TNCA_MAX_BLOCK` exceeds 16. `tnca_cc.register_blocks`
has no wide lowering of `tt.scan`, so before this change
`op.cumsum/medium/fp32-r128-c256` refused `TNCA_MAX_BLOCK=32/64` and
`R40-03.wide-blocks` could not cover its frozen subset.

Offline compile-only check on GB10 (no device opened), the same 112
(point, configuration) compilations as the v3 compile preflight, before and
after the change (`tools/r40/v3_compile_preflight.py`,
`tools/r40/v3_identity_diff.py`):

- `compile_identity_diff_80dd_bc5a.json`: every TNCA artifact (`.hex`),
  assembly (`.s`) and TNCA IR file is byte-identical, except
  `op.cumsum/medium` at `TNCA_MAX_BLOCK=64`, which now compiles. The `.ttir`
  files differ only in debug locations (run directory and source root
  paths); the Inductor kernel-cache hashes are equal.
- `compile_preflight_bc5a_report.json.gz`: wide blocks at 64 compile 15/15
  (was 14/15); every other item is unchanged.

This is compiler evidence only. Per the amendment, all board evidence for the
v3 gates is re-collected on the new tree.
