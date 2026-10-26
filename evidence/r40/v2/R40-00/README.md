# R40 v2 R40-00 re-audit

The frozen v2 contract is unchanged. Protocol refusal tests were rerun locally; B39 reproduction was re-audited against the original raw 77-point rerun and protocol-2 records, plus the original 27/27 merge-gate run. This is an audit of historical board measurements, not a claim of a new board run or a rerun of every merge gate on the current software. Freeze regeneration/API checks were rerun with the pinned Torch environment on the jump host from an archive of 66a5f33; 33/33 pass.

The compressed raw JSONL files preserve exact bytes. inputs.json identifies original paths and raw SHA256 values. Historical provenance bundle references in those records retain their original paths. The v1 archive remains under evidence/r40/R40-00; no historical measurements were relabelled as final-image performance. All three v2 R40-00 gates carry the immutable v2 contract hash. Other work packages remain pending.
