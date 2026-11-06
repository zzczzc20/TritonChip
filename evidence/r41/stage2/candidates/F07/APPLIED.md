# F07 applied on r41s2: C1 (658c811) slice MSHR service snapshot

- **Source:** `rtl/nuca/tnca_nuca_slice.v` = `git show 658c811:rtl/nuca/tnca_nuca_slice.v`, sha256 `4994243e…`.
- **Base:** C1 was derived from the slice at 54249f7, which is byte-identical to r41s1's slice (sha256 `a68ada2f…`; see
  `evidence/r41/stage1/f07_ooc/`).
- **When it applies:** C1 acts only with `CACHE_DATA_PIPELINE=1`. The B40 image sets `NUCA_CACHE_DATA_PIPELINE 1`.
- **Functional evidence, carried from R40** (files in this directory, copied from 658c811): cycle miter against
  54249f7 plus an independent numerical oracle on the 32-set, 8-way, 8-MSHR, 16-target slice; detected mutation.
- **Unit timing (Stage 1 screen):** worst data-RAM address path drops from 19 to 16 levels (unrouted), at +574 LUT /
  +305 FF per slice.
- **Still needed in Stage 2:** the integrated regression, the kernel physical screen and the routed census.
