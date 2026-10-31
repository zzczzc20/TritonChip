# Preserved evidence versions

The Stage 2 integrity check on 2026-10-30 found three mismatches: both reports
had acquired a later run-2 update, and the current SystemC tensor-engine
source had changed after the P1 study. The original hashed contents here were
retrieved from git, not reconstructed:

| File | Source commit | SHA-256 |
| --- | --- | --- |
| `r41_report.md` | `d9bba800060c367fc729685d955c60fb435baddb` | `d069ad6833e133b05e201bc15de3069225e65632c6dc50809d60380bb463da00` |
| `r41_stage2_report.md` | `d9bba800060c367fc729685d955c60fb435baddb` | `9715e8ca4c05701b45ee8109059a25fbc9454220c946aed2ac2c650ef4a180c9` |
| `tnca_tensor_engine.hpp` | `792d9b7ee16e93eb210b0e58879efa50a9b7fbff` | `46efa44368b4b3e3654c2646a8d418f9f02e3bfb87b106707c9b1e32ff8b0ee7` |

S2-02.B now cites the actual historical source version; this does not assert
that the later SystemC version was measured in that study. S2-06.10 retains
the historical reports and cites the updated reports, which document the
corrected candidate and its remaining gates. The integrity check now passes;
the 38 passed, two failed, 47 not measured and seven blocked judgments are
unchanged, so Stage 2 is still not accepted. The eight acceptance-checker unit
tests and `python3 -m tools.docmaint check` passed after this reconciliation.
