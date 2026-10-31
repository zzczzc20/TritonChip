# R41 K8/G2: multi-precision matrix operand interface

Status: interface contract for the run-2 integration, 2026-10-29. G2 owns `tnca_mac_lane.v` and matrix opcode decode; K8 owns VRF response capture, A/B group selection, `MTX_ABUF`, and the independent matrix-latency parameter. Coordinate changes through git and preserve the default legacy path.

## Command and formats

- `cmd_dt[1:0]` remains the A fragment format from `mod[3:2]`.
- Add `cmd_bdt[1:0]`, driven from `mod[6:5]` for `m.mma.dp.async`; legacy `m.mma.async` uses `cmd_bdt=cmd_dt`.
- Add `cmd_packed`: zero for legacy `m.mma.async` and accumulator row operations, one for `m.mma.dp.async`.
- Packed lane `fa` and `fb` receive the latched A and B two-bit codes directly: 0 FP16, 1 BF16, 2 E5M2, 3 E4M3.
- Packed lane mode is DP4 (2) only for E4M3×E4M3 and DP2 (1) for every other legal pair. Legacy and row operations use FP32 mode (0); their three-bit widening format remains the existing `mac_fmt`.

## Operand capture

The VRF response and optional `MTX_VRF_REG` remain one full 512-bit response. `MTX_ABUF` continues to store complete 512-bit responses; neither structure is replicated for packed mode. Only the post-response group selection and B latch change.

The B latch is `VLEN*32` bits. For lane `n`, bits `[32*n +: 32]` contain the group for consecutive k, with the lowest-k element in the least-significant slot:

- Legacy: one right-aligned A/B element, exactly as before; B slot bits `[31:16]` are zero.
- DP2 with 16-bit elements: `{element(k+1)[15:0], element(k)[15:0]}`.
- DP2 with 8-bit elements: `{8'd0, element(k+1)[7:0], 8'd0, element(k)[7:0]}`. Each byte occupies the low byte of its own 16-bit lane slot, as required by `tnca_mac_lane`.
- DP4 E4M3: `{element(k+3), element(k+2), element(k+1), element(k)}`, four contiguous bytes.

The A group uses the same slot layout as B, selected from `MTX_ABUF` or the current VRF response. One 512-bit B response supplies the whole group for every supported packed layout. The sequencer advances k by 1 for legacy, 2 for DP2, and 4 for DP4. Source-free fires after the final A/B group has been consumed; completion still comes from the result pipeline.

## Latency

Add core parameter `MTX_FMA_LAT=FP_LAT` and pass it only to `u_mtx.FMA_LAT`; do not change vector-FPU latency. The matrix result pipeline remains `FMA_LAT+OPND_REG`, and its drain counter, tag pipeline, scoreboard release, and barrier events use the actual resulting `done` pulse.

For the shared lane with `OPND_REG=IN_REG=1`:

| `CUTS` | Total lane latency | `MTX_FMA_LAT` |
|---|---:|---:|
| `4'b1100` | 3 | 2 (legacy-compatible) |
| `4'b1110` | 4 | 3 |
| `4'b1111` | 5 | 4 |

The default keeps `MTX_FMA_LAT=FP_LAT=2`, so legacy numerics and cycles are unchanged. The accumulator scoreboard is completion-driven and needs no separate fixed-latency shift register.
