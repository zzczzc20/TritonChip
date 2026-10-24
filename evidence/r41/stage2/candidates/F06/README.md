# F06: registered grid admission (dispatch DSP `total` → `remaining`)

Status: **implemented on `r41s2-io`; functional regression passes; unit OOC screen
passes the 9.0 ns projected target.** It is not a routed result. The kernel synthesis,
the routed census and board runs are still to be done.

- **Responsible source:** `rtl/tnca_grid_dispatch.v`. Base: `cc240819…`, the same hash
  as the preflight's F06 entry. New: `d7cd43b7…`.
- **Bench:** `tb/tb_grid_dispatch.v`.
- **OOC harness:** `flow/fpga/tnca_grid_dispatch_ooc.v`.

## The census family

The A3 routed census (`evidence/r40/v3/final_image_candidate/census/`) has 36 residual
endpoints in `g_dispatch.u_dispatch`:
- TNS −23.77 ns; worst −0.907 ns;
- all start at `total0__1`, the DSP for `launch_gx*launch_gy`, which absorbed the launch
  register;
- they end at `remaining_reg[*]/R` (32), `done_status_reg[1:0]`, `done_valid_reg` and
  `running_reg`.

The worst path is `remaining_reg[11]/R`:
- data path 10.657 ns: 5.636 ns logic, 5.021 ns route;
- 31 levels: two cascaded DSP multipliers, 8 CARRY8 and the `total > MAX_PROGRAMS`
  compare.

The 96-bit `gx*gy*gz` product and the refusal decision were evaluated in the accepting
cycle. That decision drove the synchronous reset of `remaining`, plus `done_valid`,
`done_status` and `running`.

## Design

The accepting edge only captures the launch (as before). It also captures
`launch_group_log2 === 3`, clears `remaining` and arms a one-hot admission shift
register `adm`. Four registered stages, fed only by registers that the accepting edge
wrote, then make the old decision:

| Stage | Registered result |
|---|---|
| A | Each factor saturated to `SW = clog2(MAX_PROGRAMS+2)` = 25 bits (a factor above `MAX_PROGRAMS` becomes `MAX_PROGRAMS+1`); x rounded up to chunks; the zero test; the alignment, argument-range and core-mask checks. |
| B | `x*y` and `chunks*y` (25×25 products). |
| C | `(x*y)[24:0]*z` and `(chunks*y)[24:0]*z`, plus `x*y > MAX_PROGRAMS`. |
| D | Decide: refuse (status 3), complete an empty grid (status 0), or start running with `remaining` loaded from the registered chunk count. |

`launch_ready` is low for the four admission cycles. Because `tnca_top` derives the host
permission and `grid_ready` from it, a second launch, a host access and the MMIO all see
the dispatcher as busy.

## Correctness argument

- **Same decision.** Assume every factor is nonzero (zero is tested exactly, and only
  then).
  - If all three factors are at most `MAX_PROGRAMS`, saturation leaves them unchanged.
    Both partial products are then exact whenever they are at most `MAX_PROGRAMS`, and
    `x*y > MAX` is flagged.
  - If any factor exceeds `MAX_PROGRAMS`, its saturated value is `MAX+1`. With the
    others ≥ 1, the saturated product exceeds `MAX`, as the true product does.
  - So `refuse = bad || (!zero && (xy > MAX || xy*z > MAX))` equals the old
    `total > MAX || …` for every 32-bit input, including products that wrap 32 or
    64 bits. The old else-if order is kept: refusal, then the empty grid, then run.
  - The chunk count is at most the program count. It is therefore exact whenever the
    grid is admitted, and `remaining` gets the old value.
- **No early issue.** `enqueue` and the completion test require `running`, which is
  set only by stage D. During admission `core_busy`, `pending` and `accepted` are zero.
- **Aborts and faults stay visible.**
  - `fault_now` now includes `abort_valid` during admission. A one-cycle deadline pulse
    (`tnca_grid_deadline` asserts `abort_valid` for exactly one cycle) or a sticky
    memory-protocol abort is therefore recorded in `faulted`/`done_status`, exactly as in
    the first running cycle before.
  - A stray completion from an idle core is recorded the same way.
  - An admitted grid with a recorded fault completes the next cycle with nothing issued,
    as the old first-cycle abort did.
  - A refused or empty grid reports the decision as of the accepting edge: status 3 or
    0, `fault_core` 0 and `done_cycles` 0. The old design gave the same result, because
    no abort could be observed before that edge.
- **Deadline.** The deadline counter runs from acceptance, so a grid now has four fewer
  running cycles before the same deadline. Deadlines are host-set stop-and-drain limits
  (`tb_multicore_axi` uses 20); nothing relies on cycle-exact deadline arithmetic.
- **Bounded admission.** Admission takes a fixed four cycles whatever the grid size, and
  needs no storage proportional to the grid. Grids up to 2^24 programs are admitted,
  including those above 65,535 programs.

## Verification on GB10

Tree: `r41s2` + F06 only, as `~/tritonchip-r41s2-io`. Logs are in
`f06-regression-logs.tgz`.

**New directed tests in `tb_grid_dispatch.v`** (N = 1, 2, 3, 4, 8 through
`make r40-grid-dispatch`):
- **Admission window.** Every launch holds `launch_valid` through admission and checks:
  - no second acceptance;
  - `launch_ready`, `cmd_valid`, `core_busy` and `done_valid` stay low;
  - one-hot `adm`;
  - the decision lands exactly 4 cycles after acceptance.
- **Dimension boundaries.**
  - Refused: 4097×4096, 4096×4097, 1×4096×4097, 65537×256, 2^24×2, 1×1×(2^24+1),
    65536×65536 (wraps 32 bits), 2×2^31, 65536^3 (wraps 64 bits), 2^32−1 in each
    dimension, and group_log2 = 3.
  - Empty: 0×(2^32−1)×(2^32−1) in all three positions, 0×0×0, and an empty grid with no
    cores. These complete with status 0, not 3.
- **Largest admissible grids** (admitted, `remaining` checked, then aborted after the
  first commands drain): 2^24×1×1, 4096×4096, 256^3, 1×1×2^24, 65536×256 and 70001×239.
  With 4-lane chunks: 2^24 (4,194,304 chunks), 4095×4096 and 4097×1×4095 (4,197,375
  chunks).
- **Grids above 65,535 programs run to completion**: 65,537×1×1, 257×256 (65,792) and
  65,541 programs in 4-lane chunks. Every PID is issued once.
- **Admission-window events.** At each of the four stages:
  - a one-cycle abort pulse: status 2, nothing issued;
  - the same pulse on a refused grid: status 3;
  - the same pulse on an empty grid: status 0;
  - a stray completion: status 1, `fault_core` 0, nothing issued.
- **Mutations**, each caught:
  - aborts ignored during admission;
  - no saturation;
  - no zero guard on the refusal;
  - `launch_ready` not gated by admission.

**Regression targets** (all on the F06 tree):

| Target | Result |
|---|---|
| `make r40-grid-dispatch`: the new bench, `tb_grid_cores` on real cores (N = 1–8, with and without DRAM, coherent host, global cache handoff, injected source fault) and `tb_mem_arbiter_r40` | PASS (20 configurations) |
| `make r40-multicore-shell` (with `r40-grid-deadline`) | PASS |
| `make r40-coherent-host` | PASS |
| `make r40-line-axi`: `multicore_axi_tests` with the deadline-20 timeout and the physical-fault abort, N = 1–8, write delay 300 | PASS |
| `make r40-cache-maintenance` | PASS (19/19) |
| `make r40-multicore-dma` | PASS |
| `multicore_mmio_tests.py --cores 3 --ports 3 --pipeline 2 --board-top` | PASS |
| `make r40-multicore-mmio-checks` | PASS (run with the Torch interpreter, which `bucket_decode_tests` needs) |
| `large_grid_rtl_tests.py`: PIDs above 2^16 and 2^31, 6144-program grids | PASS (10/10) |
| `make r40-physical-grid-runtime` | PASS |
| `make r40-physical-grid-transport` | `physical_grid_option_tests` PASS (21); `multicore_transport_tests` PASS. `large_grid_tests` raises `AttributeError: 'Session' object has no attribute '_measurement_owner'` identically on the unchanged base tree. That is a Python runtime test, not RTL. |

## Cycle impact

Each launch costs exactly 4 more cycles from acceptance to the first command. Nothing
changes after admission.

- **`tb_grid_cores`, no DRAM** (455 programs, completion cycle):

  | N | Base | F06 |
  |---:|---:|---:|
  | 1 | 3646 | 3650 |
  | 2 | 1830 | 1834 |
  | 3 | 1223 | 1227 |
  | 4 | 920 | 924 |
  | 8 | 468 | 472 |

  With the DRAM model, the completion cycles of both phases are identical for N = 1–8;
  the DRAM timing absorbs the shift.
- **`tb_multicore_axi`**, cumulative completion cycle of the 455-program grid on the AXI
  outlets:
  - 0 to +51 cycles across the N = 1, 2, 3, 4, 8 configurations: unchanged for most
    N ≤ 3 runs, +11 in one N = 3 run, +22 to +51 for N = 4 and 8, from outlet-timing
    interactions (≤ 0.4 %);
  - the deadline-limited grid: +3 to +5 cycles;
  - shell-refused grids: unchanged.

## Unit OOC timing at 10 ns

- **Run:** `vivado_ooc_timing.tcl`, `OOC_SV=1`, part `xcu280-fsvh2892-2L-e`, generic
  `CORE_N=3` (the image's three cores, default `MAX_PROGRAMS`).
- **Harness:** `tnca_grid_dispatch_ooc`, which registers every input and output, as
  the MMIO registers and command links do. Without it the census path would start at an
  unconstrained port.
- **Where:** FPGA host, 2026-10-21, through `vivado_slot.sh`.
- **Raw files:** reports, family queries and sources are in `f06-ooc-evidence.tgz`.

| Unrouted estimate, 10 ns | Base (`cc240819`) | F06 (`d7cd43b7`) |
|---|---:|---:|
| WNS | +1.922 ns | +7.245 ns |
| Census path (launch DSP → `remaining`/`done`/`running`) | `total0__1` DSP → `remaining_reg[0]/R`, 31 levels, **7.929 ns** | removed. Into those registers the worst is `core_busy` → `done_valid`, 6 levels, 1.779 ns; product → `remaining` is `p_xyz` DSP → `remaining_reg[31]`, 8 levels, 1.683 ns |
| Worst dispatcher path | the census path | `s_g_reg` → `p_gyz` DSP input, 8 levels, **2.487 ns** |
| CLB LUTs / registers / DSPs (harness included) | 1219 / 2659 / 18 | 1217 / 2822 / 7 |

**Projection.**
- Census routed / baseline synthesis for the census path: 10.657 / 7.929 = 1.344.
- The new worst dispatcher path projects to 2.487 × 1.344 = **3.34 ns**, and product →
  `remaining` to 1.683 × 1.344 = 2.26 ns. Both are well inside the 9.0 ns target.
- Resources: +163 registers, −11 DSPs, LUTs unchanged, in one SLR (the dispatcher's).

## Open risks

- The ratio is a per-path projection from one routed census. Placement in the real
  kernel can differ.
- The dispatcher's launch inputs still fan out from the MMIO registers into the
  capture registers. That path is unchanged and was not in the census.
- The four admission cycles make `tnca_multicore_shell`'s deadline count four cycles of
  admission. A host deadline below about 5 cycles could now expire during admission:
  - on a refused grid, `timed_out` would read 1 while the status is 3;
  - previously, such a grid completed on the next cycle.

  No caller uses deadlines that small.
