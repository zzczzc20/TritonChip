# Building the U50 kernel

`flow/fpga/ooc.xdc` ends with a pointer to this file for "the measured one". This
is that number, where it came from, and what it does not cover.

**The `.xclbin` this directory builds has been on the board.** It was loaded on
2026-09-26 at 19:54 by `flow/fpga/program.sh`, which is the third step here and
the only one that changes hardware state; the sections below record what that
measured. Loading is still a deliberate step, but it is no longer one that has
not been taken.

## The two steps

```sh
vivado -mode batch -source flow/fpga/package_xo.tcl -tclargs <repo> <build-dir>
flow/fpga/link.sh <build-dir>
```

The first builds a Vivado IP out of the same RTL the repository lints and
simulates, and wraps it into `<build-dir>/tnca_u50.xo`. The second hands that
`.xo` to `v++ -l -t hw` and produces `<build-dir>/tnca_u50.xclbin`.

They are two steps rather than one because `package_xo` takes an
`-ip_directory`, and an IP is a thing that has to exist before it can be
handed over. `package_xo.tcl`'s header argues the shape.

Neither step needs root. Neither step touches the board. The output of both is
a file.

## The clock

The platform states its kernel clocks in its own `hw.hpfm` inside `hw.xsa`:

```xml
<xd:systemClocks xd:defaultClock="0">
  <xd:clock xd:id="0" name="ulp_ucs_aclk_kernel_00"
            xd:frequency="300.000000" xd:period="3.333333"
            xd:status="scalable"/>
  <xd:clock xd:id="1" name="ulp_ucs_aclk_kernel_01"
            xd:frequency="500.000000" xd:period="2.000000"
            xd:status="scalable"/>
  <xd:clock xd:id="2" name="_bd_top_blp_s_aclk_freerun_ref_00"
            xd:frequency="100.000000" xd:status="fixed"/>
</xd:systemClocks>
```

Those two frequencies are the clocks' **power-on defaults**, not the set of
values the shell can produce. That set is declared by the clocking IP the shell
is built from -- `shell_ucs_subsystem_v3_0`, unpacked from the platform's
`hw.xsa`, whose `component.xml` gives `FREQ_HZ_KERNEL_CLOCK_00` a
`minimum="10" maximum="650"` in MHz. 150 is inside that range.

**This build asks for 150 MHz.** `link.sh` passes `--kernel_frequency 0:150`,
`ooc.xdc` constrains 6.667 ns, `package_xo.tcl` declares the same figure in the
IP, and `rtl/board/tnca_u50_xo.v` divides by `DEV_DIV`. The divide was four
-- the device at **37.5 MHz**, where it ran before, when the bus was at 300 and
the divide was eight; the bus came down because the first link failed on it,
and the device was held still so that every measurement about it survived.
Since R38's timing work it is **two: the device at 75 MHz**, measured on the
U280 (below, and `evidence/r38/u280_75mhz/`).

`link.sh` still checks the resulting frequency rather than assuming it -- `v++`
can lower a kernel clock on its own and still report success
(`AUTO-FREQ-SCALING-04`, quoted in `link.sh`). That check is not in tension with
naming 150 here: it exists so that 150 is what actually reaches the `.xclbin`,
not so that nobody ever writes a number down.

## The measurements

Both are out-of-context synthesis and implementation of **`tnca_fpga_top`**, the
device alone -- no adapter, no clock crossing, no divider. Both were run on the
device RTL as it stands: `rtl/fpga/tnca_fpga_top.v` and everything below it have
not changed since 2026-09-24 03:04, which is before either run.

| constraint | WNS | failing endpoints | worst path | verdict |
|---|---|---|---|---|
| 3.333 ns (300 MHz) | −15.514 ns | 474 875 of 499 331 | **18.829 ns** | not met |
| 40.000 ns (25 MHz) | +14.750 ns | 0 of 498 620 | **25.231 ns** | met |

Source: `~/u50_build/synth/tnca_paths500_routed.rpt` and
`~/u50_build/synth25/tnca_paths500_routed25.rpt`.

**The two worst-path figures disagree, and that is not an error.** A router
works to the constraint it is given: asked for 3.333 ns it produced an 18.829 ns
path, and asked for 40 ns it produced a 25.231 ns one and stopped. So 18.829 ns
is a lower bound on what the design can do when something is pushing on it, and
25.231 ns is what it does when nothing is -- which is not the same as an upper
bound either. What can be said without overreach:

- At 300 MHz the design fails, and by a lot: 95% of endpoints.
- The device logic routed under a 40 ns constraint with 14.75 ns of slack.

The 40 ns run was directed on 2026-09-24 as "我们先用25mhz试一下吧" -- a bring-up
measurement. The 3.333 ns script was left in place untouched so the two can be
compared. **Neither run took part in a link**: both are out-of-context
measurements of `tnca_fpga_top` alone, taken before the adapter existed.

## What the first link measured

`link.sh` has now run once, at the platform default of 300 MHz and with no
`--kernel_frequency` on the command line (`~/u50_build/pkg/link.log`, started
2026-09-26 00:53, `elapsed = 01:56:36`). **It failed**, on `write_bitstream`:

```
ERROR: [VPL 101-2] design did not meet timing - hold violation
```

173 setup endpoints and 4 hold endpoints failed, out of a design whose device
domain had **+6.494 ns of slack and nothing failing in it**. Of the 173, 131 are
not reachable by editing RTL at all -- 97 of them are a zero-logic, 0.722 ns path
that a −3.884 ns skew between two clock objects the platform never relates fails
on its own. The 131 sit in three groups -- the platform's clock identity
(97), the platform's own interior (1), and a routing outcome (33) whose worst
path spends 0.237 ns of its 4.854 in logic -- and the rest miss a single
3.333 ns period by under two nanoseconds. The 173 is de-duplicated: an endpoint
can appear in more than one clock pair, and the per-pair rows sum to 206.

That is the whole reason the bus is at 150 MHz now: doubling the period reaches
every one of those paths at once, and dividing by four instead of eight keeps the
device exactly where the +6.494 ns was measured.

## What the second link measured

At 150 MHz (`--kernel_frequency 0:150`, `~/u50_build/pkg150/link.log`, started
2026-09-26 16:13, `elapsed = 2h 11m 55s`) **the link succeeded** and
`tnca_u50.xclbin` was written. Out of the routed report:

```
    WNS(ns)   TNS Failing Endpoints     WHS(ns)   THS Failing Endpoints
      0.055                       0       0.009                       0
                TNS Total Endpoints: 901013    THS Total: 899418
All user specified timing constraints are met.
```

The same clock table names what the two runs differ by:

```
ap_clk                          6.667 ns   149.992 MHz
  clk_dev                      26.668 ns    37.498 MHz
  clk_kernel_00_unbuffered_net  6.667 ns   150.000 MHz
  clk_kernel_01_unbuffered_net  2.000 ns   500.000 MHz
```

`ap_clk` is at 6.667 ns, and `clk_dev` is at 26.668 -- exactly one quarter of it,
which is `BUFGCE_DIVIDE 4` doing what `rtl/board/tnca_u50_xo.v` says it does. The
device is where it was; the bus moved. 0.055 ns is not a comfortable margin, but
it is a closed design: the bitstream is fixed and so is the clock.

`--kernel_frequency 0:150` also did what it was asked to, which took reading the
log to confirm. v++ numbers the shell's clocks and then renames them in the
xclbin:

```
INFO: [v++ 60-991] clock name 'ulp_ucs/aclk_kernel_00' (clock ID '0') is being
mapped to clock name 'DATA_CLK' in the xclbin
```

so the entry called `KERNEL_CLK` in the finished artifact is
`ulp_ucs/aclk_kernel_01` -- a clock this design is not wired to -- and the one
this design runs on is the entry called `DATA_CLK`. `link.sh` was matching on
`Type: KERNEL`, compared 500 against 150, and reported `LINK_FAILED freq=500`
for a build whose real clock was exactly what was asked for. It now reads the
entry name out of the tool's own mapping line and fails if that line is absent,
which is the same fail-closed direction as before and no longer reads the wrong
entry.

At the time that was written, none of it had been on a board. It has since --
see the section below, which is where the 150 above stops being a request.

## What the board measured

`flow/fpga/program.sh --go` loaded `tnca_u50.xclbin` on 2026-09-26 at 19:54:26.
It reported `EXIT=0` after about a second, and the device then answered with
xclbin uuid `ccbbcf8b-5ace-7c99-63b9-9a7c065d74a5`, which is the uuid of the
file that was written. The image it replaced was `44e9640d-…`.

### The clock is 150 MHz, measured

This is the figure the section below could only report as *requested*. The shell
counts the clock it is actually handed, and the driver logs the count beside the
request whenever `xbutil examine -r platform` reads it:

```
clock_wiz_get_freq_by_id: freq = 100      <- before: the image that was loaded
clock_counter_get_freq:   khz: 100000
clock_wiz_get_freq_by_id: freq = 150      <- after: this build
clock_counter_get_freq:   khz: 150000
```

`frequency_counter_aclk_kernel_00` counts edges, so `150000` is a measurement of
the MMCM output rather than the metadata the solver was handed read back at
itself. The same log shows `clock_wiz_scaling_by_request: verify: 0 ret: 0` and
`icap_refresh_clock_freq: ret: 0` at load time, which is XRT programming the MMCM
and saying it worked. **The device therefore runs at 150/4 = 37.5 MHz**, which is
where the OOC measurements above put it.

### The device answers

Two of three rungs of `tmp_variants/board_bringup.py` -- scratch, not in the
repository -- ran against the loaded design:

| rung | what it does | result |
|---|---|---|
| `read` | three reads over XRT's AXI-Lite register API | **PASS** -- `R_MAGIC` came back `0x00010000544e4341`, "TNCA" |
| `ping` | one mailbox round trip, core not started | **PASS** -- `seq=2 OK r0=2 r1=0` |
| `run` | upload, launch the core, compare results | **PASS** -- 5 of 5 values, 3 of 3 sentinels |

`read` is the weaker of the two and it is still worth naming what it rules out:
the value is read back over PCIe from registers that only exist if the
bitstream is the one this directory built and is out of reset. A half-configured
region does not answer with a magic number.

### `run` failed first, on something that was not the board

The first attempt never reached the core. `start()`'s ping went unanswered and
the failure surfaced as `AttributeError: 'U50Backend' object has no attribute
'log'` -- a second bug, hiding the first.

The real one was in the host, and the board was right to be silent. The device's
rule for a doorbell is `req_seq > served` (`rtl/fpga/tnca_mmio.v`'s `MS_IDLE`),
and `served` is a 64-bit counter that resets when the **bitstream** does, not
when a host process starts. `Transport` numbers requests `self.seq += 1` from
zero, so every new process began at sequence 1. `ping` had been the first
mailbox user and left `served` at 3; the next process rang with sequence 1, the
device dropped it as the duplicate the rule says it is, and a dropped ring is
silent by design. `R_SERVED` read back **`0x0000000000000003`** -- exactly the
three requests `ping` had made.

So the board was usable once per bitstream load, which is not a property anyone
would find in simulation, because a simulation binding is started fresh each
time. The fix is the one the register exists for: `start()` adopts `served`
before its first request. The run then reported `seq=5`, which is what the
diagnosis predicts and not a coincidence.

The masking bug had the same shape as one the file already guards against:
`Transport.__init__` sets `bench` and `dir`, and the board's constructor
deliberately re-sets them to `None` so that no reachable path raises
`AttributeError` -- and `log` was the one that got missed, on the timeout path of
all places. It is `None` now, so a board timeout arrives as `WAIT_TIMEOUT`.

### `run`, second attempt

```
    upload ok (<Reply seq=5 OK r0=0 r1=0>)
    launch ok (status 0, 243 cycles, 72 retired)
    partial stores ok (0x333333332222ee11)
    vector round trip ok (0x00000000000000f0)
    gather ok (0x0000000000000078)
    atomic, old value ok (0x0000000000000007)
    atomic, new value ok (0x000000000000000c)
    a slot the program must not have touched (5) ok (0xdeadbeeffacef00d)
    a slot the program must not have touched (6) ok (0xdeadbeeffacef00d)
    a slot the program must not have touched (7) ok (0xdeadbeeffacef00d)
PASS run
```

**The core executed on the board and the numbers are the hand-computed ones** --
including the three slots the program was supposed to leave alone.

### Why `run` was held back first

Recorded because the hold was correct at the time and the rule is not repealed:
when `ping` returned the board was at **FPGA 91 C / HBM 86 C**, over a standing
threshold that stops hardware testing at 90 C / 85 C. Power was 22.5 W of 75 W
with nothing running -- `total execs 0`, no client attached.

It ran on the instruction that the temperature was not to be considered, after a
90-second check showed the climb had stopped (94/89 flat across three readings).
`run` itself moved nothing: 95 C before, 95 C after. Whether the rise from 82 to
94 across the session is ours (150 MHz where the previous image ran 100), the
host's, or ambient has **not** been established, and the 22.5 W figure argues
against it being the design working hard. Recorded as an open question, not
attributed.

**Nothing was done about the temperature.** Cooling, power limits and clock are
outside what this branch may touch, and so is `xbutil reset`. The design is still
loaded.

## What the measurements do not cover

The 37.5 MHz figure rests on the device having routed at 25.231 ns under a 40 ns
constraint, which is under the 26.667 ns that 150/4 gives it. That is 1.4 ns of
margin, and it is margin on a run that was not asked to try.

Neither OOC run contained `rtl/fpga/tnca_axi_lite.v`, `rtl/fpga/tnca_u50_top.v`,
or `rtl/board/tnca_u50_xo.v`. The adapter and the clock crossing are timed by the
link at `ap_clk` -- 6.667 ns -- and the first link is what showed they do not
close at 3.333 ns.

And nothing *in this file's measurements* says the clock is 150 MHz, only that
150 MHz was asked for. The kernel clock is an MMCM output --
`aclk_kernel_00_hierarchy/clkwiz_aclk_kernel_00` in the link's own `vivado.log` --
and Vitis reaches it by re-solving that MMCM and emitting a
`create_generated_clock` for the divide and multiply it picked
(`write_user_impl_clock_constraint` in `ocl_util.tcl`). So the figure `link.sh`
reads back out of the `.xclbin` is the one the solver was handed. The shell counts
the clock it actually gets, in `frequency_counter_aclk_kernel_00` at `0x9000`, and
reading that means loading the `.xclbin` -- which has since been done, and the
count agrees. See "What the board measured" above.

So: the divider ratio, the device frequency, and whether the whole thing closes
are three separate questions, and only the middle one has been measured
favorably.

## If the link fails on timing again

Not by reaching for a smaller number in one place. `ooc.xdc`, `package_xo.tcl`,
`link.sh` and the divider comment in `rtl/board/tnca_u50_xo.v` all carry the
frequency, and `link.sh` checks the result against them; a link that closed at a
clock only one of those files mentions is a different design from the one anyone
measured. 150 MHz is itself a lowering, chosen and recorded deliberately -- see
the header of `link.sh` for the argument. Going below it is a second such
decision, and it gets made the same way: with the report in hand, and the number
moved in all four places at once.

## Since this image: the r38 merge (2026-10-07)

The image measured above was built from the `fpga-u50` branch. After that
branch was merged into `r38`, the device RTL behind it changed, so the loaded
image is **not** the current design and has to be rebuilt before any r38
measurement is taken on the board:

- `rtl/tnca_shell.v`: the counter read port (`perf_sel`/`perf_rdata`), two
  data-port beat counters, and `done_retired` reporting the command's own
  instructions rather than every instruction since reset.
- `rtl/fpga/tnca_mmio.v`, `rtl/fpga/tnca_fpga_top.v`: mailbox op 7
  `PERF_READ`, answered in 32-bit halves like the simulation bench; an index
  past the last counter is refused. Checked by `R36-FPGA-08`.
- `rtl/core/tnca_tensor.v`: the transposed tensor transfer, elaborated only
  with `HAS_TENSOR_XPOSE=1` (0 here, as in `core-vrf-sync-v1`).

The build steps, the order of the checks on the new image, and what each must
show are in [`doc/r38_board_bringup.md`](../../doc/r38_board_bringup.md).

## 75 MHz on the U280 (2026-10-09)

R38's timing work (`doc/r38_board_bringup.md` sec. 6) cut the device's worst
path from a 60-level loop in the matrix unit to 7.76 ns post-synthesis. On
that RTL the build ran with `DEV_DIV = 2`:

- **The link:** clk_dev routed at **+0.471 ns** of slack against 13.333 ns,
  with nothing failing. The design WNS is +0.016 ns, the platform's own
  near-zero path. v++ printed only AUTO-FREQ-SCALING-08.
- **The board:** conformance, the profiler, and ArchBench tiny 27/27 pass.
  Every point's device cycle count is identical to the RTL simulation
  (`tools/r38/board_vs_sim.py`).
- **The clock:** the shell's counter reads 150 MHz.
  `measure_clock.py` -- the device's own cycle count over host-timed spin
  kernels -- reads 74.97 and 75.02 MHz.

The files, the image's sha256 and the routed path are in
[`evidence/r38/u280_75mhz/`](../../evidence/r38/u280_75mhz/README.md).
`DEV_DIV = 2` is now the wrapper's default.

