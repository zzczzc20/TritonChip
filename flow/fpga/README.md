# Building the U50 kernel

`flow/fpga/ooc.xdc` ends with a pointer to this file for "the measured one". This
is that number, where it came from, and what it does not cover.

**Nothing in this directory has been on a board.** No `.xclbin` built here has
been loaded. Loading one changes hardware state and is a separate decision that
has not been taken.

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
IP, and `rtl/board/tnca_u50_xo.v` divides by four -- so the device runs at
**37.5 MHz**, which is where it ran before, when the bus was at 300 and the
divide was eight. The bus came down because the first link failed on it; the
device was deliberately held still so that every measurement about it survives.

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
on its own. The remainder all miss a single 3.333 ns period by under two
nanoseconds. `u50/reports/phase3_link_report.md` is the full attribution.

That is the whole reason the bus is at 150 MHz now: doubling the period reaches
every one of those paths at once, and dividing by four instead of eight keeps the
device exactly where the +6.494 ns was measured.

## What the measurements do not cover

The 37.5 MHz figure rests on the device having routed at 25.231 ns under a 40 ns
constraint, which is under the 26.667 ns that 150/4 gives it. That is 1.4 ns of
margin, and it is margin on a run that was not asked to try.

Neither OOC run contained `rtl/fpga/tnca_axi_lite.v`, `rtl/fpga/tnca_u50_top.v`,
or `rtl/board/tnca_u50_xo.v`. The adapter and the clock crossing are timed by the
link at `ap_clk` -- 6.667 ns -- and the first link is what showed they do not
close at 3.333 ns.

And nothing here says the clock is 150 MHz, only that 150 MHz was asked for. The
kernel clock is an MMCM output -- `aclk_kernel_00_hierarchy/clkwiz_aclk_kernel_00`
in the link's own `vivado.log` -- and Vitis reaches it by re-solving that MMCM and
emitting a `create_generated_clock` for the divide and multiply it picked
(`write_user_impl_clock_constraint` in `ocl_util.tcl`). So the figure `link.sh`
reads back out of the `.xclbin` is the one the solver was handed. The shell counts
the clock it actually gets, in `frequency_counter_aclk_kernel_00` at `0x9000`, and
reading that means loading the `.xclbin`.

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
