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

The platform states what `ap_clk` will be, in its own `hw.hpfm` inside
`hw.xsa`:

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

`defaultClock="0"` is 300 MHz. That is what `ooc.xdc` constrains. The 500 MHz
entry is id 1: it exists, it is not the default, and taking it would be a change
of design rather than a change of build flag.

The device does not run at 300 MHz. `rtl/board/tnca_u50_xo.v` divides `ap_clk`
with `BUFGCE_DIV #(.BUFGCE_DIVIDE(8))`, so the device runs at **37.5 MHz**.
`link.sh` passes no `--kernel_frequency`, and it checks the resulting frequency
rather than assuming it -- `v++` can lower a kernel clock on its own and still
report success (`AUTO-FREQ-SCALING-04`, quoted in `link.sh`).

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
measurement, not a unilateral relaxation, which the working agreement lists as
something not to do ("不要为了'跑起来'直接修改：clock constraint..."). The
3.333 ns script was left in place untouched so the two can be compared.

## What the measurements do not cover

The 37.5 MHz figure rests on the device having routed at 25.231 ns under a 40 ns
constraint, which is under the 26.667 ns that 300/8 gives it. That is 1.4 ns of
margin, and it is margin on a run that was not asked to try.

Neither run contained `rtl/fpga/tnca_axi_lite.v`, `rtl/fpga/tnca_u50_top.v`, or
`rtl/board/tnca_u50_xo.v`. The adapter and the clock crossing are timed at
`ap_clk` -- 3.333 ns -- and nothing above says they meet it. `link.sh` is what
would establish that, and it has not finished.

So: the divider ratio, the device frequency, and whether the whole thing closes
are three separate questions, and only the middle one has been measured.

## If the link fails on timing

Do not lower the frequency. That is not a build flag away from being correct:
`ooc.xdc` and the divider comment in `rtl/board/tnca_u50_xo.v` both carry
numbers, `link.sh` checks the result against them, and a link that closed at a
lower clock is a different design from the one that was measured. The failure
mode to reach for is the report, not the constraint.
