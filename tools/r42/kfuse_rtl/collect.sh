#!/bin/bash
# collect.sh -- pulls the raw results into evidence/r42/kfuse_rtl (run locally; needs the scratchpad helpers).
#   x86 host  ~/r42_kfuse_out/<run>/{synth,route}_*.rpt, *_endpoints.csv, vivado.log, run.log  -> ooc/<run>/*.gz
#   GB10      ~/r42/kfuse-rtl_out/final/*.log, *.gen.log, *.sha256, lists; mutation/                -> sim/
set -eu
S=${S:-/tmp/claude-1000/-home-ubuntu-TritonChip/3dd1d4a8-6f0c-47d5-bf28-f88428fc463d/scratchpad}
EV=$(cd $(dirname $0)/../../.. && pwd)/evidence/r42/kfuse_rtl
TMP=$(mktemp -d /tmp/claude-1000/kfuse_collect.XXXXXX)
mkdir -p $EV/ooc $EV/ooc_v1 $EV/sim $EV/netlist
$S/vx 'cd ~/r42_kfuse_out && tar czf - --exclude="*.dcp" --exclude="*.vvp" --exclude="netlist.v" --exclude="vec.hex" --exclude="xsim.dir" --exclude="*.wdb" --exclude="*.pb" .' > $TMP/ooc.tgz
mkdir -p $TMP/ooc && tar xzf $TMP/ooc.tgz -C $TMP/ooc
for d in $TMP/ooc/*/; do
    n=$(basename $d)
    case $n in
        nlvec) continue;;
        nl_*) dst=$EV/netlist/$n;;
        p2_*|h2_*|s2_*|p_mac|p_mac4) dst=$EV/ooc/$n;;
        *) dst=$EV/ooc_v1/$n;;          # runs of the first RTL (commit a13994fc/f891bab0), before the B-side gate
    esac
    mkdir -p $dst
    for f in $d*; do
        b=$(basename $f)
        [ -f $f ] || continue
        case $b in
            clk.xdc) cp $f $dst/;;
            *.gz) cp $f $dst/;;
            *) gzip -9 -c $f > $dst/$b.gz;;
        esac
    done
done
for f in run.log queue1.txt queue2.txt queue3.txt; do [ -f $TMP/ooc/$f ] && cp $TMP/ooc/$f $EV/ooc/ || true; done
$S/gb_rsh x 'cd ~/r42/kfuse-rtl_out/final && tar czf - f_*.log f_*.gen.log *.sha256 *.compile.log 2>/dev/null; true' < /dev/null > $TMP/sim.tgz
tar xzf $TMP/sim.tgz -C $EV/sim
$S/gb_rsh x 'cd ~/r42/kfuse-rtl_out && tar czf - final_tool_sha256.txt list_f.txt mutation/mutation_check.log mutation/vec.sha256 mutation/vec.gen.log matrix_f.log 2>/dev/null; true' < /dev/null > $TMP/sim2.tgz
tar xzf $TMP/sim2.tgz -C $TMP && cp $TMP/final_tool_sha256.txt $TMP/list_f.txt $EV/sim/ && cp $TMP/mutation/* $EV/sim/ 2>/dev/null || true
cp $TMP/matrix_f.log $EV/sim/ 2>/dev/null || true
# netlist simulation (GB10 iverilog + unisim): logs and the hashes of what was simulated
$S/gb_rsh x 'cd ~/r42/kfuse-rtl_out/nlsim && sha256sum netlist_g4.v netlist_g16.v netlist_g8w3.v nl_G4_W0.hex nl_G16_W0.hex nl_G8_W3.hex > nlsim_sha256.txt; tar czf - nlsim_sha256.txt g4.sim.log g16.sim.log g8w3.sim.clean.log g8w3.sim.mixed_run1_and_run2.log g4.compile.log g16.compile.log g8w3.compile.log 2>/dev/null; true' < /dev/null > $TMP/nlsim.tgz
tar xzf $TMP/nlsim.tgz -C $EV/netlist
# the vector files themselves are not kept (up to 90 MB each); their sha256 are in sim/*.sha256
rm -rf $TMP
echo collected into $EV
