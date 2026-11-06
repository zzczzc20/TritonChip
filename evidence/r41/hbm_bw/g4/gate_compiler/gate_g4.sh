#!/bin/bash
# G4 non-regression gate (discipline 2.4.1) for the default-off tiled element-wise lowering.
# usage: gate_g4.sh TREE_LABEL   (head or base); trees staged under ~/r41g4_gate/{head,base}
set -u
source ~/tc_env/env.sh
G=~/r41g4_gate; T=$1; R=$G/$T; L=$G/logs/$T; mkdir -p $L
P=~/tc_env/torchenv/bin/python
LLVM=~/tc_env/root/usr/lib/llvm-20
export R1_STRICT=1
# The torchenv carries an editable tnca install from ~/tritonchip; import this tree's packages.
export PYTHONPATH=$R/python:$R/compiler/python:$R
stamp() { date -u +%FT%TZ; }
lane() { local n=$1; shift
  echo "# $(stamp) start $n: $*" > $L/$n.log
  ( cd $R && nice -n 10 "$@" ) >> $L/$n.log 2>&1; local rc=$?
  echo "# $(stamp) end $n exit $rc" >> $L/$n.log; echo "$n $rc" >> $L/exit_codes.txt; }
: > $L/exit_codes.txt
echo "tree $T $(cat $R/.gate_commit) started $(stamp) host $(hostname) PYTHONPATH=$PYTHONPATH" > $L/run.txt
( cd $R && $P -c "import tnca, tnca_cc; print('imports', tnca.__file__, tnca_cc.__file__)" ) >> $L/run.txt 2>&1
tree=$(cd $R && $P -c 'from archbench.provenance import software_tree; print(software_tree(".", refresh=True))')
echo "software_tree $tree" >> $L/run.txt
lane audit-run $P tools/performance_harness/default_change_audit.py run --source-root $R --software-tree $tree --label $T --out $G/audit-$T --jobs 3
for t in core core-vrf unit lint top compiler-rtl triton-e2e r40-pipeline-checks r40-multicore-shell \
         r40-grid-dispatch r40-cache-maintenance r40-coherent-host r40-core-parameters r41-xpose8-rtl lsu-reqreg; do
  lane make-$t make $t PYTHON=$P TRITON_PYTHON=$P
done
if [ $T = head ]; then
  [ -x $R/compiler/build/tools/tnca-opt/tnca-opt ] || lane tnca-opt make tnca-opt LLVM_PREFIX=$LLVM -j3
  lane compiler make compiler PYTHON=$P LLVM_PREFIX=$LLVM
  lane pointwise-tiles $P compiler/test/test_pointwise_tiles.py -v
fi
echo "finished $(stamp)" >> $L/run.txt
