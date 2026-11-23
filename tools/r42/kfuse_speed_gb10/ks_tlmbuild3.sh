#!/bin/bash
cd $HOME/r42/kfuse-speed || exit 2
source $HOME/tc_env/env.sh
nice -n 19 cmake -S tools/performance_harness/systemc -B $HOME/r42/kfuse-speed_out/tlm-build3 -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH=$HOME/tc_env/systemc-3.0.2/install && nice -n 19 cmake --build $HOME/r42/kfuse-speed_out/tlm-build3 --target tnca-chip --parallel 3
echo BUILD_EXIT=$?
