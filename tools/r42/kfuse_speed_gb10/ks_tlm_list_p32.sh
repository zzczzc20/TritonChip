#!/bin/bash
CFG=$1; OUTN=$2; IDF=$3
rm -rf $HOME/r42/kfuse-speed_out/tlm/$OUTN
exec $HOME/r42/kfuse-speed_out/ks_tlm_p32.sh $CFG $OUTN $HOME/r42/kfuse-speed_out/tlm-build2/tnca-chip $(cat $IDF)
