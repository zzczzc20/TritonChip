#!/bin/bash
# R41 G2: compile-only Hub FP8 kernels for the four profiles, two at a time.
cd ~/r41bw_aud
mkdir -p src_41661b75e0e3 && tar xf ~/r41bw_aud_41661b75e0e3.tar -C src_41661b75e0e3
(cd src_41661b75e0e3 && ~/tc_env/torchenv/bin/python -c 'import sys; sys.path[:0]=[".","python","compiler/python"]; from archbench.provenance import software_tree; print(software_tree(".", refresh=True))' > TREE)
./hub.sh v5x8 src_92b2c85b1518 core-vrf-sync-v5-x8 & ./hub.sh g2v0 t_g2v0-ctx2 core-vrf-sync-g2v0-ctx2 & wait
./hub.sh g2v2 t_g2v2-ctx2-bar8 core-vrf-sync-g2v2-ctx2-bar8 & ./hub.sh g2final src_41661b75e0e3 core-vrf-sync-v5-x8-g2 & wait
echo done > hub3.done
