Single-core RTL (rtl-1c) tiny tier, profile core-vrf-sync-v5-x8-g2c4-dp, `archbench.run --counters` records (device cycles = perf_cycles of the timed call).
tiny_base          run-3 core (VRF_FWD 0), tree before the parameter plumbing (RTL at default identical: tiny_base2 reproduces it, see below)
tiny_base2         the same from the immutable snapshot of the final tree (partial: gpt2_block and llama_decode both cycle-identical to tiny_base)
tiny_fwd2          VRF_FWD=1 through TNCA_RTL_PARAMS=VRF_FWD=1 (snapshot of the final tree)
tiny_nopf          VRF_PREFETCH=0 (the TLM's 'issue 2' world); records of points whose simulator build was pruned by a later build are runtime-faults and kept
tiny_behav         behavioural register file (USE_BANKED_VRF=0, read in X with forwarding): the bound of every VRF-read lever; partial
commands: ~/r42/r42_run.sh TREE core-vrf-sync-v5-x8-g2c4-dp tiny JOBS NAME --counters   (TNCA_RTL_PARAMS and TNCA_CACHE as in tools/r42/issue/README below)
