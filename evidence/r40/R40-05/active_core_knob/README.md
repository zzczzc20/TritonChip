# Frozen active-core knob reaches the physical dispatcher

The original runtime exposed explicit masks but never consumed the frozen TNCA_ACTIVE_CORES count knob. Source 85fa7d8 maps counts 1/2/3 to masks 1/3/7, rejects invalid counts, conflicting explicit masks and capacity excess, and separates cached sessions by count. Defaults are unchanged.

These three runs use unchanged standard Torch `(a+b)*0.5`, 4097 elements, on the three-core mailbox RTL implementation. Each has one whole-grid launch, 257 programs issued/completed exactly once, numerical correctness and zero kernel tensor staging. Peak active cores are 1, 2 and 3 respectively; inactive cores have zero issues, completions and occupied cycles. This is functional RTL evidence, not actual three-SLR placement or board scaling. No cycle counts are converted to seconds, and no required gate is promoted. These points do not substitute for frozen inventories.

Replay from immutable 85fa7d8 on the jump host: source ~/tc_env/env.sh, set PYTHONPATH=python:compiler/python:tools/asm and CUDA_VISIBLE_DEVICES empty. For each count 1/2/3, run `TNCA_ACTIVE_CORES=$count TNCA_HOST_TIMEOUT=3600 ~/tc_env/torchenv/bin/python -u tools/r40/native_multicore_tests.py --cores 3`, preserving a new log.
