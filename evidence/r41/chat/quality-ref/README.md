# CPU reference answers for the registered chat-quality check

Domain: **CPU** (GB10, `~/flux-venv`, torch 2.14.0+cu130 without a GPU). Recorded on 2026-10-28 with
`tools/models/chat_quality.py --model {vl,q35} --device cpu --spec spec/r41/chat/quality-v1.json`, full
checkpoints, the CPU W8A8 reference route (exact dot, IEEE scale division), from source commit 0a8d8db plus the
uncommitted loader, quality-runner and gate changes that the following commit adds. They were archived before any
device run of the check and are its expected answers (`vl.json`, `q35.json`; logs beside them).
