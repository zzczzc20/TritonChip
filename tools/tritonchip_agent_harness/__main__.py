"""Allow ``python -m tritonchip_agent_harness`` with ``tools`` on the path."""

from .cli import main

raise SystemExit(main())
