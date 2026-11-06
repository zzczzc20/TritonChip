"""Select the compile-only harness profile without changing product files."""
import os
from pathlib import Path
import runpy
import sys
root = Path(sys.argv.pop(1)).resolve()
profile = sys.argv.pop(1)
sys.path.insert(0, str(root / 'tools/r40'))
import v3_compile_preflight
v3_compile_preflight.HARNESS['TNCA_PROFILE'] = profile
sys.argv[0] = str(root / 'tools/performance_harness/default_change_audit.py')
runpy.run_path(sys.argv[0], run_name='__main__')
