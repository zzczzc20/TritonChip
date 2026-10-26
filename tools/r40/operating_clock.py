"""Compatibility import for R40 tools; shared board-clock evidence validation."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'python'))
from tnca.operating_clock import qualified_clock, from_environment
