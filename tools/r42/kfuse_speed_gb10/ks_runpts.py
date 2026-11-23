import sys
import xl_points  # noqa: F401  registers the proxy points (needs archbench.xl importable)
from archbench.run import main
main(sys.argv[1:])
