"""
TNCA: an experimental AOT compiler and RTL-simulation runtime.

Importing this package does nothing expensive. It starts no simulator, runs no
test and contacts no network -- doc/phase2_tasks_and_acceptance.md R2-02 asks
for that directly, and the way it is kept true is that the submodules which
CAN do those things are imported by the code that needs them, not here.

    from tnca.aot import compile_aot, inspect_artifact
    from tnca.runtime import Session
"""

__version__ = '0.2.0'

# Names, not modules: `from tnca import Session` should work without making
# `import tnca` pull in the runtime, the compiler and Triton.
__all__ = ['__version__', 'aot', 'runtime', 'abi', 'manifest', 'artifact',
           'cache', 'errors', 'transport']
