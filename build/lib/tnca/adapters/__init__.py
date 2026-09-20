"""
Adapters: the only places this package reaches into another one.

`doc/r2_interfaces.md` says why there is exactly one module per foreign
interface. R1's internal signatures are still moving, and a project with
twenty call sites into a moving interface has twenty things to fix per change
and no list of them. Here there is a list, and it is this directory.
"""
