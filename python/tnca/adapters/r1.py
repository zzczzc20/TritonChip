"""
The R1 boundary -- doc/r2_interfaces.md.

Everything R2 uses from R1 passes through here: the compiler, the assembler,
the target descriptions, the profile spec. Nothing else in `tnca.*` imports
`tnca_cc`, `tools.asm` or `tools.r1`, and PKG-02 checks that by installing the
package and looking.

Two things this module owes its callers.

**A stable shape.** R1 returns what it returns; R2 needs code bytes, an entry
table and a resource report. Translating between the two is this module's job,
not the caller's, so that when an R1 signature changes the diff is here.

**A record of what it called.** The manifest says which compiler produced an
artifact, and "the compiler" is not one thing: it is a frontend, a code
generator, an assembler and a set of safety checks, each with a version. The
adapter collects those into `provenance()` as it works, so the manifest
records what actually ran rather than what was configured to run.

R1's safety checks are not optional here. `emit()` runs code generation with
the ownership and footprint checking on, whatever options the caller passed:
CMP-03 requires R1's own negative cases to be refused through the product
entry point with debug and optimisation options set, and an option that turns
a safety check off is a safety check that is off.
"""
import hashlib
import json
import os
import sys

from ..errors import (CapabilityError, CompileError, ProfileMismatch,
                      TncaError, UnsupportedError)

# The repository this package was built from. In an installed wheel these
# live inside the package; in a checkout they are where R1 put them.
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, '..', '..', '..'))

for _p in (os.path.join(_ROOT, 'compiler', 'python'),
           os.path.join(_ROOT, 'tools', 'asm'),
           os.path.join(_ROOT, 'tools', 'r1')):
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.append(_p)


def _r1():
    """Import R1 lazily, and turn its absence into a diagnostic.

    Lazy because the runtime must load and run an artifact with no compiler
    installed at all (PKG-03). An import at module scope would make
    `import tnca.runtime` fail on a machine that is behaving correctly.
    """
    try:
        from tnca_cc import TargetInfo
        from tnca_cc.codegen import CodeGen
        from tnca_cc.diagnostics import Diagnostic
        from tnca_cc.frontend import TTIRImporter, Unsupported
        from tnca_cc.ir import Builder, make_kernel
        import tnca_asm
        return dict(TargetInfo=TargetInfo, CodeGen=CodeGen,
                    Diagnostic=Diagnostic, TTIRImporter=TTIRImporter,
                    Unsupported=Unsupported, Builder=Builder,
                    make_kernel=make_kernel, asm=tnca_asm)
    except ImportError as e:
        raise CompileError(
            "the TNCA compiler is not available in this environment",
            detail={'import_error': str(e)},
            remedy='install the package with its compile extra; a '
                   'runtime-only install can load and run artifacts but '
                   'cannot build them')


def available():
    """Can this environment compile? A question the CLI asks before trying."""
    try:
        _r1()
        return True
    except CompileError:
        return False


class Provenance:
    """What the adapter actually called, collected as it goes."""

    def __init__(self):
        self.entries = {}

    def record(self, what, value):
        self.entries[what] = value

    def as_dict(self):
        return dict(self.entries)


def _sha(text):
    if isinstance(text, str):
        text = text.encode('utf-8')
    return hashlib.sha256(text).hexdigest()


def profile(name):
    """The profile spec entry and its configuration hash."""
    try:
        sys.path.append(os.path.join(_ROOT, 'tools', 'r1'))
        import profiles as P
    except ImportError as e:
        raise ProfileMismatch(
            f"cannot read the profile spec: {e}",
            remedy='the package must ship spec/r1_profiles.json and the '
                   'targets generated from it')
    try:
        p = P.profile(name)
    except KeyError as e:
        raise ProfileMismatch(str(e))
    return dict(name=name, spec=p, config_hash=P.config_hash(name),
                target_file=p['target_file'])


def target_for(name, prov=None):
    """R1's TargetInfo for a profile, with its fingerprint recorded."""
    r1 = _r1()
    info = profile(name)
    target = r1['TargetInfo'].load(info['spec']['target_file'].rstrip('.json'))
    if prov is not None:
        prov.record('profile', name)
        prov.record('config_hash', info['config_hash'])
        prov.record('target_name', target.name)
        prov.record('isa_version', target.isa_version)
    return target, info


def require_capabilities(target, needed):
    """Every capability the code will use, checked before it is emitted.

    Before, not after: a capability failure after emission is an artifact
    that exists and cannot run, and ART-03 wants the refusal to happen with
    no side effect at all.
    """
    caps = target.capabilities if isinstance(target.capabilities, dict) \
        else dict(target.capabilities)
    missing = [c for c in needed if not caps.get(c)]
    if missing:
        raise CapabilityError(
            f"profile {target.name!r} does not have "
            f"{', '.join(sorted(missing))}",
            detail={'needed': sorted(needed)},
            remedy='compile for a profile that has them, or use a lowering '
                   'that does not need them')


def emit(kernel, target, *, prov=None, check_protocol=True):
    """TNCA IR -> assembly text, with R1's safety checks on.

    `check_protocol` selects the channel model, which is about a
    producer/consumer ring and not about safety; the ownership and footprint
    checks are unconditional inside R1 and are not exposed here on purpose.
    There is deliberately no argument that turns them off.
    """
    r1 = _r1()
    cg = r1['CodeGen'](kernel, target, check_protocol=check_protocol)
    try:
        cg.assign_shared()
        cg.run()
    except r1['Unsupported'] as d:
        raise UnsupportedError(str(d).splitlines()[0],
                               where='code generation',
                               detail={'diagnostic': str(d)})
    except r1['Diagnostic'] as d:
        raise CompileError(str(d).splitlines()[0],
                           where='code generation',
                           detail={'diagnostic': str(d)})
    text = cg.b.text()
    if prov is not None:
        prov.record('codegen', 'tnca_cc.codegen.CodeGen')
        prov.record('ownership_check', 'unconditional')
        prov.record('assembly_sha256', _sha(text))
    return text, cg


def assemble(asm_text, *, prov=None):
    """Assembly text -> machine code bytes and a symbol table.

    Bytes, not the hex image R1's CLI writes: an artifact stores code, and
    a text file of hex words is a different thing that happens to contain it.
    """
    r1 = _r1()
    asm = r1['asm'].Assembler()
    try:
        records = asm.assemble(asm_text)
    except r1['asm'].AsmError as e:
        raise CompileError(f"assembly failed: {e}", where='assembler')
    if not records:
        raise CompileError("the assembler produced no code")
    end = max(a + n for a, n, _v, _s in records)
    image = bytearray(((end + 7) // 8) * 8)
    for addr, nbytes, value, _src in records:
        image[addr:addr + nbytes] = int(value).to_bytes(nbytes, 'little')
    code = bytes(image)
    if prov is not None:
        prov.record('assembler', 'tools/asm/tnca_asm.py')
        prov.record('code_bytes', len(code))
        prov.record('code_sha256', _sha(code))
    return code, dict(asm.symbols)


def import_ttir(module, func_name, arg_values, *, param_names, target,
                prov=None):
    """A Triton TTIR module -> TNCA IR, through R1's importer."""
    r1 = _r1()
    imp = r1['TTIRImporter'](target, func_name)
    try:
        kernel, builder = imp.run(module, func_name, arg_values,
                                  param_names=param_names)
    except r1['Unsupported'] as d:
        raise UnsupportedError(str(d).splitlines()[0],
                               where=f"TTIR function {func_name!r}",
                               detail={'diagnostic': str(d)})
    except r1['Diagnostic'] as d:
        raise CompileError(str(d).splitlines()[0],
                           where=f"TTIR function {func_name!r}",
                           detail={'diagnostic': str(d)})
    if prov is not None:
        prov.record('frontend', 'tnca_cc.frontend.TTIRImporter')
        prov.record('bound_parameters', list(imp.arg_names))
    return kernel, builder, imp


def new_kernel(name):
    """An empty TNCA IR kernel and a builder for role 0."""
    r1 = _r1()
    kernel = r1['make_kernel'](name)
    builder = r1['Builder'](kernel, kernel.role(0).region)
    return kernel, builder
