"""
Per-stage dumps of the IR and the assembly.

This follows what Triton does, because people already know how that works
and a second convention would be a second thing to learn. Triton writes
each compilation stage into a directory and makes the same text available
on the compiled kernel as `kernel.asm[stage]`; so does this.

    TNCA_KERNEL_DUMP=1        turn dumping on         (TRITON_KERNEL_DUMP)
    TNCA_DUMP_DIR=<path>      where to write it       (TRITON_DUMP_DIR)
    TNCA_ALWAYS_COMPILE=1     never serve from cache  (TRITON_ALWAYS_COMPILE)

or, per call, `compile_aot(..., options={'dump_dir': '/tmp/x'})`.

The stages, in the order they happen:

    <kernel>.ttir   the Triton IR the frontend produced. Triton sources
                    only -- a TNCA IR program never had one, and an empty
                    file would suggest it did.
    <kernel>.tnir   the TNCA IR after import, before register allocation
    <kernel>.s      the assembly
    <kernel>.hex    the machine image, one 64-bit word per line, which is
                    the form the R1 tools already read
    <kernel>.json   what the other files are and what produced them

Two decisions worth stating, because both could reasonably have gone the
other way:

A dump does not change the artifact. `dump_dir` and `debug` are kept out
of the cache key, so turning a dump on cannot hand you a different kernel
than the one you were debugging -- which is the whole point of looking.
Triton keeps TRITON_KERNEL_DUMP out of its key for the same reason.

And asking for a dump implies a compile. A cache hit is a request not to
run the compiler; a dump is a request to see it run. When both are asked
for, the compile happens -- otherwise the first run of a session writes
the files and every later one silently does not, which is the most
confusing possible behaviour.
"""
import json
import os

# stage -> file extension. The assembly gets `.s` because that is what an
# assembly file is called; the IRs get their own names the way Triton's
# .ttir and .ttgir do.
SUFFIX = {
    'ttir': '.ttir',
    # What the pinned upstream canonicalizer and CSE made of it. Written
    # only above O0, because at O0 nothing ran and an identical second
    # file would suggest something had. Phase 3.5 R35-OBS-01 wants the
    # input and the optimised IR both visible, which is two files.
    'ttir_opt': '.opt.ttir',
    # After the inliner, which runs at EVERY optimisation level because
    # Triton's library functions arrive as `tt.call` and no backend can
    # lower a call. Worth its own file: the difference between this and
    # `.ttir` is the whole of tl.exp, tl.sum and friends appearing, and
    # that is usually what a reader is looking for.
    'ttir_inlined': '.inlined.ttir',
    'tnir': '.tnir',
    # The TNCA IR after the bridge merged its duplicate pure values.
    # Written only when something merged, for the same reason as
    # `.opt.ttir`.
    'tnir_opt': '.opt.tnir',
    # After the R39-01 scheduler reordered each basic block. Written only
    # when the kernel was compiled with schedule=1.
    'tnir_sched': '.sched.tnir',
    # After R39-05 made each launch one program per Role Context. Written
    # only when the kernel was compiled with rctx_spmd=1 and allowed it.
    'tnir_spmd': '.spmd.tnir',
    # After R39-12 made each program a producer and a consumer on a Role
    # Context pair. Written only when compiled with rctx_roles=1 and allowed.
    'tnir_roles': '.roles.tnir',
    'tnir_single_role_pipeline': '.single-role-pipeline.tnir',
    'asm': '.s',
    'hex': '.hex',
}

# The order they are produced in, which is the order a reader wants them.
STAGES = ('ttir', 'ttir_inlined', 'ttir_opt', 'tnir',
          'tnir_opt', 'tnir_spmd', 'tnir_roles', 'tnir_single_role_pipeline', 'tnir_sched', 'asm', 'hex')

# Stages that exist only for some compiles. `ttir` is absent for a TNCA IR
# source, which never had one; `ttir_opt` is absent at O0, where no pass
# ran. A file that is written anyway would say something happened.
CONDITIONAL_STAGES = frozenset({'ttir', 'ttir_inlined',
                                'ttir_opt', 'tnir_opt', 'tnir_spmd', 'tnir_roles', 'tnir_single_role_pipeline',
                                'tnir_sched'})

# What every compile produces, whatever its source kind or optimisation
# level. A stage missing from THIS list is a bug.
ALWAYS = tuple(s for s in STAGES if s not in CONDITIONAL_STAGES)

ENV_ENABLE = 'TNCA_KERNEL_DUMP'
ENV_DIR = 'TNCA_DUMP_DIR'
ENV_ALWAYS = 'TNCA_ALWAYS_COMPILE'


def _truthy(value):
    return str(value).strip().lower() not in ('', '0', 'false', 'no', 'off')


def default_dir():
    """Where dumps go when nothing says otherwise.

    `~/.tnca/dump`, next to Triton's `~/.triton/dump` rather than inside
    the build tree: a debug artefact that lands in the checkout shows up
    in `git status` and then in somebody's commit.
    """
    return os.path.join(os.path.expanduser('~'), '.tnca', 'dump')


def requested(options=None):
    """Whether this compile should write dumps, and where.

    Returns a directory or None. The explicit option wins over the
    environment, so a caller that asked for a specific directory gets it
    even in a shell that has dumping switched on somewhere else.
    """
    options = options or {}
    explicit = options.get('dump_dir')
    if explicit:
        return str(explicit)
    if _truthy(os.environ.get(ENV_ENABLE, '')):
        return os.environ.get(ENV_DIR) or default_dir()
    return None


def always_compile():
    """Whether the cache should be bypassed."""
    return _truthy(os.environ.get(ENV_ALWAYS, ''))


def hex_image(code, width=8):
    """The machine image as one word per line, low byte first.

    The same form tools/asm writes and the device bench reads, so a dumped
    kernel can be run without converting it.
    """
    out = []
    for i in range(0, len(code), width):
        word = code[i:i + width].ljust(width, b'\x00')
        out.append(f"{int.from_bytes(word, 'little'):0{width * 2}x}")
    return '\n'.join(out) + ('\n' if out else '')


class Dump:
    """The stages of one compile, collected as they happen.

    Collected whether or not anyone asked for them: they are already in
    memory at the moment each stage finishes, and keeping them costs a
    reference. They become `CompileResult.asm`, which is how Triton
    exposes the same thing, so `result.asm['tnir']` works without a file
    having been written anywhere.
    """

    def __init__(self, kernel_name):
        self.kernel_name = kernel_name
        self.stages = {}

    def record(self, stage, text):
        if stage not in SUFFIX:
            raise KeyError(f"unknown stage {stage!r}; "
                           f"known: {', '.join(STAGES)}")
        if text is not None:
            self.stages[stage] = text

    def __contains__(self, stage):
        return stage in self.stages

    def __getitem__(self, stage):
        return self.stages[stage]

    def get(self, stage, default=None):
        return self.stages.get(stage, default)

    def keys(self):
        return [s for s in STAGES if s in self.stages]

    def write(self, directory, *, metadata=None):
        """Write every collected stage. Returns the paths written.

        One directory per kernel name, so two kernels do not overwrite each
        other and a rebuild of the same kernel does.
        """
        target = os.path.join(directory, self.kernel_name)
        os.makedirs(target, exist_ok=True)
        written = {}
        for stage in self.keys():
            path = os.path.join(target,
                                f"{self.kernel_name}{SUFFIX[stage]}")
            with open(path, 'w') as fh:
                fh.write(self.stages[stage])
                if not self.stages[stage].endswith('\n'):
                    fh.write('\n')
            written[stage] = path

        meta = dict(metadata or {})
        meta['kernel_name'] = self.kernel_name
        meta['stages'] = {s: os.path.basename(p) for s, p in written.items()}
        path = os.path.join(target, f"{self.kernel_name}.json")
        with open(path, 'w') as fh:
            json.dump(meta, fh, indent=2, sort_keys=True, default=str)
            fh.write('\n')
        written['json'] = path
        return written
