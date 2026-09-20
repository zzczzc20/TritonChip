"""
The command line -- doc/phase2_tasks_and_acceptance.md R2-02.

    tnca-aot compile | inspect | doctor
    tnca-sim run | doctor

Two rules the contract states and this file obeys.

**Importing does not do anything.** No test runs, no network call, no
simulator starts because somebody typed `import tnca`. Everything expensive is
behind a subcommand, and the compiler is imported lazily so a runtime-only
install can run `tnca-sim` without one.

**A failure is a non-zero exit with a stable code.** A kernel fault, a loader
error and a lost device each exit non-zero and print the code the Python API
would have raised. A CLI that prints a traceback has told the user about our
stack rather than about their program.
"""
import argparse
import importlib
import json
import os
import struct
import sys

from .errors import TncaError


def _fail(e, as_json=False):
    if as_json:
        print(json.dumps(dict(ok=False, **e.as_dict()), indent=2))
    else:
        print(str(e), file=sys.stderr)
    return 1


def _load_source(spec):
    """`module.py:function` or `package.module:function`."""
    if ':' not in spec:
        raise TncaError(
            f"{spec!r} does not name a function",
            code='ARGUMENT_INVALID',
            remedy='use path/to/kernels.py:add_kernel')
    where, name = spec.rsplit(':', 1)
    if where.endswith('.py'):
        import importlib.util
        mod_name = os.path.splitext(os.path.basename(where))[0]
        spec_obj = importlib.util.spec_from_file_location(mod_name, where)
        if spec_obj is None:
            raise TncaError(f"cannot read {where!r}", code='ARGUMENT_INVALID')
        module = importlib.util.module_from_spec(spec_obj)
        sys.modules[mod_name] = module
        spec_obj.loader.exec_module(module)
    else:
        module = importlib.import_module(where)
    if not hasattr(module, name):
        raise TncaError(f"{where} has no {name!r}", code='ARGUMENT_INVALID')
    return getattr(module, name)


def _kv(pairs, cast=str):
    out = {}
    for item in pairs or ():
        if '=' not in item:
            raise TncaError(f"{item!r} is not name=value",
                            code='ARGUMENT_INVALID')
        k, v = item.split('=', 1)
        out[k] = cast(v)
    return out


def _number(text):
    try:
        return int(text, 0)
    except ValueError:
        return float(text)


# ---------------------------------------------------------------------------
from .dump import STAGES as _DUMP_STAGES   # noqa: E402


def cmd_compile(args):
    from . import aot
    from . import dump as _dump
    fn = _load_source(args.source)

    options = _kv(args.option, _number) if args.option else {}
    # `--emit` without `--dump-dir` still has to make the compiler produce
    # the stage, but it should not leave files behind: the caller asked to
    # SEE something, not to store it. So the stages are collected in
    # memory (they always are) and only written when a directory is named.
    if args.dump_dir:
        options['dump_dir'] = args.dump_dir

    res = aot.compile_aot(
        fn,
        kernel_name=args.kernel,
        signature=_kv(args.signature),
        constexprs=_kv(args.constexpr, _number),
        profile=args.profile,
        options=options or None,
        source_kind=args.source_kind,
        output=args.output)

    if args.emit:
        text = res.asm.get(args.emit)
        if text is None:
            available = ', '.join(res.asm) or 'none'
            raise TncaError(
                f"this compile produced no {args.emit!r} stage",
                code='ARGUMENT_INVALID',
                detail={'available': sorted(res.asm)},
                remedy=f"stages from this source: {available}. A tnca-ir "
                       f"program has no .ttir because it never had one")
        sys.stdout.write(text if text.endswith('\n') else text + '\n')
        return 0

    if args.json:
        print(json.dumps(dict(ok=True, path=res.path,
                              cache_key=res.cache_key,
                              from_cache=res.from_cache,
                              kernel=res.manifest['identity']['kernel_name'],
                              report=res.report,
                              stages=sorted(res.asm),
                              dumped=res.dumped), indent=2))
    else:
        seg = res.manifest['code']['segments'][0]
        print(f"wrote {res.path}")
        print(f"  kernel   {res.manifest['identity']['kernel_name']}")
        print(f"  profile  {res.manifest['target']['profile']}")
        print(f"  source   {res.manifest['identity']['source_kind']}")
        print(f"  code     {seg['length']} bytes, sha256 {seg['sha256'][:16]}")
        print(f"  key      {res.cache_key[:16]}")
        if res.dumped:
            where = os.path.dirname(next(iter(res.dumped.values())))
            print(f"  dump     {where}")
            for stage in sorted(res.dumped):
                print(f"    {stage:<6s} "
                      f"{os.path.basename(res.dumped[stage])}")
    return 0


def cmd_inspect(args):
    from . import aot
    info = aot.inspect_artifact(args.artifact)
    if args.json:
        print(json.dumps(info, indent=2))
        return 0
    man = info['manifest']
    print(f"{info['path']}  ({info['bytes']} bytes)")
    for section in ('identity', 'target', 'execution', 'resources',
                    'semantics'):
        print(f"  [{section}]")
        for k, v in sorted(man.get(section, {}).items()):
            print(f"    {k:<28s} {v}")
    print("  [parameters]")
    for p in man['parameters']['signature']:
        print(f"    {p['name']:<12s} {p['type']:<14s} offset {p['offset']:>3d}"
              f"  {p.get('direction', '')}")
    print("  [segments]")
    for s in info['segments']:
        print(f"    {s['name']:<12s} {s['length']:>6d} bytes at "
              f"{s['offset']:>6d}  sha256 {s['sha256'][:16]}")
    return 0


def cmd_doctor(args):
    """What this installation can do. No simulator, no network."""
    from . import abi, manifest as M
    from .adapters import r1 as adapter
    rows = [
        ('package', os.path.dirname(os.path.abspath(__file__))),
        ('manifest format', f"{M.FORMAT_VERSION[0]}.{M.FORMAT_VERSION[1]}"),
        ('argument ABI', f"{abi.ABI_VERSION[0]}.{abi.ABI_VERSION[1]}"),
        ('compiler available', adapter.available()),
    ]
    try:
        import triton
        rows.append(('triton', triton.__version__))
    except Exception:
        rows.append(('triton', 'not installed (runtime-only is fine)'))

    from . import device as _device
    found = _device.locate('core-sim')
    rows.append(('device bench', found or
                 'not built -- run `tnca-sim prepare`'))
    if args.json:
        print(json.dumps(dict(ok=True, **{k: str(v) for k, v in rows}),
                         indent=2))
    else:
        for k, v in rows:
            print(f"  {k:<20s} {v}")
    return 0


def cmd_run(args):
    """Load an artifact and run it, through the product runtime."""
    from . import artifact
    from .runtime import Session

    man, _segs = artifact.load(args.artifact)
    params = man['parameters']['signature']
    values = _kv(args.arg, _number)

    session = Session(profile=man['target']['profile'],
                      deadline_cycles=args.deadline)
    try:
        mod = session.load(args.artifact)
        buffers, resolved = {}, {}
        for p in params:
            name = p['name']
            if p['type'] == 'device_ptr64':
                size = int(values.get(f"{name}.bytes", args.buffer_bytes))
                buf = session.allocate(size, name=name)
                buffers[name] = buf
                data = values.get(f"{name}.fill")
                if data is not None:
                    session.upload(buf, struct.pack('<f', float(data)) *
                                   (size // 4))
                resolved[name] = buf
            else:
                if name not in values:
                    raise TncaError(f"no value for {name!r}",
                                    code='ARGUMENT_INVALID',
                                    remedy=f"pass --arg {name}=<value>")
                resolved[name] = values[name]
        ev = session.submit(mod, resolved, grid=tuple(args.grid))
        session.wait(ev)
        out = {}
        for name, buf in buffers.items():
            if args.download and name in args.download:
                raw = session.download(buf, min(buf.size, 64))
                out[name] = list(struct.unpack(
                    f"<{len(raw) // 4}f", raw))[:16]
        if args.json:
            print(json.dumps(dict(ok=ev.ok, state=ev.state,
                                  job=ev.job_id,
                                  cycles=ev.device_cycles,
                                  error=(ev.error.as_dict() if ev.error
                                         else None),
                                  outputs=out), indent=2))
        else:
            print(f"job {ev.job_id}: {ev.state}"
                  + (f", {ev.device_cycles} device cycles"
                     if ev.device_cycles else ''))
            for name, vals in out.items():
                print(f"  {name}: {vals}")
            if ev.error:
                print(str(ev.error), file=sys.stderr)
        return 0 if ev.ok else 1
    finally:
        try:
            session.drain(timeout=5)
        except Exception:
            pass
        session.poisoned = True
        session.close()


# ---------------------------------------------------------------------------
def _common(p):
    p.add_argument('--json', action='store_true',
                   help='machine-readable output')


def aot_main(argv=None):
    ap = argparse.ArgumentParser(prog='tnca-aot',
                                 description='Compile a kernel ahead of time')
    sub = ap.add_subparsers(dest='cmd', required=True)

    c = sub.add_parser('compile', help='source -> artifact')
    c.add_argument('source', help='kernels.py:add_kernel')
    c.add_argument('-o', '--output', required=True)
    c.add_argument('--profile', default='core-sim')
    c.add_argument('--kernel', help='defaults to the function name')
    c.add_argument('--signature', action='append', metavar='NAME=TYPE')
    c.add_argument('--constexpr', action='append', metavar='NAME=VALUE')
    c.add_argument('--option', action='append', metavar='NAME=VALUE')
    c.add_argument('--source-kind', default='triton',
                   choices=('triton', 'tnca-ir'))
    _common(c)
    c.set_defaults(func=cmd_compile)

    # `-S`/`--emit` follow the compilers people already use: `gcc -S`
    # writes assembly, `clang -emit-llvm` writes the IR. `--dump-dir`
    # follows Triton's TRITON_DUMP_DIR, which is where anyone who has
    # debugged a Triton kernel will look first.
    c.add_argument('--emit', choices=list(_DUMP_STAGES), metavar='STAGE',
                   help='print one stage to stdout instead of the summary: '
                        + ', '.join(_DUMP_STAGES))
    c.add_argument('-S', dest='emit', action='store_const', const='asm',
                   help='shorthand for --emit asm')
    c.add_argument('--dump-dir', metavar='DIR',
                   help='write every stage into DIR/<kernel>/ '
                        '(same as TNCA_DUMP_DIR with TNCA_KERNEL_DUMP=1)')

    i = sub.add_parser('inspect', help='what an artifact says about itself')
    i.add_argument('artifact')
    _common(i)
    i.set_defaults(func=cmd_inspect)

    d = sub.add_parser('doctor', help='what this installation can do')
    _common(d)
    d.set_defaults(func=cmd_doctor)

    args = ap.parse_args(argv)
    try:
        return args.func(args)
    except TncaError as e:
        return _fail(e, getattr(args, 'json', False))


def cmd_prepare(args):
    """Build this profile's simulator, once.

    Explicit on purpose. A launch never builds one: that would put a compile
    inside every kernel call, which R2-02 forbids in as many words.
    """
    from . import device
    path = device.prepare(args.profile, force=args.force)
    if args.json:
        print(json.dumps(dict(ok=True, profile=args.profile, device=path)))
    else:
        print(f"  {args.profile}: {path}")
    return 0


def cmd_replay(args):
    """Run a saved failure again and say whether it failed the same way.

    The exit code is about the REPLAY, not about the kernel: 0 means the
    bundle reproduced what it recorded, which for a failure bundle means
    the kernel failed again. A bundle that no longer reproduces is the
    interesting case, and it exits 1.
    """
    from . import failure

    index, outcome = failure.replay(args.bundle, profile=args.profile)
    saved = index['outcome']
    same, why = failure.same_failure(saved, outcome)
    if getattr(args, 'json', False):
        print(json.dumps({'bundle': args.bundle, 'saved': saved,
                          'replayed': outcome, 'reproduced': same,
                          'why': why}, indent=2, sort_keys=True,
                         default=str))
    else:
        print(f"  kernel     {index['kernel']}  ({index['profile']}, "
              f"grid {'x'.join(str(g) for g in index['grid'])})")
        print(f"  saved      {saved.get('state')} "
              f"[{saved.get('code')}]")
        print(f"  replayed   {outcome.get('state')} "
              f"[{outcome.get('code')}]")
        print(f"  {'reproduced' if same else 'DID NOT REPRODUCE'}: {why}")
    return 0 if same else 1


def sim_main(argv=None):
    ap = argparse.ArgumentParser(prog='tnca-sim',
                                 description='Run an artifact on the '
                                             'simulated device')
    sub = ap.add_subparsers(dest='cmd', required=True)

    r = sub.add_parser('run', help='load and execute')
    r.add_argument('artifact')
    r.add_argument('--arg', action='append', metavar='NAME=VALUE',
                   help='a scalar, or <ptr>.bytes / <ptr>.fill for a buffer')
    r.add_argument('--grid', type=int, nargs=3, default=[1, 1, 1])
    r.add_argument('--download', action='append', metavar='NAME')
    r.add_argument('--buffer-bytes', type=int, default=64)
    r.add_argument('--deadline', type=int, default=2_000_000,
                   help='device cycles, not a host timeout')
    _common(r)
    r.set_defaults(func=cmd_run)

    p = sub.add_parser('prepare', help="build a profile's simulator, once")
    p.add_argument('--profile', default='core-sim')
    p.add_argument('--force', action='store_true')
    _common(p)
    p.set_defaults(func=cmd_prepare)

    r = sub.add_parser('replay', help='run a saved failure bundle again')
    r.add_argument('bundle')
    r.add_argument('--profile', default=None,
                   help="override the bundle's profile; the default is the "
                        "one it was captured on")
    _common(r)
    r.set_defaults(func=cmd_replay)

    d = sub.add_parser('doctor', help='what this installation can do')
    _common(d)
    d.set_defaults(func=cmd_doctor)

    args = ap.parse_args(argv)
    try:
        return args.func(args)
    except TncaError as e:
        return _fail(e, getattr(args, 'json', False))
