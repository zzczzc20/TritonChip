"""What ran where: the evaluator's check of Inductor's generated code.

Moved out of tnca/inductor.py on 2026-11-11, when that module became the
candidate's to change (how Inductor lowers to TNCA: block caps, GEMM tiles,
how attention is decomposed). This check is the evaluator's: the evaluation
takes this file from its own platform (task evaluation.platform.paths), never
from a candidate, and archbench calls it here, not through tnca.inductor,
which re-exports it for everyone else.
"""

# ---------------------------------------------------------------------------
# What ran where. Inductor's generated wrapper (torch._inductor.utils.
# run_and_get_code) is the record of a compiled graph: every Triton kernel it
# launches is a TNCA launch, and any computation it leaves on the host -- an
# extern kernel, an ATen call, a C++ kernel -- is a FALLBACK. Buffer
# allocation, reinterpretation and views are not computation and are not
# listed.
# ---------------------------------------------------------------------------
import re as _re

FALLBACK_PATTERNS = [
    (_re.compile(r'\bextern_kernels\.\w+'), 'extern kernel'),
    (_re.compile(r'\btorch\.ops\.aten\.(?!empty|empty_strided|as_strided|reinterpret)\w+'), 'ATen op'),
    (_re.compile(r'\baten\.(?!empty|empty_strided|as_strided)\w+\('), 'ATen op'),
    (_re.compile(r'\bcpp_fused\w*'), 'C++ kernel'),
]


def _executed(src):
    """The wrapper as it runs: no comments (Inductor annotates each kernel
    launch with the ATen ops it replaced), and nothing from the benchmark
    harness Inductor appends after `def benchmark_compiled_module`."""
    src = src.split('def benchmark_compiled_module', 1)[0]
    return '\n'.join(l for l in src.splitlines() if not l.lstrip().startswith('#'))


# A C++ kernel's argument types, from `name = async_compile.cpp_pybinding([...]`.
_CPP_SIGNATURE = _re.compile(r"\b(cpp_fused\w*)\s*=\s*async_compile\.cpp_pybinding\(\s*\[([^\]]*)\]")
_INTEGER_C_TYPES = _re.compile(r"^(const\s+)?(bool|u?int(8|16|32|64)_t)\s*\*?$")


def host_metadata_kernels(code):
    """C++ kernels whose every argument is an integer or bool (pointer or
    scalar): computation on host integer metadata the caller placed on the
    CPU (configure(): on the native device CPU tensors are host data), e.g.
    `cu_seqlens[1:] - cu_seqlens[:-1]` before a split. No floating-point
    value is computed on the host by these. Sorted names."""
    names = set()
    for src in code:
        for m in _CPP_SIGNATURE.finditer(src):
            types = [t.strip().strip("'\"") for t in m.group(2).split(',') if t.strip()]
            if types and all(_INTEGER_C_TYPES.match(t) for t in types):
                names.add(m.group(1))
    return sorted(names)


_ITEM = _re.compile(r'\b(buf\d+|arg\d+_\d+)\b[^\n]*\.item\(\)')


def scalar_reads(code):
    """`.item()` reads in the wrapper (a data-dependent size or a check's
    value), as dict(host=, device=): one of a buffer allocated on `tnca` is a
    device readback of one element; one of host memory (empty_strided_cpu,
    or a buffer a host C++ kernel reads or writes) is a host read. A graph input is counted as a
    device read (its device is not in the wrapper)."""
    host = device = 0
    for src in map(_executed, code):
        cpu = set(_re.findall(r'\b(buf\d+) = empty_strided_cpu\(', src))
        # A C++ kernel reads and writes host memory only: its buffers are
        # CPU ones, whatever reuse named them.
        for call in _re.findall(r'\bcpp_fused\w*\(([^()]*)\)', src):
            cpu.update(_re.findall(r'\b(buf\d+|arg\d+_\d+)\b', call))
        for m in _ITEM.finditer(src):
            if m.group(1) in cpu:
                host += 1
            else:
                device += 1
    return dict(host=host, device=device)


def host_fallbacks(code):
    """Host-side computation in the generated wrapper source(s), sorted.
    Host integer-metadata kernels (host_metadata_kernels) are not listed."""
    found = []
    metadata = set(host_metadata_kernels(code))
    for src in map(_executed, code):
        for pat, what in FALLBACK_PATTERNS:
            for m in pat.finditer(src):
                if m.group(0) in metadata:
                    continue
                found.append(f'{what}: {m.group(0)}')
    return sorted(set(found))


def triton_kernels(code):
    """Names of the Triton kernels the generated code defines."""
    return sorted({m.group(1) for src in code
                   for m in _re.finditer(r'def (triton_\w+)\(', src)})
