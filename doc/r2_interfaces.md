# R2 interfaces: what depends on what, and where R1 is touched

Contract: [`phase2_tasks_and_acceptance.md`](phase2_tasks_and_acceptance.md).
This file is R2-01's interface freeze. It is a set of rules about dependency
direction, not a description of finished code; each rule names the check that
enforces it, so "we moved a directory" cannot be mistaken for progress.

## The dependency graph

```text
public CLI / Python API
   ├── compiler driver ──── Triton adapter ──┐
   │                   └──  MLIR passes   ───┼── safe emission (R1 checks)
   │                   └──  TNCA IR adapter ─┘
   └── runtime ─── artifact / ABI / admission ─── scheduler ─── transport ─── RTL

verification ── calls the public interfaces above, plus its own oracles
                and fault injection
```

Two directions are forbidden, and both have a test:

- **the product may not import the tests.** `tnca.*` importing anything under
  `compiler/test/` or `tools/` fails PKG-02. The R1 test harness holds real
  product logic today — assembling a program, laying out buffers — and R2
  moves that logic into product modules so the tests can call it, rather than
  the reverse.
- **the runtime may not import the compiler.** A runtime-only environment has
  no Triton, no LLVM and no compiler, and must still load an artifact and run
  it (PKG-03). An import that only happens on an error path still breaks
  this, so it is checked by installing without them, not by reading the code.

## Where R1 is touched, and only there

R2 uses R1's compiler, assembler, target descriptions and profile tooling.
Every one of those uses goes through **`tnca.adapters.r1`** and nowhere else.
This is not tidiness. R1's internal signatures are still moving — the contract
says so — and a project with twenty call sites into a moving interface has
twenty things to fix per change and no list of them.

| R1 thing R2 needs | What R2 uses it for | Adapter surface |
|---|---|---|
| `tnca_cc.frontend.TTIRImporter` | Triton source → TNCA IR | `adapters.r1.import_ttir` |
| `tnca_cc.codegen.CodeGen` | TNCA IR → assembly, with the R1 safety checks | `adapters.r1.emit` |
| `tnca_cc.target.TargetInfo` | capabilities and resource budgets | `adapters.r1.target_for`, `adapters.r1.require_capabilities` |
| `tools/asm/tnca_asm.py` (`tnca_asm`) | assembly → machine code | `adapters.r1.assemble` |
| `spec/r1_profiles.json` (via `tnca.profiles`) | profile spec, config hash (replaces `tools/r1/profiles.py`) | `adapters.r1.profile` |
| `tnca_cc.ir.make_kernel`, `tnca_cc.ir.Builder` | the TNCA IR second input (kernel & builder) | `adapters.r1.new_kernel` |

The adapter records which R1 entry points it calls and their versions, and
that record goes into the manifest. An artifact therefore says which compiler
produced it, not merely that one did.

## What the boundary must not be used to do

The support boundary is a statement about what has been verified. It is not a
way to drop something R1 delivered:

- R1's dedicated GEMM, multi-role and rolled pipeline paths stay, and stay
  tested. They are reached through the **`tnca-ir` input**, labelled as such.
- a capability that has not moved to the real Triton frontend is reported as
  a TNCA IR input. Writing it up as "Triton GEMM works" would be false, and
  DEP-03 exists to catch exactly that phrasing.
- every path, old or new, runs the R1 footprint and ownership checks before
  emitting. CMP-03 feeds R1's own negative cases in through the product
  entry point and requires each to be refused — with debug and optimisation
  options on, because an option that turns a safety check off is a safety
  check that is off.

## Versioned contracts, frozen now

| Contract | File | Rule |
|---|---|---|
| Artifact / manifest | `spec/r2_manifest_v1.schema.json` | unknown or higher major **or minor** refused; unknown field refused; duplicate JSON key refused |
| Argument ABI | `spec/r2_abi_v1.json` | six types, 8-byte slots, `offset = 8×index`, high half zero, f32 as bits |
| Error codes | `python/tnca/errors.py` | a code is public: never renamed, never reused for a different meaning |
| Required cases | `spec/r2_required_cases.json` | 84 IDs, each with an entry point and the profiles it expands over |

`optional_metadata` is the only place an ignorable extension may live, and the
loader does not read it. Everywhere else, unknown means refused — a typo'd
`capabilties` must not quietly become an artifact that requires no
capabilities.

## R1 inputs this phase depends on

`tools/r2/baseline.py` (DEP-01) refuses to proceed without all of these, and
refuses a *partial* R1 specifically, because that is the state R1 is in while
it is being worked on rather than a prerequisite that has been met:

- an R1 evidence directory that R1's **own** verifier accepts — R2 does not
  re-implement that check, because two verifiers eventually disagree;
- a verdict of ACCEPTED over all twelve work packages. A single run cannot
  produce one: R1-09 and R1-12 are settled across two environments, so the
  baseline is the combined verdict from `tools/r1/compare_envs.py`;
- an R1 commit that is this candidate or a recorded ancestor of it;
- profile fingerprints, the toolchain lock, the normalisation rules and the
  case inventory — the things R2 binds artifacts to. An artifact bound to a
  fingerprint nobody recorded is an artifact bound to a guess.
