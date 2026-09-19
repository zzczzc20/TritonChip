//===- TNCADialect.td - The TNCA dialect --------------------*- tablegen -*-===//
//
// doc/compiler_overview.md sec 5 fixes the shape of this dialect, and most of
// that section is about what NOT to build:
//
//   sec 5.1  Value semantics stay with tt/arith/math/scf. TNCA adds resource
//            and asynchronous execution semantics and nothing else, so an
//            elementwise add is never a tnca op.
//   sec 5.2  One dialect, three abstraction layers -- not three large
//            dialects.
//   sec 5.4  Ownership must not be an annotation. Three mechanisms together:
//            SSA event/lease values, MemoryEffectsOpInterface, and a
//            dedicated asynchronous-resource verifier.
//
// The channel in sec 5.2 is emphatically not a new level of storage: TNCA has
// no software-addressable scratchpad, so a channel names a range of vector
// registers and a protocol over it, never a buffer.
//
//===----------------------------------------------------------------------===//

#ifndef TNCA_DIALECT
#define TNCA_DIALECT

include "mlir/IR/OpBase.td"

def TNCA_Dialect : Dialect {
  let name = "tnca";
  let cppNamespace = "::mlir::tnca";
  let summary = "Resource and asynchronous execution semantics for TNCA";
  let description = [{
    Added on top of the Triton dialects rather than replacing them. A TNCA
    program is still `tt`, `arith`, `math` and `scf` for everything that
    computes a value; `tnca` carries only what those cannot express -- which
    registers an asynchronous engine owns, when it stops reading them, and how
    two Role Contexts hand a range between them.
  }];

  let useDefaultTypePrinterParser = 1;
  let useDefaultAttributePrinterParser = 1;
  let dependentDialects = ["::mlir::arith::ArithDialect"];
}

class TNCA_Op<string mnemonic, list<Trait> traits = []>
    : Op<TNCA_Dialect, mnemonic, traits>;

class TNCA_Type<string name, string typeMnemonic>
    : TypeDef<TNCA_Dialect, name> {
  let mnemonic = typeMnemonic;
}

//===----------------------------------------------------------------------===//
// Types
//===----------------------------------------------------------------------===//

def TNCA_WindowType : TNCA_Type<"Window", "window"> {
  let summary = "A contiguous range of vector registers";
  let description = [{
    ISA sec 9D.2: a window names vector registers and nothing else. It has no
    byte pointer, no storage of its own, and no general
    `load [window + offset]`. Without those restrictions it would be a
    scratchpad under a new name, and removing the scratchpad is the point of
    the architecture.

    It is also the transfer engine's bounds check: a transfer past `len`
    registers raises DESCRIPTOR_FAULT rather than corrupting a neighbour.
  }];
  let parameters = (ins "unsigned":$length);
  let assemblyFormat = "`<` $length `>`";
}

def TNCA_AccType : TNCA_Type<"Acc", "acc"> {
  let summary = "An accumulator slot";
}

def TNCA_DescType : TNCA_Type<"Desc", "desc"> {
  let summary = "A versioned snapshot of a 256-byte TensorMap";
  let description = [{
    A snapshot, not a pointer: ISA sec 9D.1 gives descriptors and windows
    different lifetimes on purpose, and the descriptor carries a version so a
    transfer cannot silently pick up a rewritten shape.
  }];
}

def TNCA_EventType : TNCA_Type<"Event", "event"> {
  let summary = "The completion of one asynchronous command";
  let description = [{
    An SSA value, which is the whole point (sec 5.4): a completion that exists
    only as an attribute is invisible to every general MLIR transform, so
    scheduling and CSE may reorder across it. As a value, the dependence is
    one every pass already respects.

    ISA sec 9B counts one complete asynchronous command as one operation, so
    one event is one counted operation -- not one source-level transfer, which
    may expand into several (sec 8.2).
  }];
}

def TNCA_LeaseType : TNCA_Type<"Lease", "lease"> {
  let summary = "An engine's hold on a register range";
  let description = [{
    Separate from the event because the two ends of a command are different
    moments: when the destination registers carry the result, and when the
    engine has stopped reading the sources. ISA v0.1 fuses them into one
    completion, which is the conservative bound; sec 10.2 asks for the source
    release to be separated, and modelling them apart now means the scheduler
    tightens rather than changes when hardware provides it.
  }];
}

def TNCA_ChannelType : TNCA_Type<"Channel", "channel"> {
  let summary = "A producer/consumer protocol over a set of windows";
  let description = [{
    sec 5.2: not a new storage hierarchy. A channel is `depth` windows plus
    the barriers and the ownership discipline that move a range from one Role
    Context to another. There is no memory behind it.
  }];
  let parameters = (ins "unsigned":$depth, "unsigned":$numConsumers);
  let assemblyFormat = "`<` $depth `,` $numConsumers `>`";
}

#endif // TNCA_DIALECT
