//===- TNCAInterfaces.td ------------------------------------*- tablegen -*-===//
//
// sec 5.4 adds these two interfaces, and is explicit about why
// MemoryEffectsOpInterface alone is not enough: MLIR's own documentation says
// the side-effect interface is not a fine-grained address-range or alias
// mechanism, so `MemRead`/`MemWrite` will not by itself solve asynchronous RF
// lifetimes. The effects interface stops wrong deletion, duplication and
// motion; these two carry the facts a TNCA-specific verifier needs.
//
//===----------------------------------------------------------------------===//

#ifndef TNCA_INTERFACES
#define TNCA_INTERFACES

include "mlir/IR/OpBase.td"

def TNCAAsyncOpInterface : OpInterface<"TNCAAsyncOpInterface"> {
  let cppNamespace = "::mlir::tnca";
  let description = [{
    An operation an engine executes after the issuing Role Context has moved
    on. The footprints are register ranges, not addresses: the question that
    matters is which physical vector registers may not be rewritten yet.
  }];

  let methods = [
    InterfaceMethod<[{
      Register ranges the engine reads. Live until the source-release
      condition, which may be later than the issuing instruction's retirement.
    }], "::llvm::SmallVector<::mlir::tnca::RFFootprint>", "getReadFootprints">,

    InterfaceMethod<[{
      Register ranges the engine writes. Reading these before the completion
      event yields whatever the engine has reached so far -- which is why the
      event is an SSA value and not a comment.
    }], "::llvm::SmallVector<::mlir::tnca::RFFootprint>", "getWriteFootprints">,

    InterfaceMethod<[{
      Which completions this command raises. sec 8.2: the count is taken after
      the command expands, so an op that becomes several engine commands
      reports several.
    }], "::llvm::SmallVector<::mlir::tnca::CompletionKind>", "getCompletionKinds">,

    InterfaceMethod<[{
      When the sources may be overwritten. On ISA v0.1 this is
      `OnCommandCompletion` for every command; a scheduler must take that
      bound rather than assume the looser one sec 10.2 asks the hardware for.
    }], "::mlir::tnca::SourceReleaseCondition", "getSourceReleaseCondition">,
  ];
}

def TNCAExecutionResourceInterface : OpInterface<"TNCAExecutionResourceInterface"> {
  let cppNamespace = "::mlir::tnca";
  let description = [{
    What an operation costs on the units Role Contexts share. sec 10.4: the
    cost model must accumulate per shared execution resource, because two
    roles that look independent in the IR contend for one SIMD pipe, one
    transfer engine and one set of RF ports.
  }];

  let methods = [
    InterfaceMethod<"The unit that executes this.",
                    "::mlir::tnca::ExecutionUnit", "getExecutionUnit">,
    InterfaceMethod<"Issue slots consumed.", "unsigned", "getIssueDemand">,
    InterfaceMethod<"Register-file port cycles consumed.",
                    "unsigned", "getRFPortDemand">,
    InterfaceMethod<"Engine command slots occupied, and for how long.",
                    "unsigned", "getCommandSlotDemand">,
  ];
}

#endif // TNCA_INTERFACES
