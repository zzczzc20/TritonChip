//===- VerifyAsyncOwnership.cpp -------------------------------------------===//
//
// The asynchronous-resource verifier of compiler_overview.md sec 5.4.
//
// MLIR's side-effect interface is explicitly not a fine-grained address-range
// or alias mechanism -- its own documentation says so -- which is why marking
// an operation MemRead does not answer the question that matters here: *which*
// registers is the engine still reading, and until when. Before register
// allocation the answer is expressed over SSA values; the same check runs over
// concrete ranges afterwards, using the footprints in TNCAAsyncOps.cpp.
//
// Two rules, both from the ISA rather than invented here:
//
//   ISA sec 9C.3  the source registers of an incomplete m.mma.async may not
//                 be overwritten, and its accumulator may not be read by the
//                 vector unit or reassigned.
//
//   sec 10.2      an asynchronous lifetime ends at a completion event. A
//                 command whose event is never awaited owns its registers
//                 forever, so they can never be safely reused -- which is a
//                 resource leak the allocator cannot see.
//
//===----------------------------------------------------------------------===//

#include "tnca/Dialect/TNCA/Transforms/Passes.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringSet.h"
#include "mlir/Interfaces/FunctionInterfaces.h"

#include "tnca/Dialect/TNCA/IR/TNCADialect.h"

using namespace mlir;
using namespace mlir::tnca;

namespace {

/// Describes what an operation does to a window value.
enum class Touch { Reads, Writes };

struct VerifyAsyncOwnershipPass
    : public PassWrapper<VerifyAsyncOwnershipPass, OperationPass<>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(VerifyAsyncOwnershipPass)

  StringRef getArgument() const final { return "tnca-verify-async"; }

  StringRef getDescription() const final {
    return "check that no asynchronous command's registers are disturbed "
           "while it still owns them";
  }

  void runOnOperation() override;
};

/// Values this pass tracks ownership of. Windows were the only ones for a
/// while, which meant ISA sec 9C.3's second clause -- the accumulator an
/// outstanding m.mma.async writes may not be read by the vector unit -- was
/// stated in the ISA and checked nowhere.
static bool isTrackedResource(Value v) {
  return isa<WindowType>(v.getType()) || isa<AccType>(v.getType());
}

/// Which resource values an asynchronous op holds, and how.
static void collectTouches(Operation *op,
                           SmallVectorImpl<std::pair<Value, Touch>> &out) {
  if (auto load = dyn_cast<DMALoadOp>(op)) {
    out.emplace_back(load.getDest(), Touch::Writes);
    return;
  }
  if (auto store = dyn_cast<DMAStoreOp>(op)) {
    // The late read of sec 13.4: the engine is still reading this window
    // after the issuing instruction has retired.
    out.emplace_back(store.getSrc(), Touch::Reads);
    return;
  }
  if (auto mma = dyn_cast<MMAAsyncOp>(op)) {
    out.emplace_back(mma.getA(), Touch::Reads);
    out.emplace_back(mma.getB(), Touch::Reads);
    // The matrix unit WRITES the accumulator for as long as the command runs.
    // Two commands accumulating into the same register in sequence is the
    // ordinary K loop and stays legal; what is not is reading it meanwhile,
    // and that is what tracking it as a written range makes checkable.
    out.emplace_back(mma.getAcc(), Touch::Writes);
    return;
  }
}

//===----------------------------------------------------------------------===//
// The support boundary
//
// This verifier reasons within one block. That is a real limit, and the
// contract (R1-07, IR-02) requires it to be a *stated* one: an unimplemented
// structure must be refused in so many words, never accepted unchecked. Two
// of these used to be silent. An event produced in one block and awaited in
// another left the first block reporting "never awaited" -- the right verdict
// for the wrong reason -- while the second block retired a command it had
// never seen. And `await` on an event that arrives as a block argument hit
// `ev.getDefiningOp()` returning null and simply moved on, so the wait was
// neither honoured nor refused.
//
// Freezing the boundary is not deleting the capability. When general CFG
// support arrives these checks come out; until then a program outside the
// subset gets a diagnostic instead of an answer nobody verified.
//===----------------------------------------------------------------------===//

/// Operations this pass understands. A TNCA operation that is neither here
/// nor an async command is an unknown resource effect: refusing it is what
/// stops a newly added op from being silently assumed inert (IR-05).
static const llvm::StringSet<> &knownInertOps() {
  static llvm::StringSet<> ops = {
      "tnca.rf.reserve",   "tnca.rf.view",           "tnca.acc.reserve",
      "tnca.acc.read",     "tnca.desc.snapshot",     "tnca.await",
      "tnca.use_complete", "tnca.prefetch",          "tnca.channel.create",
      "tnca.producer.acquire", "tnca.producer.publish",
      "tnca.consumer.acquire", "tnca.consumer.release",
  };
  return ops;
}

/// Checks the structural subset. Returns true when something unsupported was
/// found, in which case the caller must NOT go on to "verify" the block.
static bool checkSupportBoundary(Region &r, Operation *owner) {
  bool bad = false;
  if (!r.hasOneBlock()) {
    owner->emitError()
        << "unsupported: this region has " << r.getBlocks().size()
        << " blocks, and the asynchronous-resource verifier reasons within "
           "one block. A lease that crosses a branch is neither checked nor "
           "rejected today, so it is rejected explicitly "
           "(compiler_overview.md sec 5.4; R1-07 IR-02)";
    return true;
  }

  Block &block = r.front();
  // Only events and leases. A window or accumulator arriving as an argument
  // is an ordinary resource with no command attached; an event is a claim
  // that some command is outstanding, and in this block there is nothing to
  // attach that claim to.
  for (BlockArgument arg : block.getArguments()) {
    if (isa<EventType>(arg.getType()) || isa<LeaseType>(arg.getType())) {
      owner->emitError()
          << "unsupported: a completion event or lease arrives as a block "
             "argument. Its producing command is not in this block, so "
             "nothing here can say what it still owns";
      bad = true;
    }
  }

  for (Operation &op : block) {
    if (op.getNumRegions() != 0 && !isa<SpecializeOp>(&op)) {
      op.emitError()
          << "unsupported: a nested region. The commands inside it and the "
             "awaits outside it are checked separately, which is not a check "
             "at all (R1-07 IR-02)";
      bad = true;
    }
    if (auto await = dyn_cast<AwaitOp>(&op)) {
      for (Value ev : await.getEvents()) {
        if (!ev.getDefiningOp()) {
          op.emitError()
              << "unsupported: awaits an event that is not produced in this "
                 "block. This used to be ignored in silence, which retired "
                 "nothing and reported nothing";
          bad = true;
        }
      }
    }
    StringRef name = op.getName().getStringRef();
    if (name.starts_with("tnca.")) {
      bool known = knownInertOps().contains(name) ||
                   isa<TNCAAsyncOpInterface>(&op) || isa<SpecializeOp>(&op);
      if (!known) {
        op.emitError()
            << "unsupported: unknown resource effect. This operation is in "
               "the TNCA dialect, implements no asynchronous-resource "
               "interface, and is not on the inert list, so what it does to "
               "a window or accumulator is unknown. An unknown effect is "
               "refused, never assumed to be none (R1-07 IR-05)";
        bad = true;
      }
    } else {
      // A foreign operation that takes a TNCA resource is the same question
      // from the other side: nothing here knows whether it reads it, writes
      // it, or keeps it past its lifetime.
      for (Value v : op.getOperands()) {
        if (isTrackedResource(v)) {
          op.emitError()
              << "unsupported: operation outside the TNCA dialect takes a "
                 "window or accumulator operand, and its effect on that "
                 "resource is not described (R1-07 IR-05)";
          bad = true;
          break;
        }
      }
    }
  }
  return bad;
}

/// The completion event an asynchronous op produces, if any.
static Value getEvent(Operation *op) {
  if (auto load = dyn_cast<DMALoadOp>(op))
    return load.getEvent();
  if (auto store = dyn_cast<DMAStoreOp>(op))
    return store.getEvent();
  if (auto mma = dyn_cast<MMAAsyncOp>(op))
    return mma.getEvent();
  return nullptr;
}

void VerifyAsyncOwnershipPass::runOnOperation() {
  Operation *root = getOperation();
  bool failed = false;

  root->walk([&](Operation *region) {
    if (region->getNumRegions() == 0)
      return;
    // Function bodies and role regions, not every op that happens to have a
    // region -- a module's region holds functions, and calling those "nested
    // regions" would refuse every well-formed program.
    if (!isa<FunctionOpInterface>(region) && !isa<SpecializeOp>(region))
      return;

    for (Region &r : region->getRegions()) {
      if (r.empty())
        continue;
      if (checkSupportBoundary(r, region)) {
        failed = true;
        // Deliberately not falling through into the block walk: reporting
        // ownership verdicts about a structure this pass does not model is
        // how "no errors" comes to mean "nothing was looked at".
        continue;
      }
      for (Block &block : r) {
        // Window value -> the commands currently holding it.
        DenseMap<Value, SmallVector<Operation *, 2>> readers;
        DenseMap<Value, Operation *> writer;
        DenseSet<Operation *> outstanding;

        for (Operation &op : block) {
          // An await retires every command whose event it takes.
          if (auto await = dyn_cast<AwaitOp>(&op)) {
            for (Value ev : await.getEvents()) {
              Operation *producer = ev.getDefiningOp();
              if (!producer)
                continue;
              outstanding.erase(producer);
              for (auto &entry : readers)
                llvm::erase(entry.second, producer);
              // DenseMap::erase returns void, so collect first rather than
              // erasing through the iterator.
              SmallVector<Value, 2> retired;
              for (auto &entry : writer)
                if (entry.second == producer)
                  retired.push_back(entry.first);
              for (Value v : retired)
                writer.erase(v);
            }
            continue;
          }

          SmallVector<std::pair<Value, Touch>, 3> touches;
          collectTouches(&op, touches);

          // Any operand of a non-async op that names a resource a command
          // still writes is a read of data that is not there yet; any
          // resource a command still reads must not be written.
          if (touches.empty()) {
            for (Value operand : op.getOperands()) {
              if (!isTrackedResource(operand))
                continue;
              if (Operation *w = writer.lookup(operand)) {
                op.emitError()
                    << "reads a "
                    << (isa<AccType>(operand.getType()) ? "accumulator"
                                                        : "window")
                    << " that command " << w->getName().getStringRef()
                    << " has not finished writing; its completion event must "
                       "be awaited first (compiler_overview.md sec 10.2)";
                failed = true;
              }
            }
            continue;
          }

          for (auto [value, touch] : touches) {
            if (touch == Touch::Writes) {
              if (!readers.lookup(value).empty()) {
                op.emitError()
                    << "writes a window that an outstanding command is still "
                       "reading; ISA sec 9C.3 leaves that undefined and "
                       "assigns it to the compiler";
                failed = true;
              }
              if (Operation *w = writer.lookup(value)) {
                (void)w;
                op.emitError()
                    << "writes a window another outstanding command is "
                       "already writing";
                failed = true;
              }
              writer[value] = &op;
            } else {
              if (Operation *w = writer.lookup(value)) {
                (void)w;
                op.emitError()
                    << "reads a window an outstanding command is still "
                       "writing; the data it would see is not there yet";
                failed = true;
              }
              readers[value].push_back(&op);
            }
          }

          if (getEvent(&op))
            outstanding.insert(&op);
        }

        // sec 10.2: a lifetime ends at a completion event. Without one the
        // registers are owned forever.
        for (Operation *op : outstanding) {
          op->emitError()
              << "completion event is never awaited in this block, so its "
                 "registers can never be safely reused "
                 "(compiler_overview.md sec 10.2)";
          failed = true;
        }
      }
    }
  });

  if (failed)
    signalPassFailure();
}

} // namespace

namespace mlir {
namespace tnca {

std::unique_ptr<Pass> createVerifyAsyncOwnershipPass() {
  return std::make_unique<VerifyAsyncOwnershipPass>();
}

void registerTNCAPasses() {
  PassRegistration<VerifyAsyncOwnershipPass>();
}

} // namespace tnca
} // namespace mlir
