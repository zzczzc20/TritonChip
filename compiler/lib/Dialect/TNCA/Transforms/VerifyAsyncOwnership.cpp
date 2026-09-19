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

/// Which window values an asynchronous op holds, and how.
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
    return;
  }
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

    for (Region &r : region->getRegions()) {
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

          // Any operand of a non-async op that names a window a command still
          // writes is a read of data that is not there yet; any window a
          // command still reads must not be written.
          if (touches.empty()) {
            for (Value operand : op.getOperands()) {
              if (!isa<WindowType>(operand.getType()))
                continue;
              if (Operation *w = writer.lookup(operand)) {
                op.emitError()
                    << "reads a window that command "
                    << w->getName().getStringRef()
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
