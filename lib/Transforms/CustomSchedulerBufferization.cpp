//===----------------------------------------------------------------------===//
//
// Part of the Dataflow Scheduler project.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
//===----------------------------------------------------------------------===//
//
// CustomSchedulerBufferization: rebuild a linalg.generic that reads a
// bufferization.to_tensor so it operates on buffers, and drop the to_tensor.
//
// A device pattern that materializes a buffer has to hand it back as a tensor,
// because PDL replaces a value with one of the same type and cannot retype the
// consumer's operands or drop its results.  This pass finishes what the pattern
// could not express.
//
//===----------------------------------------------------------------------===//

#include <memory>

#include "dataflow-scheduler/Analysis/ArchViews/GroupLocalMemory.h"
#include "dataflow-scheduler/Conversion/Utils/Utils.h"
#include "dataflow-scheduler/Dialect/KTDF/KTDF.h"
#include "dataflow-scheduler/Dialect/KTDF/Utils/Utils.h"
#include "dataflow-scheduler/Dialect/KTDFArch/Analysis/DeviceManager.h"
#include "dataflow-scheduler/Transforms/Passes.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/DebugLog.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Pass/Pass.h"

#define PASS_NAME "custom-scheduler-bufferization"
#define DEBUG_TYPE PASS_NAME

static llvm::cl::opt<bool> DisableThisPass(
    "disable-" PASS_NAME,
    llvm::cl::desc("Disable the Custom Scheduler Bufferization pass"),
    llvm::cl::init(false));

namespace scheduler {
#define GEN_PASS_DEF_CUSTOMSCHEDULERBUFFERIZATIONPASS
#include "dataflow-scheduler/Transforms/Passes.h.inc"
}  // namespace scheduler

namespace {

/// Absorb one `bufferization.to_tensor` into the `linalg.generic` that reads
/// it, so the generic runs on buffers and the tensor view can go.
///
/// Returns failure with a diagnostic when an operand has no buffer to name.
static mlir::LogicalResult absorbIntoConsumer(
    mlir::bufferization::ToTensorOp to_tensor,
    const scheduler::arch_view::GroupLocalMemory& group_local_mem) {
  mlir::Value as_tensor = to_tensor.getResult();

  // The generic that reads the tensor view. Anything else keeps the view alive,
  // and the caller leaves it in place.
  auto users = as_tensor.getUsers();
  auto generic_it = llvm::find_if(users, [](mlir::Operation* user) {
    return mlir::isa<mlir::linalg::GenericOp>(user);
  });
  if (generic_it == users.end()) return mlir::success();
  auto generic_op = mlir::cast<mlir::linalg::GenericOp>(*generic_it);

  LDBG(1) << "absorbing " << to_tensor << " into " << generic_op;

  mlir::OpBuilder builder(generic_op);

  // A buffer for every operand the generic still holds as a tensor.  It has to
  // be all of them: linalg rejects an op that mixes tensor and buffer
  // semantics, so one tensor left behind would make the rebuilt op invalid.
  //
  // The results get a fresh allocation.  A sibling input arrives over a FIFO,
  // and is re-read as a memref -- that read has no address to load from, but
  // the lowering layer turns a memref read into a receive yielding the value
  // directly, which is what the rebuilt op ends up consuming.
  //
  // Whatever produced those tensors is about to be left with nothing reading
  // it, so note it down here: converting erases ops and would leave a handle
  // collected afterwards dangling.
  llvm::DenseMap<mlir::Value, mlir::Value> buffers;
  llvm::SmallVector<mlir::Operation*> maybe_dead;
  llvm::DenseSet<mlir::Operation*> seen_producers;
  auto noteProducer = [&](mlir::Value tensor) {
    if (mlir::Operation* producer = tensor.getDefiningOp())
      if (seen_producers.insert(producer).second)
        maybe_dead.push_back(producer);
  };

  for (mlir::Value input : generic_op.getInputs()) {
    if (input == as_tensor) continue;
    if (!mlir::isa<mlir::RankedTensorType>(input.getType())) continue;

    auto read_op = input.getDefiningOp<mlir::ktdf::ReadFromFifoOp>();
    if (!read_op)
      return generic_op.emitError(
          "cannot absorb bufferization.to_tensor: a tensor input of the "
          "consuming linalg.generic is not a ktdf.read_from_fifo, so there is "
          "no buffer to give it");
    buffers[input] = mlir::ktdf::convertFromTensorToMemref(builder, read_op);
    noteProducer(input);
  }

  // The results land in the memory local to the unit that computes them, the
  // same place a reduction accumulator goes.  Naming it is what lets address
  // assignment place the buffer; one with no address reaches the backend as a
  // bare allocation it cannot read.
  auto stage = generic_op->getParentOfType<mlir::ktdf::StageOp>();
  assert(stage && "a linalg.generic is always inside a ktdf.stage");
  mlir::Attribute mem_space = group_local_mem.getLocalMemoryKindForStage(stage);
  if (!mem_space)
    return generic_op.emitError(
        "cannot absorb bufferization.to_tensor: no unambiguous local memory "
        "for this stage's unit, so its results have nowhere to live");

  for (auto [result, output] :
       llvm::zip(generic_op.getResults(), generic_op.getOutputs())) {
    mlir::MemRefType memref_type =
        scheduler::memRefTypeFor(result.getType(), mem_space);
    if (!memref_type)
      return generic_op.emitError(
          "cannot absorb bufferization.to_tensor: a result of the consuming "
          "linalg.generic has no static shape to allocate a buffer for");
    buffers[output] =
        mlir::memref::AllocOp::create(builder, generic_op.getLoc(), memref_type)
            .getResult();
    noteProducer(output);
  }

  // Rebuilding the generic and re-pointing whatever read its results is the
  // shared conversion, which fails on any consumer it does not recognise. It
  // also erases the tensor view itself once nothing reads it, so this must not
  // touch `to_tensor` afterwards.
  if (mlir::failed(mlir::ktdf::convertTensorUsesToMemref(
          builder, as_tensor, to_tensor.getBuffer(), buffers)))
    return mlir::failure();

  for (mlir::Operation* op : maybe_dead)
    if (op->use_empty()) op->erase();

  return mlir::success();
}

struct CustomSchedulerBufferizationPass
    : public scheduler::impl::CustomSchedulerBufferizationPassBase<
          CustomSchedulerBufferizationPass> {
  void runOnOperation() override {
    if (DisableThisPass) return;
    LDBG(1) << "========= " PASS_NAME " =========";

    mlir::ModuleOp module_op = getOperation();

    auto& device_manager = getAnalysis<mlir::ktdf_arch::DeviceManager>();
    auto* const device = device_manager.getOrImportDevice();
    if (!device) {
      module_op->emitError(
          "Unable to import the device specification. This could happen if the "
          "device spec file is empty or contains multiple devices");
      signalPassFailure();
      return;
    }
    auto& group_local_mem =
        device_manager.getOrCreateView<scheduler::arch_view::GroupLocalMemory>(
            *device);

    // Snapshot first: absorbing one rebuilds and erases ops around it, which is
    // not safe to do with a walk in flight.
    llvm::SmallVector<mlir::bufferization::ToTensorOp> to_tensors;
    module_op.walk(
        [&](mlir::bufferization::ToTensorOp op) { to_tensors.push_back(op); });

    for (mlir::bufferization::ToTensorOp op : to_tensors)
      if (mlir::failed(absorbIntoConsumer(op, group_local_mem)))
        return signalPassFailure();
  }
};

}  // namespace

std::unique_ptr<mlir::Pass>
scheduler::createCustomSchedulerBufferizationPass() {
  return std::make_unique<CustomSchedulerBufferizationPass>();
}
