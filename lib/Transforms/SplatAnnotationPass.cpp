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
// SplatAnnotation: annotate splat data_transfer ops and their consumers with
// arch-derived shuffle parameters so the KTDFLowToDFIR lowering passes can
// read attributes instead of querying the arch at lowering time.
//
// See docs/passes.md for algorithm documentation.
//
//===----------------------------------------------------------------------===//

#include <memory>

#include "dataflow-scheduler/Conversion/Utils/Utils.h"
#include "dataflow-scheduler/Dialect/KTDF/KTDF.h"
#include "dataflow-scheduler/Dialect/KTDFArch/Analysis/DeviceManager.h"
#include "dataflow-scheduler/Dialect/KTDFArch/Analysis/ResourceKinds.h"
#include "dataflow-scheduler/Dialect/KTDFLowering/KTDFLowering.h"
#include "dataflow-scheduler/Transforms/Passes.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/DebugLog.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

#define PASS_NAME "splat-annotation"
#define DEBUG_TYPE PASS_NAME

static llvm::cl::opt<bool> DisableThisPass(
    "disable-" PASS_NAME, llvm::cl::desc("Disable the Splat Annotation pass"),
    llvm::cl::init(false));

namespace scheduler {
#define GEN_PASS_DEF_SPLATANNOTATIONPASS
#include "dataflow-scheduler/Transforms/Passes.h.inc"
}  // namespace scheduler

namespace {

/// Return the applicable-units kind for a stage, or nullptr.
static mlir::Attribute getStageResourceKind(mlir::ktdf::StageOp stage) {
  auto maybe_units = stage.getApplicableUnits();
  if (!maybe_units || maybe_units->size() != 1) return nullptr;
  return maybe_units->getValue().front();
}

/// Resolve the load unit kind for an op that may live inside either a
/// ktdf.stage (pre-KTDFToKTDFLow IR) or a ktdf_lowering.execute_on
/// (post-KTDFToKTDFLow / direct test IR).
static mlir::Attribute resolveLoadUnitKind(
    mlir::Operation* op, const mlir::ktdf_arch::ResourceKinds& resource_kinds) {
  // Pre-lowering: inside a ktdf.stage with applicable_units.
  if (auto stage = op->getParentOfType<mlir::ktdf::StageOp>())
    return getStageResourceKind(stage);

  // Post-lowering: inside a ktdf_lowering.execute_on with unit operands.
  if (auto exec = op->getParentOfType<mlir::ktdf_lowering::ExecuteOnOp>()) {
    if (!exec.getUnits().empty()) {
      auto kind = scheduler::getUnitResourceType(exec.getUnits().front());
      return kind.value_or(mlir::Attribute{});
    }
  }
  return nullptr;
}

struct SplatAnnotationPass
    : public scheduler::impl::SplatAnnotationPassBase<SplatAnnotationPass> {
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
    auto& resource_kinds =
        device_manager.getOrCreateView<mlir::ktdf_arch::ResourceKinds>(*device);

    // Walk 1 — for each splat data_transfer, widen the last source size
    // dimension to the arch-aligned load width when needed, then mark every
    // consuming read_from_fifo with read_with_splat = true.
    mlir::WalkResult walk_result =
        module_op.walk([&](mlir::ktdf::DataTransferOp data_transfer) {
          if (!data_transfer.isDestFifo()) return mlir::WalkResult::advance();

          auto mode_attr = data_transfer->getDiscardableAttr("transfer_mode");
          if (!mode_attr) return mlir::WalkResult::advance();
          if (llvm::cast<mlir::StringAttr>(mode_attr).getValue() != "splat")
            return mlir::WalkResult::advance();
          // Resolve the load unit kind from the enclosing stage or execute_on.
          mlir::Attribute load_unit_kind =
              resolveLoadUnitKind(data_transfer, resource_kinds);

          // Validate that the load unit declares splat capability.
          auto simd_feature =
              resource_kinds.getFeature<mlir::ktdf_arch::feature::SIMD>(
                  load_unit_kind);
          if (!simd_feature || !simd_feature.canSplat()) {
            data_transfer.emitError(
                "splat data_transfer requires the load unit to declare "
                "ktdf_arch.feature.simd = { splat, ... }");
            return mlir::WalkResult::interrupt();
          }

          // Compute the effective load width from the arch's granularity.
          auto fifo_slot_type = mlir::cast<mlir::ktdf::FifoSlotType>(
              data_transfer.getDestination().getType());
          mlir::Type elem_type = fifo_slot_type.getElementType();
          auto static_src_sizes =
              data_transfer.getStaticSourceSizesArray().value();
          int64_t src_elements = 1;
          for (int64_t s : static_src_sizes)
            src_elements *= s;
          int64_t load_elements = scheduler::computeSplatGranularityElements(
              src_elements, elem_type, load_unit_kind, resource_kinds);

          // When widening actually occurs, update the last source dimension so
          // the lowering pass picks up the correct load width directly from the
          // size operand (no extra attribute needed).
          if (load_elements > src_elements) {
            llvm::SmallVector<int64_t> new_sizes(static_src_sizes.begin(),
                                                 static_src_sizes.end());
            new_sizes.back() = load_elements;
            data_transfer.setStaticSourceSizes(
                llvm::ArrayRef<int64_t>(new_sizes));
          }

          // When widening actually occurs, mark every consuming read_from_fifo.
          if (load_elements <= src_elements) return mlir::WalkResult::advance();

          mlir::Value fifo_slot = data_transfer.getDestination();
          for (mlir::Operation* user : fifo_slot.getUsers()) {
            auto read_op = mlir::dyn_cast<mlir::ktdf::ReadFromFifoOp>(user);
            if (read_op && read_op.getFifoSlot() == fifo_slot) {
              read_op->setDiscardableAttr(
                  "read_with_splat",
                  mlir::BoolAttr::get(read_op->getContext(), true));
            }
          }
          return mlir::WalkResult::advance();
        });

    if (walk_result.wasInterrupted()) {
      signalPassFailure();
      return;
    }

    // Walk 2 — for each read_from_fifo annotated with read_with_splat, derive
    // the sub-SIMD granularity and register-file kind from the arch and
    // annotate the op with splat_granularity_elements + splat_register_kind.
    walk_result = module_op.walk([&](mlir::ktdf::ReadFromFifoOp read_op) {
      auto splat_attr = read_op->getDiscardableAttr("read_with_splat");
      if (!splat_attr || !mlir::cast<mlir::BoolAttr>(splat_attr).getValue())
        return mlir::WalkResult::advance();

      // Resolve the compute unit kind from the enclosing stage or execute_on.
      mlir::Attribute compute_kind =
          resolveLoadUnitKind(read_op, resource_kinds);

      auto fifo_slot_type =
          mlir::cast<mlir::ktdf::FifoSlotType>(read_op.getFifoSlot().getType());
      mlir::Type elem_type = fifo_slot_type.getElementType();
      int64_t fifo_elements = fifo_slot_type.getNumElements();

      auto sub_simd_or_err = scheduler::computeSplatSubSimdElements(
          fifo_elements, elem_type, compute_kind, resource_kinds,
          read_op.getOperation());
      if (mlir::failed(sub_simd_or_err)) return mlir::WalkResult::interrupt();

      // Derive the register-file kind co-located with the compute unit in the
      // arch graph.  This is needed by the receive-side lowering to materialise
      // the shuffle result through the register file.
      mlir::Attribute reg_kind =
          scheduler::getComputeRegisterKind(compute_kind, resource_kinds);
      if (!reg_kind) {
        read_op->emitError(
            "splat receive: no register-file memory found in the same "
            "arch group as the compute unit");
        return mlir::WalkResult::interrupt();
      }

      read_op->setDiscardableAttr(
          "splat_granularity_elements",
          mlir::IntegerAttr::get(
              mlir::IntegerType::get(read_op->getContext(), 64),
              sub_simd_or_err.value()));
      read_op->setDiscardableAttr("splat_register_kind", reg_kind);
      return mlir::WalkResult::advance();
    });

    if (walk_result.wasInterrupted()) signalPassFailure();
  }
};

}  // namespace

std::unique_ptr<mlir::Pass> scheduler::createSplatAnnotationPass() {
  return std::make_unique<SplatAnnotationPass>();
}
