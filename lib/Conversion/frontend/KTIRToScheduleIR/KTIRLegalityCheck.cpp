//===-- KTIRLegalityCheck.cpp -----------------------------------*- c++ -*-===//
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
// Rejects KTIR constructs that V1 cannot lower yet (fail-fast legality gate).
//
//===----------------------------------------------------------------------===//

#include "dataflow-scheduler/Conversion/frontend/KTIRToScheduleIR/Passes.h"
#include "dataflow-scheduler/Dialect/KTDF/KTDFDialect.h"
#include "ktir/Dialect/KTDP/KTDP.h"
#include "ktir/Dialect/SpyreOp/SpyreOpDialect.h"
#include "llvm/Support/DebugLog.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/Pass/Pass.h"

#define PASS_NAME "ktir-legality-check"
#define DEBUG_TYPE PASS_NAME

static llvm::cl::opt<bool> DisableThisPass(
    "disable-" PASS_NAME, llvm::cl::desc("Disable KTIR Legality Check pass"),
    llvm::cl::init(false));

using namespace scheduler;

namespace scheduler {
#define GEN_PASS_DEF_KTIRLEGALITYCHECKPASS
#include "dataflow-scheduler/Conversion/frontend/KTIRToScheduleIR/Passes.h.inc"
}  // namespace scheduler

namespace {

/// Determines whether @p op is legal within the body of a 'linalg.generic'.
[[nodiscard]] auto isLegalGenericBodyOp(mlir::Operation* op) -> bool {
  // Accept supported arith operations and the 'linalg.yield' terminator.
  if (mlir::isa<mlir::arith::AddFOp, mlir::arith::MulFOp, mlir::arith::SubFOp,
                mlir::arith::AddIOp, mlir::arith::MaximumFOp,
                mlir::arith::MinimumFOp, mlir::arith::MaxNumFOp,
                mlir::arith::CmpFOp, mlir::arith::SelectOp, mlir::math::AbsFOp,
                mlir::linalg::YieldOp>(op)) {
    return true;
  }

  // Accept a constant the body computes against: materialize-registers puts it
  // in the register its maps_to names, so it is a value read, not an operation.
  if (mlir::isa<mlir::arith::ConstantOp>(op)) {
    return true;
  }

  // Accept 'spyreop' intrinsics.
  if (mlir::isa<mlir::spyreop::SpyreOpDialect>(op->getDialect())) {
    return true;
  }

  // Accept an index cast. A kernel holds a runtime scalar as index and casts it
  // where it is used -- the base and the stride of an address computation, say.
  if (mlir::isa<mlir::arith::IndexCastUIOp>(op)) {
    return true;
  }

  // Reject everything else.
  return false;
}

struct KTIRLegalityCheckPass
    : public impl::KTIRLegalityCheckPassBase<KTIRLegalityCheckPass> {
  void runOnOperation() final {
    if (DisableThisPass) return;
    LDBG(1) << "========= " PASS_NAME " =========";
    mlir::ModuleOp module = getOperation();
    bool failed = false;

    module.walk<mlir::WalkOrder::PreOrder>([&](mlir::Operation* op)
                                               -> mlir::WalkResult {
      // Rule 1: loop-carried control flow.
      if (auto forOp = mlir::dyn_cast<mlir::scf::ForOp>(op)) {
        if (forOp.getNumRegionIterArgs() > 0) {
          forOp.emitError(
              "V1 does not support scf.for with loop-carried arguments "
              "(iter_args)");
          failed = true;
          return mlir::WalkResult::interrupt();
        }
        return mlir::WalkResult::advance();
      }
      if (auto whileOp = mlir::dyn_cast<mlir::scf::WhileOp>(op)) {
        whileOp.emitError("V1 does not support scf.while loops");
        failed = true;
        return mlir::WalkResult::interrupt();
      }

      // Rule 3 & 4: unsupported ktdp construct ops.
      if (mlir::isa<mlir::ktdp::ConstructDistributedMemoryViewOp>(op)) {
        op->emitError(
            "V1 does not support ktdp.construct_distributed_memory_view");
        failed = true;
        return mlir::WalkResult::interrupt();
      }
      if (mlir::isa<mlir::ktdp::ConstructIndirectAccessTilesOp>(op)) {
        op->emitError(
            "V1 does not support ktdp.construct_indirect_access_tile");
        failed = true;
        return mlir::WalkResult::interrupt();
      }

      // Rule 2: compute ops.
      // 2a: named linalg ops (anything in the linalg dialect that is not a
      // generic/yield). Allow add/mul/sub/reduce; reject other named ops.
      if (op->getDialect() ==
          op->getContext()->getLoadedDialect<mlir::linalg::LinalgDialect>()) {
        if (mlir::isa<mlir::linalg::GenericOp>(op)) {
          // Inspect the generic body: every op must be a legal body op.
          mlir::WalkResult bodyResult = op->getRegion(0).walk(
              [&](mlir::Operation* inner) -> mlir::WalkResult {
                if (!isLegalGenericBodyOp(inner)) {
                  inner->emitError(
                      "V1 only supports add/mul/sub compute ops; found "
                      "unsupported compute op");
                  return mlir::WalkResult::interrupt();
                }
                return mlir::WalkResult::advance();
              });
          if (bodyResult.wasInterrupted()) {
            failed = true;
            return mlir::WalkResult::interrupt();
          }
          // skip descending again into the body (already inspected)
          return mlir::WalkResult::skip();
        }
        if (mlir::isa<mlir::linalg::AddOp, mlir::linalg::MulOp,
                      mlir::linalg::SubOp, mlir::linalg::MaxOp,
                      mlir::linalg::MinOp, mlir::linalg::YieldOp>(op)) {
          return mlir::WalkResult::advance();
        }
        if (mlir::isa<mlir::linalg::ReduceOp>(op)) {
          // Inspect the reduce body: every op must be a legal body op.
          mlir::WalkResult bodyResult = op->getRegion(0).walk(
              [&](mlir::Operation* inner) -> mlir::WalkResult {
                if (!isLegalGenericBodyOp(inner)) {
                  inner->emitError(
                      "V1 only supports add/mul/sub compute ops; found "
                      "unsupported compute op");
                  return mlir::WalkResult::interrupt();
                }
                return mlir::WalkResult::advance();
              });
          if (bodyResult.wasInterrupted()) {
            failed = true;
            return mlir::WalkResult::interrupt();
          }
          return mlir::WalkResult::skip();
        }
        // Any other named linalg op is unsupported.
        op->emitError(
            "V1 only supports add/mul/sub/reduce compute ops; found "
            "unsupported compute op");
        failed = true;
        return mlir::WalkResult::interrupt();
      }

      return mlir::WalkResult::advance();
    });

    if (failed) signalPassFailure();
  }
};

}  // namespace

std::unique_ptr<mlir::Pass> scheduler::createKTIRLegalityCheckPass() {
  return std::make_unique<KTIRLegalityCheckPass>();
}

// Made with Bob
