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

#include <llvm/Support/CommandLine.h>
#include <llvm/Support/DebugLog.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/Linalg/IR/Linalg.h>
#include <mlir/Dialect/Math/IR/Math.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/PDL/IR/PDL.h>
#include <mlir/Dialect/PDLInterp/IR/PDLInterp.h>
#include <mlir/IR/AffineExpr.h>
#include <mlir/IR/AffineMap.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Support/TypeID.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

#include "dataflow-scheduler/Dialect/Agen/Agen.h"  // IWYU pragma: keep
#include "dataflow-scheduler/Dialect/KTDFArch/KTDFArchDialect.h"  // IWYU pragma: keep
#include "dataflow-scheduler/Dialect/KTDFArch/Transforms/ApplyPatterns.h"
#include "dataflow-scheduler/Dialect/VectorChain/VectorChain.h"  // IWYU pragma: keep
#include "dataflow-scheduler/Transforms/Passes.h"

#define PASS_NAME "apply-device-patterns"
#define DEBUG_TYPE PASS_NAME

static llvm::cl::opt<bool> disable_this_pass(
    "disable-" PASS_NAME, llvm::cl::desc("Disable Device Patterns pass"),
    llvm::cl::init(false));

using namespace scheduler;

namespace scheduler {
#define GEN_PASS_DEF_APPLYDEVICEPATTERNSPASS
#include "dataflow-scheduler/Transforms/Passes.h.inc"
}  // namespace scheduler

namespace {

// Returns success (and no results) when `op` is a linalg.generic whose
// rightmost input map dimension maps to a reduction iterator — i.e. it is the
// inner-dim reduction generic produced by MapReductionPartials — and whose
// single output operand is defined by a memref.subview.
auto ktdfIsInnerDimReduction(mlir::PatternRewriter& /*rewriter*/,
                             mlir::Operation* op) -> mlir::LogicalResult {
  auto generic = llvm::dyn_cast_if_present<mlir::linalg::GenericOp>(op);
  if (!generic || generic.getInputs().empty()) return mlir::failure();

  auto input_type =
      mlir::dyn_cast<mlir::MemRefType>(generic.getInputs().front().getType());
  if (!input_type) return mlir::failure();

  mlir::AffineMap input_map = generic.getIndexingMapsArray().front();
  int64_t last = input_type.getRank() - 1;
  auto dim_expr =
      mlir::dyn_cast<mlir::AffineDimExpr>(input_map.getResult(last));
  if (!dim_expr) return mlir::failure();

  unsigned loop_dim = dim_expr.getPosition();
  if (generic.getIteratorTypesArray()[loop_dim] !=
      mlir::utils::IteratorType::reduction)
    return mlir::failure();

  // The output must be a rank-reducing subview produced by
  // MapReductionPartials.
  if (generic.getOutputs().empty()) return mlir::failure();
  if (!llvm::isa_and_nonnull<mlir::memref::SubViewOp>(
          generic.getOutputs().front().getDefiningOp()))
    return mlir::failure();

  return mlir::success();
}

// Rewrite helper: ktdf.subview_source(value) → Value
// Returns the source operand of the memref.subview that defines `value`.
auto ktdfSubviewSource(mlir::PatternRewriter& /*rewriter*/,
                       mlir::PDLResultList& results,
                       llvm::ArrayRef<mlir::PDLValue> values)
    -> mlir::LogicalResult {
  assert(values.size() == 1);
  auto val = values[0].cast<mlir::Value>();
  auto subview =
      llvm::dyn_cast_if_present<mlir::memref::SubViewOp>(val.getDefiningOp());
  if (!subview) return mlir::failure();
  results.push_back(subview.getSource());
  return mlir::success();
}

// What accumulates `yielded`, as one of the kinds a reduction can be:
//   "sum"    — arith.addf
//   "max"    — arith.maximumf
//   "min"    — arith.minimumf
//   "absmax" — arith.maxnumf over math.absf of each side
// Empty where it is none of them.
auto reductionKindOf(mlir::Value yielded) -> llvm::StringRef {
  mlir::Operation* accumulates = yielded.getDefiningOp();
  if (!accumulates) return {};

  if (mlir::isa<mlir::arith::AddFOp>(accumulates)) return "sum";
  if (mlir::isa<mlir::arith::MaximumFOp>(accumulates)) return "max";
  if (mlir::isa<mlir::arith::MinimumFOp>(accumulates)) return "min";

  auto maxf = mlir::dyn_cast<mlir::arith::MaxNumFOp>(accumulates);
  if (!maxf) return {};
  auto lhs = mlir::dyn_cast_if_present<mlir::math::AbsFOp>(
      maxf.getLhs().getDefiningOp());
  auto rhs = mlir::dyn_cast_if_present<mlir::math::AbsFOp>(
      maxf.getRhs().getDefiningOp());
  if (!lhs || !rhs) return {};
  return "absmax";
}

// Rewrite helper: ktdf.with_precision(op, params) → DictionaryAttr
//
// Adds to \p params the precision a template's `mode` field takes, read off the
// element type \p op accumulates in. Fails for a type no template names, so the
// compute is left alone rather than lowered at the wrong width.
auto ktdfWithPrecision(mlir::PatternRewriter& rewriter,
                       mlir::PDLResultList& results,
                       llvm::ArrayRef<mlir::PDLValue> values)
    -> mlir::LogicalResult {
  assert(values.size() == 2);
  auto written = llvm::dyn_cast_if_present<mlir::DestinationStyleOpInterface>(
      values[0].cast<mlir::Operation*>());
  if (!written || written.getDpsInits().empty()) return mlir::failure();
  auto params = llvm::dyn_cast_if_present<mlir::DictionaryAttr>(
      values[1].cast<mlir::Attribute>());
  if (!params) return mlir::failure();

  auto accumulator =
      llvm::dyn_cast<mlir::ShapedType>(written.getDpsInits().front().getType());
  if (!accumulator) return mlir::failure();

  const mlir::Type element = accumulator.getElementType();
  llvm::StringRef precision;
  if (element.isF16()) {
    precision = "fp16";
  } else if (element.isF32()) {
    precision = "fp32";
  } else {
    return mlir::failure();
  }

  mlir::NamedAttrList named(params);
  named.set("prec", rewriter.getStringAttr(precision));
  results.push_back(named.getDictionary(rewriter.getContext()));
  return mlir::success();
}

// Constraint: ktdf.is_reduction_kind(op, expected_kind)
//
// Expected `values` entries (in order):
//   [0] mlir::Operation*   — the linalg.generic op to inspect
//   [1] mlir::Attribute    — a StringAttr naming the expected reduction kind
//
// Succeeds when `op` is a linalg.generic whose body matches `expected_kind`:
//   "sum"    — single arith.addf
//   "max"    — single arith.maximumf
//   "min"    — single arith.minimumf
//   "absmax" — math.absf(%in), math.absf(%acc), arith.maxnumf(abs1, abs2)
auto ktdfIsReductionKind(mlir::PatternRewriter& /*rewriter*/,
                         mlir::PDLResultList& /*results*/,
                         llvm::ArrayRef<mlir::PDLValue> values)
    -> mlir::LogicalResult {
  assert(values.size() == 2);
  auto generic = llvm::dyn_cast_if_present<mlir::linalg::GenericOp>(
      values[0].cast<mlir::Operation*>());
  if (!generic) return mlir::failure();

  auto expected = llvm::dyn_cast_if_present<mlir::StringAttr>(
      values[1].cast<mlir::Attribute>());
  if (!expected) return mlir::failure();

  mlir::Operation* yield = generic.getRegion().front().getTerminator();
  if (yield->getNumOperands() == 0) return mlir::failure();

  for (mlir::Value yielded : yield->getOperands()) {
    if (reductionKindOf(yielded) != expected.getValue()) return mlir::failure();
  }
  return mlir::success();
}

class PatternCache : public mlir::ktdf_arch::PatternCache {
 public:
  using mlir::ktdf_arch::PatternCache::PatternCache;

  void registerNativeFunctions(mlir::PDLPatternModule& patterns) final {
    mlir::ktdf_arch::PatternCache::registerNativeFunctions(patterns);

    patterns.registerConstraintFunction("ktdf.is_inner_dim_reduction",
                                        ktdfIsInnerDimReduction);
    patterns.registerConstraintFunction("ktdf.is_reduction_kind",
                                        ktdfIsReductionKind);
    patterns.registerRewriteFunction("ktdf.subview_source", ktdfSubviewSource);
    patterns.registerRewriteFunction("ktdf.with_precision", ktdfWithPrecision);
  }

  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(PatternCache)
};

struct ApplyDevicePatternsPass
    : impl::ApplyDevicePatternsPassBase<ApplyDevicePatternsPass> {
  using ApplyDevicePatternsPassBase::ApplyDevicePatternsPassBase;

  void runOnOperation() override {
    if (disable_this_pass) {
      return;
    }

    // Find the device that this function maps to.
    auto declaration =
        mlir::ktdf_arch::findDeviceDeclarationFor(getOperation());
    if (!declaration) {
      return;
    }
    mlir::ktdf_arch::DeviceRef device(declaration, getAnalysisManager());
    LDBG() << "processing "
           << mlir::OpWithFlags(getOperation(),
                                mlir::OpPrintingFlags().skipRegions())
           << " with device " << declaration.getName();

    // Get the (cached) rewrite pattern set. This prevents cloning the PDL
    // module.
    const auto patterns = device.getOrCreateView<PatternCache>().get(
        mlir::ktdf_arch::PatternGroups(llvm::from_range, enabled_groups));

    // Run all the patterns.
    auto changed = false;
    if (failed(applyPatternsGreedily(getOperation(), patterns,
                                     mlir::GreedyRewriteConfig(), &changed))) {
      signalPassFailure();
      return;
    }

    if (!changed) {
      markAllAnalysesPreserved();
    }
  }
};

}  // namespace

auto scheduler::createApplyDevicePatternsPass(
    std::initializer_list<llvm::StringRef> enabled_groups)
    -> std::unique_ptr<mlir::Pass> {
  ApplyDevicePatternsPassOptions options;
  for (auto group : enabled_groups) {
    options.enabled_groups.emplace_back(group.str());
  }
  return createApplyDevicePatternsPass(options);
}
