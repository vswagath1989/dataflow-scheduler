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

#include "dataflow-scheduler/Conversion/backend/ScheduleIRToDFIR/KTDFLowToDFIR/DataTransferLowering.h"

#include "dataflow-scheduler/Conversion/backend/ScheduleIRToDFIR/KTDFLowToDFIR/Utils.h"
#include "dataflow-scheduler/Dialect/Agen/Agen.h"
#include "dataflow-scheduler/Dialect/Dataflow/Dataflow.h"
#include "dataflow-scheduler/Dialect/KTDF/KTDF.h"
#include "dataflow-scheduler/Dialect/KTDFArch/Analysis/ResourceKinds.h"
#include "dataflow-scheduler/Dialect/VectorChain/VectorChain.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Analysis/Presburger/IntegerRelation.h"
#include "mlir/Analysis/Presburger/PresburgerSpace.h"
#include "mlir/Dialect/Affine/Analysis/AffineStructures.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/IntegerSet.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LogicalResult.h"

#define DEBUG_TYPE "ktdflowering-to-dfir"

using namespace scheduler;

namespace {

/// Resolve the resource kind attribute from the first unit of a program_unit.
/// Returns nullptr if the kind cannot be determined.
mlir::Attribute kindFromProgramUnit(
    mlir::dataflow::ProgramUnitOp program_unit) {
  if (!program_unit || program_unit.getUnits().empty()) return nullptr;
  auto get_unit = program_unit.getUnits()
                      .front()
                      .getDefiningOp<mlir::dataflow::GetUnitOp>();
  if (!get_unit) return nullptr;
  auto type_attr = get_unit->getAttrOfType<mlir::StringAttr>("type");
  if (!type_attr) return nullptr;
  return mlir::StringAttr::get(get_unit->getContext(),
                               type_attr.getValue().upper());
}

/// One time dimension of one side of a transfer: which memref dimension it
/// advances, and by how many indices of that dimension per step.
struct TransferTimeStep {
  unsigned memref_dim;
  int64_t index_step;
};

/// How one side of a transfer traverses the AGEN time axis: the extent of each
/// time dimension, slowest-varying first, and what each dimension advances.
/// `extents` becomes `time_set`; `offsets()` becomes the results of that
/// side's `*_time_addr_map`.
struct TransferTimeDims {
  llvm::SmallVector<int64_t> extents;
  llvm::SmallVector<TransferTimeStep> steps;  // parallel to `extents`
  size_t rank = 0;

  /// A traversal of a memref of `rank` dimensions that walks nothing.
  explicit TransferTimeDims(size_t rank) : rank(rank) {}

  /// The offset added to each memref index at time step (d0, ..., dn-1); zero
  /// at every index a time dimension does not advance. Time dimensions are
  /// numbered in the order they were added, which is also why an identity
  /// `time_order` is correct: d0 is the slowest-varying.
  llvm::SmallVector<mlir::AffineExpr> offsets(
      mlir::MLIRContext* context) const {
    llvm::SmallVector<mlir::AffineExpr> result(
        rank, mlir::getAffineConstantExpr(0, context));
    for (auto [time_dim, step] : llvm::enumerate(steps)) {
      result[step.memref_dim] =
          mlir::getAffineConstantExpr(step.index_step, context) *
          mlir::getAffineDimExpr(time_dim, context);
    }
    return result;
  }

  /// Drop time dimension `time_dim`. The remaining dimensions keep their
  /// relative order and are renumbered by `offsets()`.
  void eraseDim(unsigned time_dim) {
    extents.erase(extents.begin() + time_dim);
    steps.erase(steps.begin() + time_dim);
  }
};

/// Describe how `sizes` is traversed over time. Every non-unit dimension
/// except the innermost contributes a time dimension stepping by one; the
/// innermost contributes one stepping by a whole vector, and only when it
/// holds more than one. With a 64-lane vector:
///
///   sizes           extents      offsets           time_set
///   [1, 256, 64]    [256]        (0, d0, 0)        (d0) : 0 <= d0 <= 255
///   [1, 1, 128]     [2]          (0, 0, 64 * d0)   (d0) : 0 <= d0 <= 1
///   [2, 4, 8, 64]   [2, 4, 8]    (d0, d1, d2, 0)   3 dims of those extents
///   [1, 64]         []           (0, 0)            nothing walked
///
/// The last row is a transfer that fits in one vector: the offsets are already
/// the all-zero map, and the caller supplies the single pinned time step.
TransferTimeDims describeTransferTimeDims(llvm::ArrayRef<int64_t> sizes,
                                          int64_t lanes) {
  TransferTimeDims dims(sizes.size());
  auto addDim = [&](unsigned pos, int64_t extent, int64_t index_step) {
    dims.steps.push_back({pos, index_step});
    dims.extents.push_back(extent);
  };
  for (unsigned i = 0; i + 1 < sizes.size(); ++i) {
    if (sizes[i] != 1) addDim(i, sizes[i], /*index_step=*/1);
  }
  if (const int64_t vectors = sizes.back() / lanes; vectors > 1) {
    addDim(sizes.size() - 1, vectors, /*index_step=*/lanes);
  }
  return dims;
}

/// The coefficient of each dimension of `map`, which must have one result, or
/// failure if that result is not a linear function of those dimensions and so
/// has no one coefficient per dimension. A symbol, `floordiv`, `ceildiv` or
/// `mod` in the result is what makes it fail.
///
/// Flattening the map into a Presburger relation is what makes those cases
/// detectable. Evaluating the map at sample points cannot tell them apart from
/// a genuine coefficient.
mlir::FailureOr<llvm::SmallVector<int64_t>> coefficientsIfLinearInDims(
    mlir::AffineMap map) {
  assert(map.getNumResults() == 1 && "expected a one-result map");
  // Not known until runtime, so never a static distance.
  if (map.getNumSymbols() != 0) {
    return mlir::failure();
  }

  mlir::presburger::IntegerRelation relation(
      mlir::presburger::PresburgerSpace::getRelationSpace());
  // Fails for a semi-affine map, which cannot be flattened at all.
  if (mlir::failed(mlir::affine::getRelationFromMap(map, relation))) {
    return mlir::failure();
  }

  // Flattening emits inequalities only to bound the local variables it
  // introduces for `floordiv`, `ceildiv` and `mod`, so any inequality means
  // the result is not linear in the dimensions.
  if (relation.getNumInequalities() != 0) {
    return mlir::failure();
  }
  assert(relation.getNumLocalVars() == 0 &&
         "a local variable is always bounded by inequalities");

  // What is left is the one equality getRelationFromMap states per result,
  //
  //   coefficient_0 * d_0 + ... + coefficient_n * d_n - r + constant == 0
  //
  // whose dimension columns are the answer. Its constant column is the map's
  // base offset, which a distance does not depend on.
  if (relation.getNumEqualities() != 1) {
    return mlir::failure();
  }
  const unsigned num_dims = map.getNumDims();
  // The -1 that equates the flattened expression to `r`. Any other value would
  // scale or flip every coefficient taken below.
  if (relation.atEq64(0, /*result column=*/num_dims) != -1) {
    return mlir::failure();
  }
  llvm::SmallVector<int64_t> coefficients;
  for (unsigned dim = 0; dim < num_dims; ++dim) {
    coefficients.push_back(relation.atEq64(0, dim));
  }
  return coefficients;
}

/// The distance, in elements of the underlying linear memory, between
/// consecutive indices of each dimension of `memref`. A memory view states its
/// own linearization, which need not be the row-major layout implied by the
/// memref shape, so prefer it; the type's own layout is only the fallback.
/// Fails when the layout is not a static linear function of the indices.
mlir::FailureOr<llvm::SmallVector<int64_t>> getElementStrides(
    mlir::Value memref) {
  auto memref_type = llvm::cast<mlir::MemRefType>(memref.getType());
  const unsigned rank = memref_type.getRank();
  if (auto view =
          memref.getDefiningOp<mlir::dataflow::GetLogicalMemoryViewOp>()) {
    auto layout = view.getLayoutMap();
    if (layout.getNumDims() != rank || layout.getNumResults() != 1) {
      return mlir::failure();
    }
    return coefficientsIfLinearInDims(layout);
  }
  llvm::SmallVector<int64_t> strides;
  int64_t offset = 0;
  if (mlir::failed(memref_type.getStridesAndOffset(strides, offset))) {
    return mlir::failure();
  }
  if (llvm::any_of(strides, mlir::ShapedType::isDynamic)) {
    return mlir::failure();
  }
  return strides;
}

/// Extend `map` with one trailing dimension that advances `step.memref_dim` by
/// `step.index_step` indices per unit. Appending a loop induction variable to
/// the subscript operands then drives that dimension from the loop instead of
/// from the time axis.
mlir::AffineMap foldStepIntoSubscripts(mlir::MLIRContext* context,
                                       mlir::AffineMap map,
                                       TransferTimeStep step) {
  llvm::SmallVector<mlir::AffineExpr> results(map.getResults());
  results[step.memref_dim] =
      results[step.memref_dim] +
      mlir::getAffineConstantExpr(step.index_step, context) *
          mlir::getAffineDimExpr(map.getNumDims(), context);
  return mlir::AffineMap::get(map.getNumDims() + 1, map.getNumSymbols(),
                              results, context);
}

/// Pattern to lower ktdf.data_transfer operations
struct LowerDataTransferPattern
    : public mlir::OpRewritePattern<mlir::ktdf::DataTransferOp> {
  LowerDataTransferPattern(mlir::MLIRContext* context,
                           const ResourceToUnits& components,
                           mlir::ktdf_arch::ResourceKinds& resource_kinds)
      : OpRewritePattern(context),
        components_(components),
        resource_kinds_(resource_kinds) {}

  mlir::LogicalResult matchAndRewrite(
      mlir::ktdf::DataTransferOp data_transfer_op,
      mlir::PatternRewriter& rewriter) const override {
    auto src = data_transfer_op.getSource();
    auto dst = data_transfer_op.getDestination();

    bool src_is_fifo = data_transfer_op.isSourceFifo();
    bool dst_is_fifo = data_transfer_op.isDestFifo();

    // Extract FIFO types if applicable
    // Determine transfer type based on source and destination
    auto transfer_type_or = getDataTransferType(src_is_fifo, dst_is_fifo);
    if (mlir::failed(transfer_type_or)) {
      data_transfer_op.emitError(
          "Unsupported data transfer: FIFO to FIFO transfers are not allowed");
      return mlir::failure();
    }
    auto transfer_type = *transfer_type_or;

    // Get source and destination indices
    auto src_indices = data_transfer_op.getSourceIndices();
    auto dst_indices = data_transfer_op.getDestIndices();

    // Get static sizes
    assert(data_transfer_op.hasAllStaticSourceSizes() &&
           "Expected static source sizes");
    assert(data_transfer_op.hasAllStaticDestSizes() &&
           "Expected static dest sizes");

    auto src_static_sizes = *data_transfer_op.getStaticSourceSizes();
    auto dst_static_sizes = *data_transfer_op.getStaticDestSizes();

    // Calculate total elements from static sizes
    int64_t src_total_elements = 1;
    for (int64_t size : src_static_sizes) {
      src_total_elements *= size;
    }

    int64_t dst_total_elements = 1;
    for (int64_t size : dst_static_sizes) {
      dst_total_elements *= size;
    }

    // For splat/pad transfers the source is smaller than the destination —
    // the hardware replicates or zero-pads to fill the vector. Skip the
    // equality check and use the destination size as the transfer width.
    auto transfer_mode_attr =
        data_transfer_op->getDiscardableAttr("transfer_mode");
    bool is_broadcast_transfer =
        transfer_mode_attr &&
        (llvm::cast<mlir::StringAttr>(transfer_mode_attr).getValue() ==
             "splat" ||
         llvm::cast<mlir::StringAttr>(transfer_mode_attr).getValue() == "pad");

    if (is_broadcast_transfer) {
      if (src_total_elements > dst_total_elements) {
        data_transfer_op.emitError(
            "source total elements must not exceed destination for "
            "splat/pad transfer");
        return mlir::failure();
      }
    }

    int64_t total_elements = dst_total_elements;

    // Get element type (from memref or FIFO slot)
    mlir::Type elem_type;
    if (src_is_fifo) {
      elem_type =
          llvm::cast<mlir::ktdf::FifoSlotType>(src.getType()).getElementType();
    } else {
      elem_type = llvm::cast<mlir::MemRefType>(src.getType()).getElementType();
    }

    auto vector_type = mlir::VectorType::get({total_elements}, elem_type);

    // Handle different transfer types
    switch (transfer_type) {
      case DataTransferType::kLoadAndStore: {
        // Both are memrefs
        auto src_memref = src;
        auto dst_memref = dst;
        auto src_memref_type =
            llvm::cast<mlir::MemRefType>(src_memref.getType());
        auto dst_memref_type =
            llvm::cast<mlir::MemRefType>(dst_memref.getType());
        unsigned src_num_dims = src_memref_type.getRank();
        unsigned dst_num_dims = dst_memref_type.getRank();

        auto src_map = data_transfer_op.getSourceMap().value_or(
            mlir::AffineMap::getMultiDimIdentityMap(src_num_dims,
                                                    rewriter.getContext()));
        auto dst_map = data_transfer_op.getDestMap().value_or(
            mlir::AffineMap::getMultiDimIdentityMap(dst_num_dims,
                                                    rewriter.getContext()));

        return lowerAsLoadAndStore(
            rewriter, data_transfer_op, src_memref, dst_memref, src_indices,
            dst_indices, src_static_sizes, dst_static_sizes, src_num_dims,
            dst_num_dims, vector_type, src_map, dst_map);
      }

      case DataTransferType::kLoadAndSend: {
        // Source is memref, destination is FIFO
        auto src_memref = src;
        auto src_memref_type =
            llvm::cast<mlir::MemRefType>(src_memref.getType());
        unsigned num_dims = src_memref_type.getRank();

        auto identity_map = mlir::AffineMap::getMultiDimIdentityMap(
            num_dims, rewriter.getContext());
        auto src_map = data_transfer_op.getSourceMap().value_or(identity_map);

        auto dst_fifo_slot_type =
            llvm::cast<mlir::ktdf::FifoSlotType>(dst.getType());
        return lowerAsLoadAndSend(rewriter, data_transfer_op, src_memref,
                                  src_indices, src_static_sizes, num_dims,
                                  vector_type, src_map, dst_fifo_slot_type,
                                  is_broadcast_transfer, src_total_elements);
      }

      case DataTransferType::kReceiveAndStore: {
        // Source is FIFO, destination is memref
        auto dst_memref = dst;
        auto dst_memref_type =
            llvm::cast<mlir::MemRefType>(dst_memref.getType());
        unsigned num_dims = dst_memref_type.getRank();

        auto identity_map = mlir::AffineMap::getMultiDimIdentityMap(
            num_dims, rewriter.getContext());
        auto dst_map = data_transfer_op.getDestMap().value_or(identity_map);

        auto src_fifo_slot_type =
            llvm::cast<mlir::ktdf::FifoSlotType>(src.getType());
        return lowerAsReceiveAndStore(rewriter, data_transfer_op, dst_memref,
                                      dst_indices, dst_static_sizes, num_dims,
                                      vector_type, dst_map, src_fifo_slot_type);
      }
    }

    return mlir::failure();
  }

 private:
  const ResourceToUnits& components_;
  mlir::ktdf_arch::ResourceKinds& resource_kinds_;

  /// Lower as CompositeLoadAndStore.
  ///
  /// An AGEN composite transfer moves at most one hardware vector per time
  /// step, so a transfer wider than that has to walk the remaining elements
  /// over AGEN time dimensions instead of widening `load_iv`.
  mlir::LogicalResult lowerAsLoadAndStore(
      mlir::PatternRewriter& rewriter,
      mlir::ktdf::DataTransferOp data_transfer_op, mlir::Value src_memref,
      mlir::Value dst_memref, mlir::ValueRange src_indices,
      mlir::ValueRange dst_indices, llvm::ArrayRef<int64_t> src_static_sizes,
      llvm::ArrayRef<int64_t> dst_static_sizes, unsigned src_num_dims,
      unsigned dst_num_dims, mlir::VectorType vector_type,
      mlir::AffineMap src_map, mlir::AffineMap dst_map) const {
    auto* context = rewriter.getContext();
    const int64_t total = vector_type.getNumElements();

    // FIXME: Discover compute from op.
    auto compute = resource_kinds_.getDefaultCompute();
    if (!compute) {
      return data_transfer_op.emitError(
          "cannot determine the hardware vector width: the architecture "
          "declares no default compute resource");
    }

    const auto lanes = getVectorLanes(vector_type.getElementType(), compute);

    // Sizes describing the elements covered by one AGEN vector transfer, and
    // the dimensions (if any) walked over time to cover the rest. Narrowed
    // below when the request is wider than one hardware vector.
    llvm::SmallVector<int64_t> load_sizes(src_static_sizes);
    llvm::SmallVector<int64_t> store_sizes(dst_static_sizes);
    mlir::VectorType load_iv_type = vector_type;
    TransferTimeDims src_time_dims(src_static_sizes.size());
    TransferTimeDims dst_time_dims(dst_static_sizes.size());
    // Set when one time dimension has to leave the time axis for an enclosing
    // loop; see below.
    std::optional<unsigned> loop_time_dim;

    if (total > lanes) {
      if (src_static_sizes.empty() || dst_static_sizes.empty()) {
        data_transfer_op.emitError()
            << "data transfer of " << total
            << " elements exceeds the hardware vector width of " << lanes
            << " but has no dimensions to split";
        return mlir::failure();
      }

      if (src_static_sizes.back() % lanes != 0 ||
          dst_static_sizes.back() % lanes != 0) {
        data_transfer_op.emitError()
            << "data transfer of " << total
            << " elements exceeds the hardware vector width of " << lanes
            << "; splitting requires the innermost source and destination "
               "sizes to be a multiple of the vector width, but they are "
            << src_static_sizes.back() << " and " << dst_static_sizes.back();
        return mlir::failure();
      }

      src_time_dims = describeTransferTimeDims(src_static_sizes, lanes);
      dst_time_dims = describeTransferTimeDims(dst_static_sizes, lanes);

      // Only the extents are compared, not positions or coefficients: the
      // two sides may reach the same walk through different shapes, e.g.
      // src [2, 64] and dst [1, 128] with 64 lanes both walk 2 steps.
      if (src_time_dims.extents != dst_time_dims.extents) {
        data_transfer_op.emitError()
            << "source and destination walked dimensions must match to "
               "split a transfer of "
            << total << " elements across multiple vectors";
        return mlir::failure();
      }

      load_iv_type =
          mlir::VectorType::get({lanes}, vector_type.getElementType());
      load_sizes.assign(src_static_sizes.size(), 1);
      load_sizes.back() = lanes;
      store_sizes.assign(dst_static_sizes.size(), 1);
      store_sizes.back() = lanes;

      // A time dimension is one step count shared by both sides, so it is only
      // realizable when both sides move the same distance per step: one count
      // cannot stand for two different strides. Matching extents do not imply
      // matching distances — the sides may reach the same walk through
      // different shapes and layouts — so compare the distances.
      auto src_strides = getElementStrides(src_memref);
      auto dst_strides = getElementStrides(dst_memref);
      if (mlir::succeeded(src_strides) && mlir::succeeded(dst_strides)) {
        auto distance = [](TransferTimeStep step,
                           llvm::ArrayRef<int64_t> strides) {
          return step.index_step * strides[step.memref_dim];
        };
        llvm::SmallVector<unsigned> divergent;
        for (unsigned i = 0, e = src_time_dims.extents.size(); i < e; ++i) {
          if (src_time_dims.extents[i] == 1) continue;
          if (distance(src_time_dims.steps[i], *src_strides) !=
              distance(dst_time_dims.steps[i], *dst_strides)) {
            divergent.push_back(i);
          }
        }
        // A dimension whose distances disagree can still be walked by an
        // enclosing loop: its step then lands in the memref subscripts, where
        // each side applies its own layout. Several such dimensions would need
        // a nesting order among them, which nothing here determines.
        if (divergent.size() > 1) {
          data_transfer_op.emitError()
              << "source and destination advance by different distances in "
              << divergent.size() << " walked dimensions of a transfer of "
              << total << " elements; only one such dimension can be resolved";
          return mlir::failure();
        }
        if (divergent.size() == 1) loop_time_dim = divergent.front();
      }
    }

    // Subscript operands, gaining the loop induction variable when a time
    // dimension moves to an enclosing loop.
    llvm::SmallVector<mlir::Value> load_indices(src_indices);
    llvm::SmallVector<mlir::Value> store_indices(dst_indices);

    if (loop_time_dim) {
      const auto loc = data_transfer_op.getLoc();
      auto lower = mlir::arith::ConstantIndexOp::create(rewriter, loc, 0);
      auto upper = mlir::arith::ConstantIndexOp::create(
          rewriter, loc, src_time_dims.extents[*loop_time_dim]);
      auto step = mlir::arith::ConstantIndexOp::create(rewriter, loc, 1);
      auto loop = mlir::scf::ForOp::create(rewriter, loc, lower, upper, step);

      src_map = foldStepIntoSubscripts(context, src_map,
                                       src_time_dims.steps[*loop_time_dim]);
      load_indices.push_back(loop.getInductionVar());
      dst_map = foldStepIntoSubscripts(context, dst_map,
                                       dst_time_dims.steps[*loop_time_dim]);
      store_indices.push_back(loop.getInductionVar());

      src_time_dims.eraseDim(*loop_time_dim);
      dst_time_dims.eraseDim(*loop_time_dim);

      rewriter.setInsertionPointToStart(loop.getBody());
    }

    // Build src_load_set / dst_store_set from the per-vector sizes.
    auto src_load_set =
        scheduler::buildIntegerSetFromSizes(context, load_sizes);
    auto dst_store_set =
        scheduler::buildIntegerSetFromSizes(context, store_sizes);

    // load_order and store_order must match their respective set
    // dimensionality.
    auto load_order =
        mlir::AffineMap::getMultiDimIdentityMap(src_num_dims, context);
    auto store_order =
        mlir::AffineMap::getMultiDimIdentityMap(dst_num_dims, context);

    // Time dimensions: a single pinned step for a transfer of at most one
    // vector, one dimension per walked dimension otherwise.
    llvm::SmallVector<int64_t> time_extents = src_time_dims.extents;
    if (time_extents.empty()) time_extents.push_back(1);
    const unsigned num_time_dims = time_extents.size();
    auto time_set = scheduler::buildIntegerSetFromSizes(context, time_extents);
    // time_order: identity. Correct because the walked dimensions above are
    // listed slowest-varying first, so time dimension 0 is already the
    // outermost and the last time dimension the innermost/fastest-varying.
    auto time_order =
        mlir::AffineMap::getMultiDimIdentityMap(num_time_dims, context);
    auto load_addr_map = mlir::AffineMap::get(
        num_time_dims, 0, src_time_dims.offsets(context), context);
    auto store_addr_map = mlir::AffineMap::get(
        num_time_dims, 0, dst_time_dims.offsets(context), context);

    // Create the CompositeLoadAndStoreOp
    mlir::agen::CompositeLoadAndStoreOp::create(
        rewriter, data_transfer_op.getLoc(), src_memref, dst_memref,
        /*dbgName=*/nullptr, src_map, load_indices, dst_map, store_indices,
        src_load_set, load_order, dst_store_set, store_order, {}, time_set,
        time_order, load_addr_map, store_addr_map, load_iv_type);

    // Erase the original data_transfer operation
    rewriter.eraseOp(data_transfer_op);
    return mlir::success();
  }

  /// Lower as vector_load and send (L1 to FIFO)
  mlir::LogicalResult lowerAsLoadAndSend(
      mlir::PatternRewriter& rewriter,
      mlir::ktdf::DataTransferOp data_transfer_op, mlir::Value src_memref,
      mlir::ValueRange src_indices, llvm::ArrayRef<int64_t> src_static_sizes,
      unsigned num_dims, mlir::VectorType vector_type, mlir::AffineMap src_map,
      mlir::ktdf::FifoSlotType dst_fifo_slot_type, bool is_splat,
      int64_t src_total_elements) const {
    // Build load_set from source sizes
    auto load_set =
        buildIntegerSetFromSizes(rewriter.getContext(), src_static_sizes);

    // Build load_order
    auto load_order = mlir::AffineMap::getMultiDimIdentityMap(
        num_dims, rewriter.getContext());

    // Find the enclosing program_unit (needed both for the splat granularity
    // query and for the send destination resolution below).
    auto program_unit =
        data_transfer_op->getParentOfType<mlir::dataflow::ProgramUnitOp>();
    if (!program_unit) {
      data_transfer_op.emitError("data_transfer must be inside a program_unit");
      return mlir::failure();
    }

    // When splat: verify the load unit declares the splat capability, then
    // determine the effective load width from the arch's access_granularity
    // and shuffle to the full destination width.
    int64_t load_elements = src_total_elements;
    if (is_splat) {
      mlir::Attribute load_unit_kind = kindFromProgramUnit(program_unit);
      auto simd_feature =
          resource_kinds_.getFeature<mlir::ktdf_arch::feature::SIMD>(
              load_unit_kind);
      if (!simd_feature || !simd_feature.canSplat()) {
        data_transfer_op.emitError(
            "splat data_transfer requires the load unit to declare "
            "ktdf_arch.feature.simd = { splat, ... }");
        return mlir::failure();
      }
      load_elements = computeSplatGranularityElements(
          src_total_elements, vector_type.getElementType(), load_unit_kind,
          resource_kinds_);
      if (vector_type.getNumElements() % load_elements != 0) {
        data_transfer_op.emitError(
            "dst_total_elements must be divisible by the effective splat "
            "load width (access granularity-aligned src elements)");
        return mlir::failure();
      }
    }
    auto load_type = is_splat
                         ? mlir::VectorType::get({load_elements},
                                                 vector_type.getElementType())
                         : vector_type;

    // Create vector_load operation
    auto vector_load_op = mlir::agen::VectorLoadOp::create(
        rewriter, data_transfer_op.getLoc(), load_type, src_memref,
        /*dbgName=*/nullptr, src_map, src_indices, load_set, load_order);

    // For splat: broadcast the granularity-sized load to full destination
    // width.
    mlir::Value send_value = vector_load_op.getResult();
    if (is_splat) {
      send_value = scheduler::insertSplatShuffle(
          rewriter, data_transfer_op.getLoc(), send_value, load_elements,
          vector_type.getNumElements());
    }

    // Resolve the destination unit from the FIFO dest attribute
    auto dest_unit_result = resolveUnitFromFifoAttr(
        dst_fifo_slot_type.getDest(), components_, rewriter, program_unit,
        data_transfer_op.getLoc(), data_transfer_op.getOperation());
    if (mlir::failed(dest_unit_result)) {
      return mlir::failure();
    }
    mlir::Value dest_unit = *dest_unit_result;

    // Create dataflow.send operation
    mlir::dataflow::SendOp::create(rewriter, data_transfer_op.getLoc(),
                                   dest_unit, send_value,
                                   /*dir=*/nullptr,
                                   /*dbgName=*/nullptr);

    // Erase the original data_transfer operation
    rewriter.eraseOp(data_transfer_op);
    return mlir::success();
  }

  /// Lower as receive and vector_store (FIFO to L1)
  mlir::LogicalResult lowerAsReceiveAndStore(
      mlir::PatternRewriter& rewriter,
      mlir::ktdf::DataTransferOp data_transfer_op, mlir::Value dst_memref,
      mlir::ValueRange dst_indices, llvm::ArrayRef<int64_t> dst_static_sizes,
      unsigned num_dims, mlir::VectorType vector_type, mlir::AffineMap dst_map,
      mlir::ktdf::FifoSlotType src_fifo_slot_type) const {
    // Build store_set from destination sizes
    auto store_set =
        buildIntegerSetFromSizes(rewriter.getContext(), dst_static_sizes);

    // Build store_order
    auto store_order = mlir::AffineMap::getMultiDimIdentityMap(
        num_dims, rewriter.getContext());

    // Find the enclosing program_unit
    auto program_unit =
        data_transfer_op->getParentOfType<mlir::dataflow::ProgramUnitOp>();
    if (!program_unit) {
      data_transfer_op.emitError("data_transfer must be inside a program_unit");
      return mlir::failure();
    }

    // Resolve the source unit from the FIFO src attribute
    auto src_unit_result = resolveUnitFromFifoAttr(
        src_fifo_slot_type.getSrc(), components_, rewriter, program_unit,
        data_transfer_op.getLoc(), data_transfer_op.getOperation());
    if (mlir::failed(src_unit_result)) {
      return mlir::failure();
    }
    mlir::Value src_unit = *src_unit_result;

    // Create dataflow.receive operation
    auto receive_op = mlir::dataflow::ReceiveOp::create(
        rewriter, data_transfer_op.getLoc(), vector_type, src_unit,
        /*dbgName=*/nullptr);

    // Create vector_store operation
    mlir::agen::VectorStoreOp::create(
        rewriter, data_transfer_op.getLoc(), receive_op.getData(), dst_memref,
        /*dbgName=*/nullptr, dst_map, dst_indices, store_set, store_order);

    // Erase the original data_transfer operation
    rewriter.eraseOp(data_transfer_op);
    return mlir::success();
  }
};

}  // namespace

void scheduler::populateDataTransferLoweringPatterns(
    mlir::RewritePatternSet& patterns, const ResourceToUnits& components,
    mlir::ktdf_arch::ResourceKinds& resource_kinds) {
  patterns.add<LowerDataTransferPattern>(patterns.getContext(), components,
                                         resource_kinds);
}
