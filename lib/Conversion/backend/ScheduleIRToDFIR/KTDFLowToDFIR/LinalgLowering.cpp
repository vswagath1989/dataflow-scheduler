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
// Lowering compute operations (linalg.generic, arith, math) into DFIR.
//
//===----------------------------------------------------------------------===//

#include "dataflow-scheduler/Conversion/backend/ScheduleIRToDFIR/KTDFLowToDFIR/LinalgLowering.h"

#include <llvm/ADT/APInt.h>
#include <mlir/IR/Attributes.h>
#include <mlir/IR/BuiltinTypeInterfaces.h>
#include <mlir/IR/Matchers.h>

#include "dataflow-scheduler/Conversion/backend/ScheduleIRToDFIR/KTDFLowToDFIR/Utils.h"
#include "dataflow-scheduler/Dialect/Agen/Agen.h"
#include "dataflow-scheduler/Dialect/Dataflow/Dataflow.h"
#include "dataflow-scheduler/Dialect/KTDF/KTDF.h"
#include "dataflow-scheduler/Dialect/KTDFArch/Analysis/ResourceKinds.h"
#include "dataflow-scheduler/Dialect/KTDFArch/KTDFArch.h"
#include "dataflow-scheduler/Dialect/VectorChain/VectorChain.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/TypeSwitch.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/IntegerSet.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LogicalResult.h"

#define DEBUG_TYPE "ktdflowering-to-dfir"

using namespace scheduler;

namespace {

// ---------------------------------------------------------------------------
// Returns true and sets `lhs_out`/`rhs_out` to the inputs of the abs ops when
// `maxnum_op` implements abs_max, i.e. both operands are produced by a
// math.absf.  The two abs ops are erased via `rewriter` after the caller
// emits its replacement.
// ---------------------------------------------------------------------------
static bool matchAbsMaxOperands(mlir::arith::MaxNumFOp maxnum_op,
                                mlir::Value& lhs_out, mlir::Value& rhs_out) {
  auto lhs_abs = maxnum_op.getLhs().getDefiningOp<mlir::math::AbsFOp>();
  auto rhs_abs = maxnum_op.getRhs().getDefiningOp<mlir::math::AbsFOp>();
  if (!lhs_abs || !rhs_abs) return false;
  lhs_out = lhs_abs.getOperand();
  rhs_out = rhs_abs.getOperand();
  return true;
}

/// Pattern to lower linalg.generic compute operations
struct LowerLinalgGenericPattern
    : public mlir::OpRewritePattern<mlir::linalg::GenericOp> {
  LowerLinalgGenericPattern(mlir::MLIRContext* context,
                            mlir::ktdf_arch::ResourceKinds& resource_kinds)
      : OpRewritePattern(context), resource_kinds_(resource_kinds) {}

  mlir::LogicalResult matchAndRewrite(
      mlir::linalg::GenericOp generic_op,
      mlir::PatternRewriter& rewriter) const override {
    // Buffer-semantics path: init operand is a memref accumulator.
    if (generic_op.hasPureBufferSemantics())
      return lowerMemRefGenericOp(generic_op, rewriter);

    if (!generic_op.hasPureTensorSemantics() ||
        generic_op.getNumResults() != 1) {
      return mlir::failure();
    }

    // FIXME: Discover compute from op.
    auto compute = resource_kinds_.getDefaultCompute();
    if (!compute) {
      return llvm::failure();
    }

    // If the generic has any reduction dimensions, delegate to the dedicated
    // reduction lowering path before the elementwise path touches block args.
    for (auto iter_type : generic_op.getIteratorTypesArray()) {
      if (iter_type == mlir::utils::IteratorType::reduction)
        return lowerReductionGenericOp(generic_op, rewriter);
    }

    mlir::Block& body = generic_op.getRegion().front();
    auto yield_op = mlir::dyn_cast<mlir::linalg::YieldOp>(body.getTerminator());
    if (!yield_op || yield_op.getNumOperands() != 1) {
      return mlir::failure();
    }

    // Replace block arguments with generic inputs.  Any input that is a
    // constant tensor (e.g. a dense<0.0>) is converted to an equivalent
    // arith.constant with vector type first so that vectorchain.binary always
    // receives vector-typed operands.
    unsigned num_inputs = generic_op.getNumDpsInputs();
    for (auto [block_arg, input] :
         llvm::zip(body.getArguments().take_front(num_inputs),
                   generic_op.getDpsInputs())) {
      mlir::Value converted =
          convertConstTensorInputToVector(input, generic_op, rewriter, compute);
      rewriter.replaceAllUsesWith(block_arg, converted);
    }

    // Identity affine map used as op_specific_map for binary ops.
    mlir::AffineMap identity_map =
        mlir::AffineMap::getMultiDimIdentityMap(1, rewriter.getContext());

    // Collect compute operations to replace
    llvm::SmallVector<mlir::Operation*> ops_to_lower;
    for (mlir::Operation& op : body.without_terminator()) {
      ops_to_lower.push_back(&op);
    }

    // Process and lower compute operations via visitors
    rewriter.setInsertionPoint(generic_op);
    for (mlir::Operation* op : ops_to_lower) {
      mlir::LogicalResult result =
          mlir::TypeSwitch<mlir::Operation*, mlir::LogicalResult>(op)
              .Case<mlir::arith::MulFOp>([&](mlir::arith::MulFOp op) {
                return lowerBinaryFOp(
                    op, op.getLhs(), op.getRhs(), rewriter, identity_map,
                    mlir::vectorchain::VectorChainBinaryOperator::mul, compute);
              })
              .Case<mlir::arith::AddFOp>([&](mlir::arith::AddFOp op) {
                return lowerBinaryFOp(
                    op, op.getLhs(), op.getRhs(), rewriter, identity_map,
                    mlir::vectorchain::VectorChainBinaryOperator::add, compute);
              })
              .Case<mlir::arith::SubFOp>([&](mlir::arith::SubFOp op) {
                return lowerBinaryFOp(
                    op, op.getLhs(), op.getRhs(), rewriter, identity_map,
                    mlir::vectorchain::VectorChainBinaryOperator::sub, compute);
              })
              .Case<mlir::arith::MaximumFOp>([&](mlir::arith::MaximumFOp op) {
                return lowerBinaryFOp(
                    op, op.getLhs(), op.getRhs(), rewriter, identity_map,
                    mlir::vectorchain::VectorChainBinaryOperator::max, compute);
              })
              .Case<mlir::arith::MinimumFOp>([&](mlir::arith::MinimumFOp op) {
                return lowerBinaryFOp(
                    op, op.getLhs(), op.getRhs(), rewriter, identity_map,
                    mlir::vectorchain::VectorChainBinaryOperator::min, compute);
              })
              .Case<mlir::arith::MinNumFOp>([&](mlir::arith::MinNumFOp op) {
                // A plain min: there is no absolute-min operator for the
                // abs-of-both shape that maxnumf has.
                return lowerBinaryFOp(
                    op, op.getLhs(), op.getRhs(), rewriter, identity_map,
                    mlir::vectorchain::VectorChainBinaryOperator::min);
              })
              .Case<mlir::arith::MaxNumFOp>([&](mlir::arith::MaxNumFOp op) {
                mlir::Value lhs, rhs;
                if (matchAbsMaxOperands(op, lhs, rhs)) {
                  // Capture abs ops before lowering replaces maxnumf (after
                  // which the operand values are no longer reachable via op).
                  mlir::Operation* lhs_abs = op.getLhs().getDefiningOp();
                  mlir::Operation* rhs_abs = op.getRhs().getDefiningOp();
                  mlir::LogicalResult res = lowerBinaryFOp(
                      op, lhs, rhs, rewriter, identity_map,
                      mlir::vectorchain::VectorChainBinaryOperator::abs_max,
                      compute);
                  // maxnumf is now replaced; absf results are unused — safe to
                  // erase.
                  if (mlir::succeeded(res)) {
                    rewriter.eraseOp(lhs_abs);
                    rewriter.eraseOp(rhs_abs);
                  }
                  return res;
                }
                return lowerBinaryFOp(
                    op, op.getLhs(), op.getRhs(), rewriter, identity_map,
                    mlir::vectorchain::VectorChainBinaryOperator::max, compute);
              })
              .Case<mlir::math::AbsFOp>([&](mlir::math::AbsFOp op)
                                            -> mlir::LogicalResult {
                // Consumed and erased by the MaxNumFOp abs_max case above when
                // it is an operand of a maxnumf; nothing to emit here.
                if (op->hasOneUse() &&
                    mlir::isa<mlir::arith::MaxNumFOp>(*op->user_begin()))
                  return mlir::success();
                return op->emitError(
                    "unsupported standalone math.absf in linalg.generic body");
              })
              .Case<mlir::dataflow::OpaqueOp>([&](mlir::dataflow::OpaqueOp op) {
                // Already DFIR, and it reads and writes registers rather than
                // the lanes the body deals in, so it only has to leave the
                // body.
                rewriter.moveOpBefore(op, generic_op);
                return mlir::success();
              })
              .Case<mlir::memref::StoreOp>([&](mlir::memref::StoreOp op) {
                return lowerMemRefStore(op, rewriter);
              })
              .Case<mlir::memref::LoadOp>([&](mlir::memref::LoadOp op) {
                return lowerMemRefLoad(op, rewriter, compute);
              })
              .Default([](mlir::Operation* unknown_op) {
                return unknown_op->emitError(
                    "unsupported operation type in linalg.generic body");
              });

      if (mlir::failed(result)) return mlir::failure();
    }

    // Replace the generic op with the yield operand
    rewriter.replaceOp(generic_op, yield_op.getOperand(0));
    return mlir::success();
  }

 private:
  mlir::ktdf_arch::ResourceKinds& resource_kinds_;

  // Lowers a linalg.generic with buffer semantics (memref init operand).
  // The resulting accumulated vector is written back to the output buffer via
  // agen.vector_store.
  mlir::LogicalResult lowerMemRefGenericOp(
      mlir::linalg::GenericOp generic_op,
      mlir::PatternRewriter& rewriter) const {
    mlir::Location loc = generic_op.getLoc();

    // FIXME: Discover compute from op.
    auto compute = resource_kinds_.getDefaultCompute();
    if (!compute) {
      return llvm::failure();
    }

    mlir::Block& body = generic_op.getRegion().front();
    auto yield_op = mlir::dyn_cast<mlir::linalg::YieldOp>(body.getTerminator());
    const unsigned accumulators = generic_op.getNumDpsInits();
    if (!yield_op || yield_op.getNumOperands() != accumulators) {
      return mlir::failure();
    }

    // Replace input block arguments with their corresponding linalg ins
    // operands.
    unsigned num_inputs = generic_op.getNumDpsInputs();
    for (auto [block_arg, input] :
         llvm::zip(body.getArguments().take_front(num_inputs),
                   generic_op.getDpsInputs()))
      rewriter.replaceAllUsesWith(block_arg, input);

    // Each output block argument is the accumulator value the matching memref
    // holds. Read each into a vector and let the body read that instead. There
    // is one per result: a compute that accumulates more than one thing
    // accumulates into two.
    llvm::SmallVector<mlir::Value> out_memrefs;
    rewriter.setInsertionPoint(generic_op);
    for (unsigned r = 0; r < accumulators; ++r) {
      mlir::Value out_memref = generic_op.getDpsInitOperand(r)->get();
      out_memrefs.push_back(out_memref);

      auto out_memref_type = mlir::cast<mlir::MemRefType>(out_memref.getType());
      auto acc_vec_type = getFlattenedVectorType(out_memref_type, compute);
      if (!acc_vec_type) return mlir::failure();

      rewriter.replaceAllUsesWith(
          body.getArgument(num_inputs + r),
          scheduler::emitVectorLoad(rewriter, loc, acc_vec_type, out_memref));
    }

    mlir::AffineMap identity_map =
        mlir::AffineMap::getMultiDimIdentityMap(1, rewriter.getContext());

    llvm::SmallVector<mlir::Operation*> ops_to_lower;
    for (mlir::Operation& op : body.without_terminator())
      ops_to_lower.push_back(&op);

    rewriter.setInsertionPoint(generic_op);
    for (mlir::Operation* op : ops_to_lower) {
      mlir::LogicalResult result =
          mlir::TypeSwitch<mlir::Operation*, mlir::LogicalResult>(op)
              .Case<mlir::arith::MulFOp>([&](mlir::arith::MulFOp op) {
                return lowerBinaryFOp(
                    op, op.getLhs(), op.getRhs(), rewriter, identity_map,
                    mlir::vectorchain::VectorChainBinaryOperator::mul, compute);
              })
              .Case<mlir::arith::AddFOp>([&](mlir::arith::AddFOp op) {
                return lowerBinaryFOp(
                    op, op.getLhs(), op.getRhs(), rewriter, identity_map,
                    mlir::vectorchain::VectorChainBinaryOperator::add, compute);
              })
              .Case<mlir::arith::SubFOp>([&](mlir::arith::SubFOp op) {
                return lowerBinaryFOp(
                    op, op.getLhs(), op.getRhs(), rewriter, identity_map,
                    mlir::vectorchain::VectorChainBinaryOperator::sub, compute);
              })
              .Case<mlir::arith::MaximumFOp>([&](mlir::arith::MaximumFOp op) {
                return lowerBinaryFOp(
                    op, op.getLhs(), op.getRhs(), rewriter, identity_map,
                    mlir::vectorchain::VectorChainBinaryOperator::max, compute);
              })
              .Case<mlir::arith::MinimumFOp>([&](mlir::arith::MinimumFOp op) {
                return lowerBinaryFOp(
                    op, op.getLhs(), op.getRhs(), rewriter, identity_map,
                    mlir::vectorchain::VectorChainBinaryOperator::min, compute);
              })
              .Case<mlir::arith::MinNumFOp>([&](mlir::arith::MinNumFOp op) {
                // A plain min: there is no absolute-min operator for the
                // abs-of-both shape that maxnumf has.
                return lowerBinaryFOp(
                    op, op.getLhs(), op.getRhs(), rewriter, identity_map,
                    mlir::vectorchain::VectorChainBinaryOperator::min);
              })
              .Case<mlir::memref::StoreOp>([&](mlir::memref::StoreOp op) {
                return lowerMemRefStore(op, rewriter);
              })
              .Case<mlir::memref::LoadOp>([&](mlir::memref::LoadOp op) {
                return lowerMemRefLoad(op, rewriter, compute);
              })
              .Case<mlir::arith::MaxNumFOp>([&](mlir::arith::MaxNumFOp op) {
                mlir::Value lhs, rhs;
                if (matchAbsMaxOperands(op, lhs, rhs)) {
                  mlir::Operation* lhs_abs = op.getLhs().getDefiningOp();
                  mlir::Operation* rhs_abs = op.getRhs().getDefiningOp();
                  mlir::LogicalResult res = lowerBinaryFOp(
                      op, lhs, rhs, rewriter, identity_map,
                      mlir::vectorchain::VectorChainBinaryOperator::abs_max,
                      compute);
                  if (mlir::succeeded(res)) {
                    rewriter.eraseOp(lhs_abs);
                    rewriter.eraseOp(rhs_abs);
                  }
                  return res;
                }
                return lowerBinaryFOp(
                    op, op.getLhs(), op.getRhs(), rewriter, identity_map,
                    mlir::vectorchain::VectorChainBinaryOperator::max, compute);
              })
              .Case<mlir::math::AbsFOp>([&](mlir::math::AbsFOp op)
                                            -> mlir::LogicalResult {
                // Consumed and erased by the MaxNumFOp abs_max case above when
                // it is an operand of a maxnumf; nothing to emit here.
                if (op->hasOneUse() &&
                    mlir::isa<mlir::arith::MaxNumFOp>(*op->user_begin()))
                  return mlir::success();
                return op->emitError(
                    "unsupported standalone math.absf in linalg.generic body");
              })
              .Default([](mlir::Operation* unknown_op) {
                return unknown_op->emitError(
                    "unsupported operation type in linalg.generic body");
              });
      if (mlir::failed(result)) return mlir::failure();
    }

    // Write each accumulated vector back to the buffer it came from.
    for (unsigned r = 0; r < accumulators; ++r) {
      scheduler::emitVectorStore(rewriter, loc, yield_op.getOperand(r),
                                 out_memrefs[r]);
    }

    rewriter.eraseOp(generic_op);
    return mlir::success();
  }

  // Lowers a linalg.generic that has one or more reduction dimensions.
  //
  // One scf.for loop is emitted per reduction dimension (outermost first),
  // carrying the output vector as an iter_arg accumulator.  Each innermost
  // iteration extracts a parallel-shaped slice from the input tensor (size 1
  // along every reduction dim, full extent along parallel dims), then
  // accumulates it into the current accumulator via vectorchain.binary.
  //
  // The body op (addf / mulf / subf) determines the binary operator; the
  // existing lowerXxxFOp helpers are reused for the accumulation step.
  mlir::LogicalResult lowerReductionGenericOp(
      mlir::linalg::GenericOp generic_op,
      mlir::PatternRewriter& rewriter) const {
    mlir::Location loc = generic_op.getLoc();

    // FIXME: Discover compute from op.
    auto compute = resource_kinds_.getDefaultCompute();
    if (!compute) {
      return llvm::failure();
    }

    // Require exactly one body op (plus the linalg.yield terminator and
    // possibly arith.constant ops).
    mlir::Block& body = generic_op.getRegion().front();
    llvm::SmallVector<mlir::Operation*> body_ops;
    for (mlir::Operation& op : body.without_terminator()) {
      if (op.hasTrait<mlir::OpTrait::ConstantLike>()) continue;
      body_ops.push_back(&op);
    }
    // Allow either one compute op or the abs_max pattern:
    //   math.absf %in  /  math.absf %out  /  arith.maxnumf %abs_in, %abs_out
    if (body_ops.size() != 1 && body_ops.size() != 3)
      return generic_op.emitError(
          "reduction linalg.generic body must have exactly one compute op");

    // Map body op kind to the vectorchain binary operator.
    mlir::vectorchain::VectorChainBinaryOperator binary_kind;
    if (body_ops.size() == 3) {
      // The only supported 3-op body is abs+abs+maxnumf → abs_max.
      auto* maxnum = body_ops[2];
      if (!mlir::isa<mlir::arith::MaxNumFOp>(maxnum) ||
          !mlir::isa<mlir::math::AbsFOp>(body_ops[0]) ||
          !mlir::isa<mlir::math::AbsFOp>(body_ops[1]))
        return generic_op.emitError(
            "unsupported 3-op reduction body: expected absf/absf/maxnumf");
      binary_kind = mlir::vectorchain::VectorChainBinaryOperator::abs_max;
    } else if (mlir::isa<mlir::arith::AddFOp>(body_ops[0]))
      binary_kind = mlir::vectorchain::VectorChainBinaryOperator::add;
    else if (mlir::isa<mlir::arith::MulFOp>(body_ops[0]))
      binary_kind = mlir::vectorchain::VectorChainBinaryOperator::mul;
    else if (mlir::isa<mlir::arith::SubFOp>(body_ops[0]))
      binary_kind = mlir::vectorchain::VectorChainBinaryOperator::sub;
    else if (mlir::isa<mlir::arith::MaximumFOp>(body_ops[0]))
      binary_kind = mlir::vectorchain::VectorChainBinaryOperator::max;
    else if (mlir::isa<mlir::arith::MinimumFOp>(body_ops[0]))
      binary_kind = mlir::vectorchain::VectorChainBinaryOperator::min;
    else
      return body_ops[0]->emitError(
          "unsupported reduction body op in linalg.generic");

    // Collect reduction dim indices and their sizes from the input type.
    auto input_type = mlir::dyn_cast<mlir::RankedTensorType>(
        generic_op.getDpsInputOperand(0)->get().getType());
    if (!input_type) return mlir::failure();

    const auto iterator_types = generic_op.getIteratorTypesArray();
    llvm::SmallVector<int64_t> red_dims;
    for (int64_t i = 0; i < static_cast<int64_t>(iterator_types.size()); ++i) {
      if (iterator_types[i] == mlir::utils::IteratorType::reduction)
        red_dims.push_back(i);
    }

    // The output type has only parallel dims and always fits in vector_length.
    auto output_type = mlir::dyn_cast<mlir::RankedTensorType>(
        generic_op.getDpsInitOperand(0)->get().getType());
    if (!output_type) return mlir::failure();

    mlir::VectorType vec_type = getFlattenedVectorType(output_type, compute);
    if (!vec_type) return mlir::failure();

    mlir::AffineMap identity_map =
        mlir::AffineMap::getMultiDimIdentityMap(1, rewriter.getContext());

    rewriter.setInsertionPoint(generic_op);

    // Build the initial accumulator from the init operand.  For a tensor.empty
    // (undefined init) use a zero vector; for a constant tensor reshape it.
    mlir::Value init = generic_op.getDpsInitOperand(0)->get();
    mlir::Value acc =
        convertConstTensorInputToVector(init, generic_op, rewriter, compute);
    if (acc.getType() != vec_type) {
      // Non-constant init (e.g. tensor.empty) — zero is the correct identity
      // for reductions that start with an uninitialised accumulator.
      acc = mlir::arith::ConstantOp::create(rewriter, loc, vec_type,
                                            rewriter.getZeroAttr(vec_type));
    }

    // Emit one scf.for per reduction dimension (outermost first), each
    // carrying the accumulator as its single iter_arg.
    //
    // The input to the linalg.generic comes from a ktdf.read_from_fifo whose
    // result type includes all dims (parallel + reduction).  Rather than
    // tensor.extract_slice-ing that large tensor inside the loop (which would
    // require LowerReadFromFifoPattern to lower a tensor larger than
    // vector_length), we instead emit one new ktdf.read_from_fifo per loop
    // iteration — each producing the parallel-only shaped slice directly.
    // The original read_from_fifo is replaced by the new ones so it is erased.
    mlir::Value input = generic_op.getDpsInputOperand(0)->get();
    auto read_from_fifo = mlir::dyn_cast_or_null<mlir::ktdf::ReadFromFifoOp>(
        input.getDefiningOp());
    if (!read_from_fifo)
      return generic_op.emitError(
          "reduction linalg.generic input must be produced by "
          "ktdf.read_from_fifo");

    // Parallel-only result type for each per-step read.
    llvm::SmallVector<int64_t> parallel_shape;
    for (int64_t i = 0; i < static_cast<int64_t>(iterator_types.size()); ++i) {
      if (iterator_types[i] != mlir::utils::IteratorType::reduction)
        parallel_shape.push_back(input_type.getShape()[i]);
    }
    auto parallel_type = mlir::RankedTensorType::get(
        parallel_shape, input_type.getElementType());

    mlir::Value cur_acc = acc;

    // Build nested scf.for loops, one per reduction dimension.
    llvm::SmallVector<mlir::scf::ForOp> for_ops;
    for (int64_t red_dim : red_dims) {
      mlir::Value lb = mlir::arith::ConstantIndexOp::create(rewriter, loc, 0);
      mlir::Value ub = mlir::arith::ConstantIndexOp::create(
          rewriter, loc, input_type.getShape()[red_dim]);
      mlir::Value step = mlir::arith::ConstantIndexOp::create(rewriter, loc, 1);
      auto for_op = mlir::scf::ForOp::create(rewriter, loc, lb, ub, step,
                                             mlir::ValueRange{cur_acc});
      for_ops.push_back(for_op);
      rewriter.setInsertionPointToStart(for_op.getBody());
      cur_acc = for_op.getRegionIterArgs()[0];
    }

    // Innermost body: emit a new read_from_fifo producing parallel_type,
    // then accumulate via vectorchain.binary.
    auto new_read = mlir::ktdf::ReadFromFifoOp::create(
        rewriter, loc, parallel_type, read_from_fifo.getFifoSlot());

    // The parallel slice fits in vector_length.
    auto binary = mlir::vectorchain::BinaryOp::create(
        rewriter, loc, vec_type, cur_acc, new_read.getResult(),
        /*mask=*/nullptr, /*dbgName=*/nullptr, binary_kind, identity_map);

    // Yield the new accumulator up through each loop level.
    mlir::Value result = binary.getData();
    for (auto for_op : llvm::reverse(for_ops)) {
      mlir::scf::YieldOp::create(rewriter, loc, mlir::ValueRange{result});
      result = for_op.getResult(0);
      rewriter.setInsertionPointAfter(for_op);
    }

    // Replace the linalg.generic result and erase the original read_from_fifo
    // (which is now unused).
    rewriter.replaceOp(generic_op, result);
    rewriter.eraseOp(read_from_fifo);
    return mlir::success();
  }

  /// Converts a constant tensor input of a linalg.generic to a vector-typed
  /// arith.constant, preserving all element values.  This is needed because
  /// linalg.generic inputs can be constant tensors (e.g. a dense<0.0>), while
  /// vectorchain.binary requires vector operands.
  ///
  /// Only arith.constant ops whose value is a DenseElementsAttr are handled;
  /// any other input (non-constant tensors, vectors, scalars) is returned
  /// unchanged.
  mlir::Value convertConstTensorInputToVector(
      mlir::Value input, mlir::linalg::GenericOp generic_op,
      mlir::PatternRewriter& rewriter,
      mlir::ktdf_arch::ExecutionUnitOp compute) const {
    // Only act on tensor-typed inputs — vectors and scalars pass through.
    auto tensor_type = mlir::dyn_cast<mlir::RankedTensorType>(input.getType());
    if (!tensor_type) return input;

    // Must be a constant op with a dense attribute to convert.
    auto const_op =
        mlir::dyn_cast_or_null<mlir::arith::ConstantOp>(input.getDefiningOp());
    if (!const_op) return input;
    auto dense_attr =
        mlir::dyn_cast<mlir::DenseElementsAttr>(const_op.getValue());
    if (!dense_attr) return input;

    // Determine the target vector type (same element type, flattened shape).
    auto vector_type = getFlattenedVectorType(tensor_type, compute);
    if (!vector_type) return input;

    // Re-materialise the constant with the vector type, preserving all element
    // values by reinterpreting the same dense data into the flat vector shape.
    auto vec_attr = dense_attr.reshape(vector_type);
    rewriter.setInsertionPoint(generic_op);
    return mlir::arith::ConstantOp::create(rewriter, const_op.getLoc(),
                                           vector_type, vec_attr)
        .getResult();
  }

  /// Lowers a store into a register to the vector store that writes it.
  ///
  /// A register is written whole, so the value has to be a vector by the time
  /// this runs. Returns failure while it is still the body's scalar, so the
  /// driver comes back to it.
  mlir::LogicalResult lowerMemRefStore(mlir::memref::StoreOp op,
                                       mlir::PatternRewriter& rewriter) const {
    if (!mlir::isa<mlir::VectorType>(op.getValueToStore().getType())) {
      return mlir::failure();
    }

    const auto access = getRegisterAccess(op.getMemRef());
    if (!access) return mlir::failure();

    mlir::agen::VectorStoreOp::create(
        rewriter, op.getLoc(), op.getValueToStore(), op.getMemRef(),
        /*dbgName=*/nullptr, access->map, op.getIndices(), access->set,
        access->order);
    rewriter.eraseOp(op);
    return mlir::success();
  }

  /// Lowers a load out of a register to the vector load that reads it.
  mlir::LogicalResult lowerMemRefLoad(
      mlir::memref::LoadOp op, mlir::PatternRewriter& rewriter,
      mlir::ktdf_arch::ExecutionUnitOp compute) const {
    const auto vector_type = getFlattenedVectorType(
        llvm::cast<mlir::MemRefType>(op.getMemRef().getType()), compute);
    if (!vector_type) return mlir::failure();

    const auto access = getRegisterAccess(op.getMemRef());
    if (!access) return mlir::failure();

    auto load = mlir::agen::VectorLoadOp::create(
        rewriter, op.getLoc(), vector_type, op.getMemRef(),
        /*dbgName=*/nullptr, access->map, op.getIndices(), access->set,
        access->order, /*multicast_info=*/nullptr);
    rewriter.replaceOp(op, load.getResult());
    return mlir::success();
  }

  /// Holds the maps that address a register: the whole of it, in lane order.
  struct RegisterAccess {
    mlir::AffineMap map;
    mlir::IntegerSet set;
    mlir::AffineMap order;
  };

  std::optional<RegisterAccess> getRegisterAccess(mlir::Value mem_ref) const {
    const auto type = mlir::dyn_cast<mlir::MemRefType>(mem_ref.getType());
    if (!type || !type.hasStaticShape()) return std::nullopt;

    auto* const ctx = mem_ref.getContext();
    const auto rank = static_cast<unsigned>(type.getRank());
    return RegisterAccess{mlir::AffineMap::getMultiDimIdentityMap(rank, ctx),
                          buildIntegerSetFromSizes(ctx, type.getShape()),
                          mlir::AffineMap::getMultiDimIdentityMap(rank, ctx)};
  }

  // Unified helper: lowers any two-operand arith float op to
  // vectorchain.binary with the given binary_kind.
  mlir::LogicalResult lowerBinaryFOp(
      mlir::Operation* op, mlir::Value lhs, mlir::Value rhs,
      mlir::PatternRewriter& rewriter, mlir::AffineMap identity_map,
      mlir::vectorchain::VectorChainBinaryOperator binary_kind,
      mlir::ktdf_arch::ExecutionUnitOp compute) const {
    const auto lhs_ty = llvm::dyn_cast<mlir::ShapedType>(lhs.getType());
    if (!lhs_ty) {
      return llvm::failure();
    }
    auto vector_type = getFlattenedVectorType(lhs_ty, compute);
    if (!vector_type) return mlir::failure();

    auto binary_op = mlir::vectorchain::BinaryOp::create(
        rewriter, op->getLoc(), vector_type, lhs, rhs,
        /*mask=*/nullptr, /*dbgName=*/nullptr, binary_kind, identity_map);

    rewriter.replaceOp(op, binary_op.getData());
    return mlir::success();
  }
};

/// Pattern to lower linalg.fill into:
///   vectorchain.constant_bitstream {value = [0x0]} : vector<1xT>
///   vectorchain.shuffle ... {indices = [0 : i32], repetition = N}
///       : vector<1xT>, vector<NxT>
///
/// Buffer semantics (memref output): the shuffle result is written to the
/// output memref via agen.vector_store and the fill is erased.
///
/// Tensor semantics (tensor output): the shuffle result directly replaces
/// the fill result (consumed by downstream vectorchain / FIFO ops).
///
/// N and T are derived from the output type shape and element type, and the
/// bitstream carries the fill value as the bits a lane holds.
struct LowerLinalgFillPattern
    : public mlir::OpRewritePattern<mlir::linalg::FillOp> {
  LowerLinalgFillPattern(mlir::MLIRContext* context,
                         mlir::ktdf_arch::ResourceKinds& resource_kinds,
                         scheduler::SymbolAllocator& symbols)
      : OpRewritePattern(context),
        resource_kinds_(resource_kinds),
        symbols_(symbols) {}

  mlir::LogicalResult matchAndRewrite(
      mlir::linalg::FillOp fill_op,
      mlir::PatternRewriter& rewriter) const override {
    // Match constant fill value input.
    if (fill_op.getInputs().size() != 1) {
      return rewriter.notifyMatchFailure(fill_op,
                                         "must have exactly one operand");
    }
    const mlir::Value filled = fill_op.getInputs()[0];
    auto* const fill_def = filled.getDefiningOp();
    mlir::Attribute value;
    const bool is_constant =
        fill_def && mlir::m_Constant(&value).match(fill_def);

    // FIXME: Discover compute from op.
    auto compute = resource_kinds_.getDefaultCompute();
    if (!compute) {
      return llvm::failure();
    }

    // A fill value the compiler does not know becomes a symbol: the bitstream
    // carries the id, and whatever resolves the symbols writes the value in.
    // What the symbol is is declared next to the program.
    int64_t symbol_id = 0;
    if (!is_constant) {
      auto program = fill_op->getParentOfType<mlir::func::FuncOp>();
      if (!program) {
        return rewriter.notifyMatchFailure(fill_op, "fill is in no function");
      }
      const auto declared =
          scheduler::declareScalarSymbol(filled, program, symbols_);
      if (mlir::failed(declared)) {
        return rewriter.notifyMatchFailure(
            fill_op, "fill value is neither a constant nor a symbol");
      }
      symbol_id = *declared;
    }

    llvm::APInt fill_bits;
    if (is_constant) {
      if (const auto attr = mlir::dyn_cast<mlir::FloatAttr>(value)) {
        fill_bits = attr.getValue().bitcastToAPInt();
      } else if (const auto attr = mlir::dyn_cast<mlir::IntegerAttr>(value)) {
        fill_bits = attr.getValue();
      } else {
        return rewriter.notifyMatchFailure(fill_op,
                                           "fill value must be int or float");
      }
      if (fill_bits.getBitWidth() > 64) {
        return rewriter.notifyMatchFailure(
            fill_op, "fill value must not exceed 64 bits");
      }
      fill_bits = fill_bits.zext(64U);
    }

    // Derive output vector type from the output operand (memref or tensor).
    mlir::Value out_operand = fill_op.getOutputs()[0];
    mlir::VectorType out_vec_type = getFlattenedVectorType(
        llvm::cast<mlir::ShapedType>(out_operand.getType()), compute);
    if (!out_vec_type) {
      return rewriter.notifyMatchFailure(
          fill_op, "output must convert to a flattened vector type");
    }

    mlir::Location loc = fill_op.getLoc();
    int64_t total_elements = out_vec_type.getNumElements();
    mlir::Type elem_type = out_vec_type.getElementType();

    rewriter.setInsertionPoint(fill_op);

    // Step 1: vectorchain.constant_bitstream {value = [0x0]} : vector<1xT>
    mlir::VectorType seed_type = mlir::VectorType::get({1}, elem_type);
    mlir::ArrayAttr value_attr = rewriter.getArrayAttr(
        {is_constant
             ? mlir::IntegerAttr::get(rewriter.getI64Type(), fill_bits)
             : mlir::IntegerAttr::get(rewriter.getI64Type(), symbol_id)});
    auto bitstream = mlir::vectorchain::ConstantBitstreamOp::create(
        rewriter, loc, seed_type, value_attr);
    if (!is_constant) {
      bitstream->setAttr("is_symbol", rewriter.getBoolAttr(true));
    }

    // Step 2: vectorchain.shuffle — splat to vector<NxT>
    mlir::ArrayAttr indices_attr = rewriter.getArrayAttr(
        {mlir::IntegerAttr::get(rewriter.getI32Type(), 0)});
    auto shuffle = mlir::vectorchain::ShuffleOp::create(
        rewriter, loc, out_vec_type, bitstream.getResult(),
        /*variable=*/mlir::ValueRange{}, /*pad=*/mlir::ValueRange{},
        /*mask=*/nullptr, /*dbgName=*/nullptr, indices_attr,
        static_cast<uint32_t>(total_elements));

    // Step 3a (tensor): replace the fill result directly with the vector.
    if (!fill_op.getResultTensors().empty()) {
      rewriter.replaceOp(fill_op, shuffle.getOutput());
      return mlir::success();
    }

    // Step 3b (memref): write the filled vector into the output memref.
    scheduler::emitVectorStore(rewriter, loc, shuffle.getOutput(), out_operand);

    rewriter.eraseOp(fill_op);
    return mlir::success();
  }

 private:
  mlir::ktdf_arch::ResourceKinds& resource_kinds_;
  scheduler::SymbolAllocator& symbols_;
};

}  // namespace

void scheduler::populateLinalgLoweringPatterns(
    mlir::RewritePatternSet& patterns,
    mlir::ktdf_arch::ResourceKinds& resource_kinds, SymbolAllocator& symbols) {
  patterns.add<LowerLinalgGenericPattern>(patterns.getContext(),
                                          resource_kinds);
  patterns.add<LowerLinalgFillPattern>(patterns.getContext(), resource_kinds,
                                       symbols);
}
