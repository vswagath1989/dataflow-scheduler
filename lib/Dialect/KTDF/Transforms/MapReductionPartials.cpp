//===-- MapReductionPartials.cpp --------------------------------*- c++ -*-===//
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
// MapReductionPartials: lower reduction linalg.generic ops to buffer-semantics
// form backed by pre-allocated memrefs.  Two kinds of generics are handled:
//
// ── Outer-dim generic (loop-exposed by ReductionLoopExposure) ────────────────
//
// After ReductionLoopExposure the compute stage looks like:
//
//   %empty = tensor.empty() : tensor<CxDxf16>
//   %result = scf.for %r = 0 to R iter_args(%carry = %empty)
//                 {loop_type = reduction_loop}
//     %slice = ktdf.read_from_fifo %fifo -> tensor<CxDxf16>
//     %new_carry = linalg.generic ins(%slice) outs(%carry)
//     scf.if (%r == R-1): ktdf.write_to_fifo %new_carry, %fifo_out
//     scf.yield %new_carry
//
// This pass rewrites that to:
//
//   %alloc = memref.alloc() : memref<CxDxf16, compute_kind>
//   linalg.fill(%zero, %alloc)
//   scf.for %r = 0 to R {
//     %slice = ktdf.read_from_fifo %fifo -> memref<CxDxf16>
//     linalg.generic ins(%slice) outs(%alloc)
//     scf.if (%r == R-1): ktdf.write_to_fifo %alloc, %fifo_out
//   }
//
// When the iter_arg initializer is a conditional (scf.if), the pass recurses
// into each branch and handles the yielded value:
//
//   %alloc = memref.alloc() : memref<CxDxf16, compute_kind>
//   scf.if %cond {                     // no result; was -> (tensor<...>)
//     linalg.fill(%zero, %alloc)       // was: tensor.empty + yield
//   } else {
//     %r = ktdf.read_from_fifo ...     // memref form of read_from_fifo
//     memref.copy %r, %alloc
//   }
//   scf.for %r = 0 to R { ... }       // iter_arg stripped
//
// The iter_arg / loop result / scf.yield operand are stripped, and any
// write_to_fifo that consumed the loop result is patched to use %alloc.
//
// ── Inner-dim generic (produced by SplitReductionInnerOuterDim) ──────────────
//
// After the outer-dim rewrite above, %alloc_G1 (memref<D0x...xDNxf16, ms>)
// has been RAUW'd in place of the loop result, so the inner-dim generic sees:
//
//   %empty = tensor.empty() : tensor<D0x...xDN-1xf16>
//   %result = linalg.generic ins(%alloc_G1: memref<D0x...xDNxf16, ms>)
//                            outs(%empty: tensor<D0x...xDN-1xf16>)
//   ktdf.write_to_fifo %result, %fifo_out
//
// A rank-reducing subview of %alloc_G1 (size=1 on reduction dims, original
// size on parallel dims) is used as the outs buffer; ins is the full %alloc_G1.
// After the reduction, the full %alloc_G1 — not the subview — is written to
// the FIFO so the downstream stage receives all partial sums.
// The FIFO slot type is widened to match %alloc_G1's shape.
//
//   %sv = memref.subview %alloc_G1[0,...][D0,...,1,...][1,...]
//             : memref<D0x...xDNxf16, ms> to memref<D0x...xDN-1xf16, strided,
//             ms>
//   linalg.generic ins(%alloc_G1) outs(%sv)  {original maps & iter_types}
//   ktdf.write_to_fifo %alloc_G1, %fifo_out  // full buffer
//
//===----------------------------------------------------------------------===//

#include <memory>

#include "dataflow-scheduler/Analysis/ArchViews/GroupLocalMemory.h"
#include "dataflow-scheduler/Dialect/KTDF/KTDF.h"
#include "dataflow-scheduler/Dialect/KTDF/Transforms/Passes.h"
#include "dataflow-scheduler/Dialect/KTDFArch/Analysis/DeviceManager.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/Support/DebugLog.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Pass/Pass.h"

#define PASS_NAME "map-reduction-partials"
#define DEBUG_TYPE PASS_NAME

static llvm::cl::opt<bool> DisableThisPass(
    "disable-" PASS_NAME, llvm::cl::desc("Disable Map Reduction Partials pass"),
    llvm::cl::init(false));

using namespace mlir;

namespace mlir::ktdf {
#define GEN_PASS_DEF_MAPREDUCTIONPARTIALSPASS
#include "dataflow-scheduler/Dialect/KTDF/Transforms/Passes.h.inc"
}  // namespace mlir::ktdf

namespace {

// ---------------------------------------------------------------------------
// Return true if `generic_op` has at least one reduction iterator.
// ---------------------------------------------------------------------------
static bool hasReductionIterator(linalg::GenericOp generic_op) {
  for (auto it : generic_op.getIteratorTypesArray())
    if (it == utils::IteratorType::reduction) return true;
  return false;
}

// ---------------------------------------------------------------------------
// Return the typed neutral-element attribute for the reduction combiner of
// `generic_op`.
//
// The combiner is identified as the unique user of the output block argument
// (outs[0], the last block argument) — that is the accumulation step.  Any
// other ops in the body (e.g. element-wise pre-processing in a fused generic)
// are irrelevant, except for the abs-max pattern described below.
//
// Neutral elements per combiner:
//   addf / subf                          →  0.0
//   mulf                                 →  1.0
//   maximumf                             → -inf
//   minimumf                             → +inf
//   maxnumf(absf(%in), absf(%out))       →  0.0   (abs-max: result is >= 0)
//   addi / subi                          →  0
//   muli                                 →  1
//
// Returns failure() (with an emitted error) if the output block argument has
// zero or multiple users, the combiner is unrecognised, or the element type is
// unsupported.
// ---------------------------------------------------------------------------
static FailureOr<TypedAttr> getNeutralAttr(linalg::GenericOp generic_op,
                                           Type elem_type) {
  // The output block argument is the last argument of the body block.
  Value out_arg = generic_op.getRegion().front().getArguments().back();

  // The combiner is the unique op that uses out_arg as an operand.
  Operation* combiner =
      out_arg.hasOneUse() ? out_arg.getUses().begin()->getOwner() : nullptr;
  if (!combiner)
    return generic_op.emitError(
        "MapReductionPartials: output block argument of reduction "
        "linalg.generic must have exactly one user (the accumulation op)");

  // ── Abs-max pattern: maxnumf(absf(%in), absf(%out)) ───────────────────────
  // When the direct user of out_arg is math.absf (pre-processing the
  // accumulator before combining), check whether that absf feeds a
  // maxnumf whose other operand is also an absf.  Both inputs are
  // non-negative after the abs, so the correct neutral element is 0.0.
  bool is_absmax = false;
  if (isa<math::AbsFOp>(combiner) && combiner->hasOneUse()) {
    Operation* max_op = combiner->getUses().begin()->getOwner();
    if (isa<arith::MaxNumFOp>(max_op)) {
      Value abs_out_result = combiner->getResult(0);
      for (Value operand : max_op->getOperands()) {
        if (operand == abs_out_result) continue;
        if (operand.getDefiningOp<math::AbsFOp>()) {
          is_absmax = true;
          break;
        }
      }
    }
  }

  // ── Floating-point combiners ───────────────────────────────────────────────
  if (auto ftype = dyn_cast<FloatType>(elem_type)) {
    const llvm::fltSemantics& sem = ftype.getFloatSemantics();
    if (is_absmax)
      return cast<TypedAttr>(
          FloatAttr::get(elem_type, APFloat::getZero(sem, /*negative=*/false)));
    if (isa<arith::AddFOp, arith::SubFOp>(combiner))
      return cast<TypedAttr>(
          FloatAttr::get(elem_type, APFloat::getZero(sem, /*negative=*/false)));
    if (isa<arith::MulFOp>(combiner))
      return cast<TypedAttr>(FloatAttr::get(elem_type, APFloat(sem, 1)));
    if (isa<arith::MaximumFOp>(combiner))
      return cast<TypedAttr>(
          FloatAttr::get(elem_type, APFloat::getInf(sem, /*negative=*/true)));
    if (isa<arith::MinimumFOp>(combiner))
      return cast<TypedAttr>(
          FloatAttr::get(elem_type, APFloat::getInf(sem, /*negative=*/false)));
    return combiner->emitError(
        "MapReductionPartials: unsupported floating-point reduction combiner");
  }

  // ── Integer combiners ──────────────────────────────────────────────────────
  if (auto itype = dyn_cast<IntegerType>(elem_type)) {
    unsigned width = itype.getWidth();
    if (isa<arith::AddIOp, arith::SubIOp>(combiner))
      return cast<TypedAttr>(IntegerAttr::get(elem_type, APInt(width, 0)));
    if (isa<arith::MulIOp>(combiner))
      return cast<TypedAttr>(IntegerAttr::get(elem_type, APInt(width, 1)));
    return combiner->emitError(
        "MapReductionPartials: unsupported integer reduction combiner");
  }

  return generic_op.emitError(
      "MapReductionPartials: unsupported accumulator element type — "
      "expected float or integer");
}

// ---------------------------------------------------------------------------
// Transform the initializer `init_val` of a loop-carried accumulator in-place,
// emitting linalg.fill (or memref.copy) into `alloc_val` wherever the original
// tensor value was produced.  `alloc_val` is a memref.alloc already emitted at
// the top of the stage.
//
// `generic_op` is the reduction linalg.generic whose combiner determines the
// correct neutral element for the fill.
//
// Three leaf cases are handled; scf.if recurses into both branches:
//
//   tensor.empty()
//     → linalg.fill(%neutral, %alloc) inserted just before the tensor.empty,
//       then the tensor.empty is erased.
//
//   ktdf.read_from_fifo ... -> tensor<...>
//     → emit memref-typed read_from_fifo + memref.copy into %alloc just
//       before the original op, then erase it.
//
//   scf.if %cond -> (tensor<...>) { yield A } else { yield B }
//     → recurse on A (then-branch yield operand 0) and B (else-branch yield
//       operand 0), drop the yield operands so both yields become result-less,
//       then rebuild the scf.if with no result types.
// ---------------------------------------------------------------------------
static LogicalResult lowerIterArgInitializer(Value init_val, Value alloc_val,
                                             linalg::GenericOp generic_op) {
  Operation* defining_op = init_val.getDefiningOp();
  assert(defining_op &&
         "lowerIterArgInitializer: init_val must be an op result");

  Type elem_type = cast<MemRefType>(alloc_val.getType()).getElementType();

  // ── Case: tensor.empty ────────────────────────────────────────────────────
  if (auto empty_op = dyn_cast<tensor::EmptyOp>(defining_op)) {
    auto neutral = getNeutralAttr(generic_op, elem_type);
    if (failed(neutral)) return failure();
    OpBuilder builder(empty_op);
    Location loc = empty_op.getLoc();
    Value neutral_val =
        arith::ConstantOp::create(builder, loc, neutral.value());

    // The identity goes to a buffer of its own, which the accumulator is then
    // copied from. Filling the accumulator directly would put the value in the
    // copy instruction's immediate, and that field is too narrow for a 32-bit
    // one. A buffer written once and never read back is invariant, so what
    // hoists such a write can lift it to where a register is initialised
    // instead -- which carries the whole value.
    auto identity = memref::AllocOp::create(
        builder, loc, cast<MemRefType>(alloc_val.getType()));
    linalg::FillOp::create(builder, loc, ValueRange{neutral_val},
                           ValueRange{identity.getResult()});
    memref::CopyOp::create(builder, loc, identity.getResult(), alloc_val);

    empty_op->erase();
    return success();
  }

  // ── Case: ktdf.read_from_fifo returning a tensor ──────────────────────────
  if (auto read_op = dyn_cast<ktdf::ReadFromFifoOp>(defining_op)) {
    auto tensor_type = cast<RankedTensorType>(read_op.getResult().getType());
    auto memref_type =
        MemRefType::get(tensor_type.getShape(), tensor_type.getElementType());
    OpBuilder builder(read_op);
    Location loc = read_op.getLoc();
    Value new_read = ktdf::ReadFromFifoOp::create(builder, loc, memref_type,
                                                  read_op.getFifoSlot())
                         .getResult();
    memref::CopyOp::create(builder, loc, new_read, alloc_val);
    read_op->erase();
    return success();
  }

  // ── Case: scf.if returning a tensor — recurse into both branches ──────────
  if (auto if_op = dyn_cast<scf::IfOp>(defining_op)) {
    // Drop the yield operand *before* recursing so that when the recursive
    // call erases the defining op (e.g. tensor.empty) no uses remain.
    {
      auto then_yield =
          cast<scf::YieldOp>(if_op.getThenRegion().front().getTerminator());
      Value then_init = then_yield.getOperand(0);
      then_yield->setOperands({});
      if (failed(lowerIterArgInitializer(then_init, alloc_val, generic_op)))
        return failure();
    }

    {
      auto else_yield =
          cast<scf::YieldOp>(if_op.getElseRegion().front().getTerminator());
      Value else_init = else_yield.getOperand(0);
      else_yield->setOperands({});
      if (failed(lowerIterArgInitializer(else_init, alloc_val, generic_op)))
        return failure();
    }

    // Rebuild the scf.if without result types and transfer the rewritten
    // regions.
    OpBuilder builder(if_op);
    auto new_if =
        scf::IfOp::create(builder, if_op.getLoc(),
                          /*resultTypes=*/TypeRange{}, if_op.getCondition(),
                          /*withElseRegion=*/true);
    new_if.getThenRegion().takeBody(if_op.getThenRegion());
    new_if.getElseRegion().takeBody(if_op.getElseRegion());
    if_op.erase();
    return success();
  }

  return defining_op->emitError(
      "lowerIterArgInitializer: unrecognised initializer form — expected "
      "tensor.empty, ktdf.read_from_fifo, or scf.if");
}

// ---------------------------------------------------------------------------
// Walk up the iter_arg chain rooted at `acc_iter_arg` (outs[0] of the
// original linalg.generic, captured before erasure) and rebuild each
// enclosing scf.for without that iter_arg slot, replacing every use of
// the dropped iter_arg and its loop result with `alloc_val`.
//
// After the innermost loop is rebuilt the old loop result may itself be an
// iter_arg of an outer scf.for — we keep climbing until the chain exits a
// scf.for.
//
// The initializer at the top of the chain has already been handled by
// lowerIterArgInitializer before this function is called, so it is NOT
// erased here.
//
// Cannot remove an iter_arg in-place: getInitArgsMutable().erase() only drops
// the operand but leaves the region block argument and loop result intact,
// producing an inconsistent op that fails the verifier.  We clone instead.
// ---------------------------------------------------------------------------
static void removeUnusedIterArgChain(ArrayRef<BlockArgument> acc_iter_args,
                                     ArrayRef<Value> alloc_vals) {
  // `current` starts as the innermost iter_args, one per accumulator, and
  // advances to their inits at each level. All of them belong to the same loop
  // at every level, so a level is rebuilt once without all of their slots --
  // rebuilding it once per accumulator would leave the others' block arguments
  // pointing at a loop that is gone.
  SmallVector<Value> current(acc_iter_args.begin(), acc_iter_args.end());
  while (!current.empty() && isa<BlockArgument>(current.front())) {
    auto for_op = cast<scf::ForOp>(
        cast<BlockArgument>(current.front()).getOwner()->getParentOp());

    llvm::SmallDenseSet<unsigned> dropped;
    SmallVector<Value> next;
    for (auto [k, value] : llvm::enumerate(current)) {
      auto iter_arg = cast<BlockArgument>(value);

      // Block arg 0 of an scf.for is the induction variable; iter_args start
      // at index 1, so subtract 1 to get the iter_arg slot index.
      const unsigned iter_idx = iter_arg.getArgNumber() - 1;
      dropped.insert(iter_idx);

      // The init is what this accumulator advances to next.
      next.push_back(for_op.getInits()[iter_idx]);

      // Replace all uses of the dropped iter_arg and its loop result.
      for_op.getResult(iter_idx).replaceAllUsesWith(alloc_vals[k]);
      iter_arg.replaceAllUsesWith(alloc_vals[k]);
    }

    // Rebuild the loop without the dropped iter_args.
    OpBuilder builder(for_op);
    Location loc = for_op.getLoc();

    SmallVector<Value> new_inits;
    for (auto [i, init] : llvm::enumerate(for_op.getInits()))
      if (!dropped.contains(i)) new_inits.push_back(init);

    auto new_for =
        scf::ForOp::create(builder, loc, for_op.getLowerBound(),
                           for_op.getUpperBound(), for_op.getStep(), new_inits);

    for (auto attr : for_op->getAttrs())
      if (attr.getName() != "operandSegmentSizes")
        new_for->setAttr(attr.getName(), attr.getValue());

    // Map old to new iter args, skipping the dropped ones.
    IRMapping body_map;
    body_map.map(for_op.getInductionVar(), new_for.getInductionVar());
    unsigned new_arg_idx = 0;
    for (auto [i, arg] : llvm::enumerate(for_op.getRegionIterArgs()))
      if (!dropped.contains(i))
        body_map.map(arg, new_for.getRegionIterArgs()[new_arg_idx++]);

    // Copy old scf.for yield operands, skipping the dropped ones.
    auto* old_yield = for_op.getBody()->getTerminator();
    Operation* new_yield = new_for.getBody()->getTerminator();
    OpBuilder body_builder(new_yield);
    for (auto& op : for_op.getBody()->without_terminator())
      body_builder.clone(op, body_map);

    SmallVector<Value> new_yield_operands;
    for (auto [i, operand] : llvm::enumerate(old_yield->getOperands()))
      if (!dropped.contains(i))
        new_yield_operands.push_back(body_map.lookupOrDefault(operand));
    new_yield->setOperands(new_yield_operands);

    unsigned new_res_idx = 0;
    for (auto [i, res] : llvm::enumerate(for_op.getResults()))
      if (!dropped.contains(i))
        res.replaceAllUsesWith(new_for.getResult(new_res_idx++));

    // Advance to the next parent loop in the iter arg chain.
    current = next;
    for_op.erase();
  }
}

// ---------------------------------------------------------------------------
// Emit a new ktdf.read_from_fifo with a memref result type that mirrors the
// tensor-typed input at position 0 of `generic_op`.
//
// Asserts that `generic_op` has exactly one input and that the input is
// defined by a ktdf.read_from_fifo — no other producer is supported.
// ---------------------------------------------------------------------------
static Value convertInputToMemref(OpBuilder& builder,
                                  linalg::GenericOp generic_op) {
  assert(generic_op.getInputs().size() == 1 &&
         "convertInputToMemref: expected exactly one input on linalg.generic");
  Value input = generic_op.getInputs()[0];
  auto orig_read = input.getDefiningOp<ktdf::ReadFromFifoOp>();
  assert(orig_read &&
         "convertInputToMemref: input[0] must be a ktdf.read_from_fifo");

  auto in_tensor_type = cast<RankedTensorType>(input.getType());
  auto in_memref_type = MemRefType::get(in_tensor_type.getShape(),
                                        in_tensor_type.getElementType());
  return ktdf::ReadFromFifoOp::create(builder, generic_op.getLoc(),
                                      in_memref_type, orig_read.getFifoSlot())
      .getResult();
}

// ---------------------------------------------------------------------------
// Lower a single reduction linalg.generic to buffer semantics.
// `generic_op` must have at least one reduction iterator and its input must
// be a ktdf.read_from_fifo.
// ---------------------------------------------------------------------------
static LogicalResult rewriteGeneric(
    linalg::GenericOp generic_op,
    scheduler::arch_view::GroupLocalMemory& group_local_mem) {
  auto stage = generic_op->getParentOfType<ktdf::StageOp>();
  assert(stage && "expected enclosing ktdf.stage");

  // Step 1: decide which memory kind backs the accumulator.
  Attribute mem_space = group_local_mem.getLocalMemoryKindForStage(stage);
  if (!mem_space) return failure();

  OpBuilder builder(generic_op);
  Location loc = generic_op.getLoc();

  // One accumulator per result: a compute that accumulates more than one thing
  // has one of each, and each is buffered on its own.
  const unsigned accumulators =
      static_cast<unsigned>(generic_op.getOutputs().size());

  // Step 2: allocate an accumulator memref per result at the top of the stage.
  builder.setInsertionPointToStart(stage.getBody());
  SmallVector<Value> allocs;
  for (unsigned r = 0; r < accumulators; ++r) {
    auto out_tensor_type = cast<RankedTensorType>(
        generic_op.getDpsInitOperand(r)->get().getType());
    auto alloc_type = MemRefType::get(out_tensor_type.getShape(),
                                      out_tensor_type.getElementType(),
                                      MemRefLayoutAttrInterface{}, mem_space);
    allocs.push_back(
        memref::AllocOp::create(builder, loc, alloc_type).getResult());
  }

  // Step 3: capture the iter_args and walk up to find the outermost initializer
  // of each, before anything is erased.
  SmallVector<BlockArgument> acc_iter_args;
  SmallVector<Value> outermost_inits;
  for (unsigned r = 0; r < accumulators; ++r) {
    auto acc_iter_arg = dyn_cast<BlockArgument>(generic_op.getOutputs()[r]);
    if (!acc_iter_arg ||
        !isa<scf::ForOp>(acc_iter_arg.getOwner()->getParentOp())) {
      return generic_op.emitError(
                 "rewriteGeneric: every output must be an iter_arg of an "
                 "scf.for, and output ")
             << r << " is not";
    }
    acc_iter_args.push_back(acc_iter_arg);

    Value outermost_init;
    Value cursor = acc_iter_arg;
    while (auto ba = dyn_cast<BlockArgument>(cursor)) {
      auto for_op = cast<scf::ForOp>(ba.getOwner()->getParentOp());
      assert(for_op);
      outermost_init = for_op.getInits()[ba.getArgNumber() - 1];
      cursor = outermost_init;
    }
    assert(outermost_init && "could not find iter_arg initializer");
    outermost_inits.push_back(outermost_init);
  }

  // Step 4: replace all uses of each outermost initializer (e.g. the scf.for
  // init operand) with its alloc *before* lowering it, so that when
  // lowerIterArgInitializer erases the defining op no uses remain.
  for (unsigned r = 0; r < accumulators; ++r) {
    outermost_inits[r].replaceAllUsesWith(allocs[r]);
  }

  // Step 4b: lower each initializer — emit linalg.fill (and/or memref.copy) in
  // the right place and clean up tensor ops.
  for (unsigned r = 0; r < accumulators; ++r) {
    if (failed(
            lowerIterArgInitializer(outermost_inits[r], allocs[r], generic_op)))
      return failure();
  }

  // Step 5: emit a new ktdf.read_from_fifo with a memref result type so the
  // buffer-semantics linalg.generic below has a pure-buffer input.
  builder.setInsertionPoint(generic_op);
  Value new_read = convertInputToMemref(builder, generic_op);

  // Step 6: pure-buffer linalg.generic — memref ins + memref outs, no result.
  auto buf_generic = linalg::GenericOp::create(
      builder, loc,
      /*resultTensorTypes=*/TypeRange{},
      /*inputs=*/ValueRange{new_read},
      /*outputs=*/allocs, generic_op.getIndexingMapsAttr(),
      generic_op.getIteratorTypesAttr(),
      /*doc=*/StringAttr{},
      /*library_call=*/StringAttr{});
  IRMapping mapping;
  generic_op.getRegion().cloneInto(&buf_generic.getRegion(), mapping);
  // cloneInto prepends an empty placeholder block; drop it, keep the clone.
  Block& placeholder = buf_generic.getRegion().front();
  if (&placeholder != &buf_generic.getRegion().back()) placeholder.erase();

  // Step 7: replace generic result with alloc, patch write_to_fifo users,
  // then erase the generic and its now-dead tensor read_from_fifo input.
  for (unsigned r = 0; r < accumulators; ++r) {
    Value generic_result = generic_op.getResult(r);
    for (Operation* user : generic_result.getUsers())
      if (auto write_op = dyn_cast<ktdf::WriteToFifoOp>(user))
        write_op.getDataMutable().assign(allocs[r]);
    generic_result.replaceAllUsesWith(allocs[r]);
  }

  Value orig_input = generic_op.getInputs()[0];
  generic_op.erase();
  if (orig_input.use_empty())
    if (auto* orig_input_op = orig_input.getDefiningOp())
      orig_input_op->erase();

  // Step 8: walk up the iter_arg chain and rebuild each scf.for without it.
  // The initializer has already been transformed; no erasure is needed here.
  removeUnusedIterArgChain(acc_iter_args, allocs);
  return success();
}

// ---------------------------------------------------------------------------
// Set `new_type` on a PrivateOp result and its corresponding inner value
// (the private_yield operand at the same index).
//
// Always called with a direct PrivateOp result; the inner value is derived
// by indexing into private_yield.
//
// Example:
//   %2#2 = ktdf.private -> (!ktdf.fifo.slot<"A"->"B", 4xf16>, ...)
//     → widenPrivateResult(%2#2, !ktdf.fifo.slot<"A"->"B", 32xf16>)
//   %2#3 = ktdf.private -> (memref<4xf16, ms>, ...)
//     → widenPrivateResult(%2#3, memref<2x16xf16, ms>)
// ---------------------------------------------------------------------------
static void widenPrivateResult(Value priv_res, Type new_type) {
  auto priv_op = cast<ktdf::PrivateOp>(cast<OpResult>(priv_res).getOwner());
  auto result_idx = cast<OpResult>(priv_res).getResultNumber();
  Value inner = priv_op.getYieldOp().getOperand(result_idx);
  inner.setType(new_type);
  priv_res.setType(new_type);
}

// ---------------------------------------------------------------------------
// Iterate the direct uses of `fifo_slot` and for each data_transfer that
// references it update the FIFO-side static size field to reflect `new_shape`.
// ---------------------------------------------------------------------------
static void widenFifoUses(Value fifo_slot, ArrayRef<int64_t> new_shape,
                          MLIRContext* ctx) {
  auto new_sizes_attr = DenseI64ArrayAttr::get(ctx, new_shape);

  for (Operation* user : fifo_slot.getUsers()) {
    auto xfer = dyn_cast<ktdf::DataTransferOp>(user);
    if (!xfer) continue;

    // Only update the FIFO-side size on this transfer.  The alloc side (its
    // size, map, and type) is left entirely unchanged — the alloc's shape is
    // determined by its own allocation context and is shared with other
    // transfers (e.g. the store-out to global memory) that must not be
    // disturbed.
    if (xfer.isSourceFifo())
      xfer.setStaticSourceSizesAttr(new_sizes_attr);
    else if (xfer.isDestFifo())
      xfer.setStaticDestSizesAttr(new_sizes_attr);
  }
}

// ---------------------------------------------------------------------------
// Lower an inner-dim reduction linalg.generic to buffer semantics.
//
// Two sub-cases are handled:
//
// ── Case A: alloc_in is a local accumulator from rewriteGeneric ──────────
//
// At this point (after rewriteGeneric has run on G1) the stage contains:
//
//   %result = linalg.generic ins(%alloc_G1: memref<D0x...xDNxf16, ms>)
//                            outs(%empty:   tensor<D0x...xDN-1xf16>)
//   ktdf.write_to_fifo %result, %fifo_out
//
// %alloc_G1 is already a memref (RAUW'd by rewriteGeneric).  We reuse it as
// both input and output buffer:
//
//   ins  = %alloc_G1 (full memref<D0x...xDNxf16>)
//   outs = a rank-reducing subview of %alloc_G1 with size=1 on all reduction
//          dims (rank-reduced away) and original size on parallel dims.  This
//          gives a strided view into alloc_G1 where reduced results are
//          written back in-place.
//
// After the inner reduction, the full %alloc_G1 buffer — not the subview —
// is sent to the FIFO so the downstream stage receives all partial sums.
// The FIFO slot type and the paired memref.alloc in the ktdf.private block are
// widened to match %alloc_G1's shape.
//
// Example (memref<2x64xf16, ms>, reduction over dim 1):
//
//   %sv = memref.subview %alloc_G1[0, 0][2, 1][1, 1]
//             : memref<2x64xf16, ms> to memref<2xf16, strided<[64]>, ms>
//   linalg.generic ins(%alloc_G1) outs(%sv)
//   ktdf.write_to_fifo %alloc_G1, %fifo_out    // full buffer, not subview
//
//   // ktdf.private widened (flat count 2 → 128, shape 2xf16 → 2x64xf16):
//   %fifo = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"A" -> "B", 128xf16>
//   %acc  = memref.alloc() : memref<2x64xf16, ct_local>
//
// ── Case B: alloc_in is a ktdf.read_from_fifo (no outer-dim loop) ────────
//
// When there is no outer-dim reduction loop, rewriteGeneric is never called.
// alloc_in is the memref result of a ktdf.read_from_fifo.
//
// We allocate ONE local buffer (local_reg), copy the FIFO data into it, and
// use it as BOTH ins and the subview source:
//
//   %local_reg = memref.alloc() : memref<D0x...xDNxf16, "REG">
//   memref.copy %fifo_read_memref, %local_reg
//   %sv = memref.subview %local_reg[0,...][..., 1, ...][...]
//   linalg.generic ins(%local_reg) outs(%sv)   // both refer to local_reg
//   ktdf.write_to_fifo %local_reg, %fifo_out
// ---------------------------------------------------------------------------
static LogicalResult rewriteInnerDimGeneric(
    linalg::GenericOp generic_op,
    scheduler::arch_view::GroupLocalMemory& group_local_mem) {
  auto stage = generic_op->getParentOfType<ktdf::StageOp>();
  assert(stage && "expected enclosing ktdf.stage");

  OpBuilder builder(generic_op);
  Location loc = generic_op.getLoc();

  // ins[0] is a memref — either the outer-dim alloc from rewriteGeneric, or a
  // memref-typed read_from_fifo emitted by the caller when no outer-dim loop
  // exists. One input per accumulator: a compute reducing more than one thing
  // has an intermediate of each, and each half reduces into its own buffer.
  // They are the same type, so one set of offsets and sizes describes every
  // subview.
  SmallVector<Value> allocs_in(generic_op.getInputs());
  Value alloc_in = allocs_in.front();
  auto in_memref_type = cast<MemRefType>(alloc_in.getType());
  unsigned in_rank = in_memref_type.getRank();
  if (llvm::any_of(allocs_in,
                   [&](Value in) { return in.getType() != in_memref_type; })) {
    return generic_op.emitError(
        "rewriteInnerDimGeneric: the inputs are not all the same memref type");
  }
  if (generic_op.getOutputs().size() != allocs_in.size()) {
    return generic_op.emitError(
        "rewriteInnerDimGeneric: expected an output per input");
  }

  // Inspect indexing maps and iterator types to classify each dim.
  auto iter_types = generic_op.getIteratorTypesArray();
  auto indexing_maps = generic_op.getIndexingMapsArray();
  AffineMap in_map = indexing_maps.front();
  AffineMap out_map = indexing_maps.back();
  ArrayRef<int64_t> in_shape = in_memref_type.getShape();

  // Build a loop-dim → input-dimension-size lookup from the input map so we
  // can resolve sizes for dims that appear in the output map but not in a
  // direct positional correspondence to in_shape.
  SmallVector<int64_t> loop_dim_to_size(iter_types.size(), 1);
  for (unsigned d = 0; d < in_rank; ++d) {
    auto dim_expr = dyn_cast<AffineDimExpr>(in_map.getResult(d));
    assert(dim_expr && "inner-dim input map must be identity-like");
    loop_dim_to_size[dim_expr.getPosition()] = in_shape[d];
  }

  // SubViewOp requires offsets/sizes/strides arrays of length == source rank
  // (in_rank).  The rank reduction is expressed by placing
  // size=1 on reduction dimensions; inferRankReducedResultType then produces
  // the correctly strided result type with the reduced rank.
  unsigned out_rank = out_map.getNumResults();

  // Build source-rank-sized arrays by iterating over the input map: each
  // source dim d maps to loop dim loop_dim; if that loop dim is a reduction
  // iterator set size=1 (to be dropped), otherwise keep the full size.
  SmallVector<OpFoldResult> sv_offsets(in_rank, builder.getIndexAttr(0));
  SmallVector<OpFoldResult> sv_sizes_ofr(in_rank);
  SmallVector<OpFoldResult> sv_strides(in_rank, builder.getIndexAttr(1));
  for (unsigned d = 0; d < in_rank; ++d) {
    auto dim_expr = dyn_cast<AffineDimExpr>(in_map.getResult(d));
    assert(dim_expr && "inner-dim input map must be identity-like");
    unsigned loop_dim = dim_expr.getPosition();
    int64_t sz = (iter_types[loop_dim] == utils::IteratorType::reduction)
                     ? 1
                     : loop_dim_to_size[loop_dim];
    sv_sizes_ofr[d] = builder.getIndexAttr(sz);
  }

  // Rank-reduced result shape: parallel dims only, in out_map order, so the
  // subview result rank matches the output indexing map.
  SmallVector<int64_t> sv_result_shape;
  for (unsigned d = 0; d < out_rank; ++d) {
    auto dim_expr = dyn_cast<AffineDimExpr>(out_map.getResult(d));
    assert(dim_expr && "inner-dim output map must be identity-like");
    unsigned loop_dim = dim_expr.getPosition();
    if (iter_types[loop_dim] != utils::IteratorType::reduction)
      sv_result_shape.push_back(loop_dim_to_size[loop_dim]);
  }

  // Case B: alloc_in is a FIFO-read buffer (no outer-dim loop).
  //
  // ktdf.opaque requires a constant-address operand (read_from_fifo fails
  // that check).  Allocate a local register per accumulator, copy the FIFO data
  // into each, and use them as both ins and subview sources.
  //
  // One each rather than one shared: two accumulations cannot be reduced
  // straight out of the fifo, since each element's own contribution still has
  // to be worked out, so the second would overwrite the first.
  SmallVector<Value> effective(allocs_in);
  if (allocs_in.front().getDefiningOp<ktdf::ReadFromFifoOp>()) {
    Attribute mem_space = group_local_mem.getLocalMemoryKindForStage(stage);
    if (!mem_space) return failure();

    auto alloc_type = MemRefType::get(in_shape, in_memref_type.getElementType(),
                                      MemRefLayoutAttrInterface{}, mem_space);

    for (auto [r, in] : llvm::enumerate(allocs_in)) {
      // Each alloc serves as both ins and subview source for its accumulator.
      builder.setInsertionPointToStart(stage.getBody());
      Value local_reg =
          memref::AllocOp::create(builder, loc, alloc_type).getResult();

      // Copy its FIFO data into it immediately before the generic.
      builder.setInsertionPoint(generic_op);
      memref::CopyOp::create(builder, loc, in, local_reg);
      effective[r] = local_reg;
    }
  }

  // Let SubViewOp infer the correct strided layout from the subview source.
  auto sv_result_type =
      cast<MemRefType>(memref::SubViewOp::inferRankReducedResultType(
          sv_result_shape, cast<MemRefType>(effective.front().getType()),
          sv_offsets, sv_sizes_ofr, sv_strides));

  builder.setInsertionPoint(generic_op);
  SmallVector<Value> subviews;
  for (Value in : effective) {
    subviews.push_back(memref::SubViewOp::create(builder, loc, sv_result_type,
                                                 in, sv_offsets, sv_sizes_ofr,
                                                 sv_strides)
                           .getResult());
  }

  // Buffer linalg.generic: ins = the full buffers, outs = a rank-reduced
  // subview of each. In Case A those are the outer-dim accumulators; in Case B
  // they are the local registers copied from the fifo. Original maps and
  // iterator_types are preserved.
  auto buf_generic = linalg::GenericOp::create(
      builder, loc,
      /*resultTensorTypes=*/TypeRange{},
      /*inputs=*/effective,
      /*outputs=*/subviews, generic_op.getIndexingMapsAttr(),
      generic_op.getIteratorTypesAttr(),
      /*doc=*/StringAttr{},
      /*library_call=*/StringAttr{});
  IRMapping mapping;
  generic_op.getRegion().cloneInto(&buf_generic.getRegion(), mapping);
  Block& placeholder = buf_generic.getRegion().front();
  if (&placeholder != &buf_generic.getRegion().back()) placeholder.erase();

  // Patch each write_to_fifo to send the whole buffer its result reduced into
  // rather than the subview, and capture the slot it wrote to, whose type is
  // widened below.
  SmallVector<Value> old_fifo_slots;
  for (auto [r, in] : llvm::enumerate(effective)) {
    Value generic_result = generic_op.getResult(r);
    for (Operation* user :
         llvm::make_early_inc_range(generic_result.getUsers())) {
      if (auto write_op = dyn_cast<ktdf::WriteToFifoOp>(user)) {
        old_fifo_slots.push_back(write_op.getFifoSlot());
        write_op.getDataMutable().assign(in);
      }
    }
    generic_result.replaceAllUsesWith(subviews[r]);
  }

  // Widen each FIFO slot type and all downstream allocs/transfers that use
  // it. The new element count is the product of all input dimensions.
  for (Value old_fifo_slot : old_fifo_slots) {
    auto old_slot_type = cast<ktdf::FifoSlotType>(old_fifo_slot.getType());
    Type elem_type = old_slot_type.getElementType();

    int64_t new_num_elems = 1;
    for (int64_t sz : in_shape) new_num_elems *= sz;

    auto new_slot_type = ktdf::FifoSlotType::get(
        generic_op.getContext(), old_slot_type.getSrc(),
        old_slot_type.getDest(), new_num_elems, elem_type);
    // Widen the PrivateOp result for the fifo slot (and its inner allocate).
    widenPrivateResult(old_fifo_slot, new_slot_type);

    // Update the FIFO-side size on all data_transfer uses of the fifo slot.
    // The paired alloc in ktdf.private is intentionally left unchanged.
    widenFifoUses(old_fifo_slot, in_shape, generic_op.getContext());
  }

  // Erase the original tensor.empty output initializers and the tensor
  // generic.
  SmallVector<Value> orig_out_inits(generic_op.getOutputs());
  generic_op.erase();
  for (Value init : orig_out_inits) {
    if (!init.use_empty()) continue;
    if (auto* def = init.getDefiningOp()) def->erase();
  }

  return success();
}

struct MapReductionPartialsPass
    : public ktdf::impl::MapReductionPartialsPassBase<
          MapReductionPartialsPass> {
  using MapReductionPartialsPassBase<
      MapReductionPartialsPass>::MapReductionPartialsPassBase;

  void runOnOperation() override {
    if (DisableThisPass) return;
    LDBG(1) << "========= " PASS_NAME " =========";
    ModuleOp module = getOperation();

    auto& device_manager = getAnalysis<mlir::ktdf_arch::DeviceManager>();
    auto* const device = device_manager.getOrImportDevice();
    if (!device) {
      module->emitError("Unable to import the device specification.");
      signalPassFailure();
      return;
    }
    auto& group_local_mem =
        device_manager.getOrCreateView<scheduler::arch_view::GroupLocalMemory>(
            *device);

    // Collect generics in two buckets.  Loop-exposed (outer-dim) generics
    // must be processed first so their loop results are RAUW'd to alloc
    // memrefs before the inner-dim generics are processed (which expect
    // ins[0] to already be a memref).
    SmallVector<linalg::GenericOp> loop_exposed, inner_dim;
    module.walk([&](linalg::GenericOp generic_op) {
      if (!hasReductionIterator(generic_op)) return;
      if (generic_op.getOutputs().empty()) return;
      auto iter_arg = dyn_cast<BlockArgument>(generic_op.getOutputs()[0]);
      if (iter_arg && isa<scf::ForOp>(iter_arg.getOwner()->getParentOp()))
        loop_exposed.push_back(generic_op);
      else
        inner_dim.push_back(generic_op);
    });

    for (auto generic_op : loop_exposed) {
      if (failed(rewriteGeneric(generic_op, group_local_mem))) {
        signalPassFailure();
        return;
      }
    }
    for (auto generic_op : inner_dim) {
      // ins[0] is a tensor-typed ktdf.read_from_fifo when there was no
      // outer-dim loop feeding this generic (i.e. rewriteGeneric did not run
      // on its pipeline).  Emit a memref-typed read in its place so that
      // rewriteInnerDimGeneric can assume ins[0] is already a memref.
      Operation* stale_tensor_read = nullptr;
      if (generic_op.getInputs()[0].getDefiningOp<ktdf::ReadFromFifoOp>()) {
        OpBuilder builder(generic_op);
        stale_tensor_read = generic_op.getInputs()[0].getDefiningOp();
        Value new_read = convertInputToMemref(builder, generic_op);
        generic_op.getInputsMutable().assign(new_read);
      }
      // Transform inner-dim generic into memref-typed generic.
      if (failed(rewriteInnerDimGeneric(generic_op, group_local_mem))) {
        signalPassFailure();
        return;
      }
      if (stale_tensor_read && stale_tensor_read->use_empty())
        stale_tensor_read->erase();
    }
  }
};

}  // namespace

auto mlir::ktdf::createMapReductionPartialsPass() -> std::unique_ptr<Pass> {
  return std::make_unique<MapReductionPartialsPass>();
}
