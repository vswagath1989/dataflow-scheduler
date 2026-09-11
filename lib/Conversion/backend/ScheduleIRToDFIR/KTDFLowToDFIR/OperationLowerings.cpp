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
/// Operation lowerings for KTDFLowToDFIR pass
///
//
//===----------------------------------------------------------------------===//

#include "dataflow-scheduler/Conversion/backend/ScheduleIRToDFIR/KTDFLowToDFIR/OperationLowerings.h"

#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/StringExtras.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/Matchers.h>
#include <mlir/IR/OperationSupport.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Interfaces/ViewLikeInterface.h>
#include <mlir/Transforms/DialectConversion.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

#include "dataflow-scheduler/Analysis/Utils.h"
#include "dataflow-scheduler/Conversion/Utils/Utils.h"
#include "dataflow-scheduler/Conversion/backend/ScheduleIRToDFIR/KTDFLowToDFIR/BufferPhaseLowering.h"
#include "dataflow-scheduler/Conversion/backend/ScheduleIRToDFIR/KTDFLowToDFIR/DataTransferLowering.h"
#include "dataflow-scheduler/Conversion/backend/ScheduleIRToDFIR/KTDFLowToDFIR/LinalgLowering.h"
#include "dataflow-scheduler/Conversion/backend/ScheduleIRToDFIR/KTDFLowToDFIR/ParallelLowering.h"
#include "dataflow-scheduler/Conversion/backend/ScheduleIRToDFIR/KTDFLowToDFIR/Utils.h"
#include "dataflow-scheduler/Conversion/backend/ScheduleIRToDFIR/KTDFToKTDFLow/UniformInfra.h"
#include "dataflow-scheduler/Dialect/Agen/Agen.h"   // IWYU pragma: keep
#include "dataflow-scheduler/Dialect/Agen/Utils.h"  // IWYU pragma: keep
#include "dataflow-scheduler/Dialect/Dataflow/Dataflow.h"
#include "dataflow-scheduler/Dialect/Dataflow/DataflowDialect.h"  // IWYU pragma: keep
#include "dataflow-scheduler/Dialect/Dataflow/Utils.h"
#include "dataflow-scheduler/Dialect/KTDF/KTDF.h"
#include "dataflow-scheduler/Dialect/KTDFArch/Analysis/ResourceKinds.h"
#include "dataflow-scheduler/Dialect/KTDFArch/KTDFArchAttributes.h"
#include "dataflow-scheduler/Dialect/KTDFLowering/KTDFLowering.h"
#include "dataflow-scheduler/Dialect/Uniform/Uniform.h"
#include "dataflow-scheduler/Dialect/VectorChain/VectorChain.h"
#include "dataflow-scheduler/Utils/SchedulerExtContext.h"

#define DEBUG_TYPE "ktdflowering-to-dfir"

using namespace scheduler;

namespace {

/// Pattern to lower ktdf.read_from_fifo operations
struct LowerReadFromFifoPattern
    : public mlir::OpRewritePattern<mlir::ktdf::ReadFromFifoOp> {
  LowerReadFromFifoPattern(mlir::MLIRContext* context,
                           mlir::ktdf_arch::ResourceKinds& resource_kinds,
                           const ResourceToUnits& components)
      : OpRewritePattern(context),
        resource_kinds_(resource_kinds),
        components_(components) {}

  mlir::LogicalResult matchAndRewrite(
      mlir::ktdf::ReadFromFifoOp read_op,
      mlir::PatternRewriter& rewriter) const override {
    // Get the FIFO slot type
    auto fifo_slot_type =
        llvm::cast<mlir::ktdf::FifoSlotType>(read_op.getFifoSlot().getType());

    // FIXME: Discover compute from op.
    auto compute = resource_kinds_.getDefaultCompute();
    if (!compute) {
      return llvm::failure();
    }

    // Convert result type (tensor or memref) to flattened vector type
    auto vector_type = getFlattenedVectorType(read_op.getType(), compute);
    if (!vector_type) {
      return mlir::failure();
    }

    // Find the enclosing program_unit
    auto program_unit =
        read_op->getParentOfType<mlir::dataflow::ProgramUnitOp>();
    if (!program_unit) {
      read_op.emitError("read_from_fifo must be inside a program_unit");
      return mlir::failure();
    }

    // Resolve the source unit from the FIFO src attribute
    auto queried_unit_result = resolveUnitFromFifoAttr(
        fifo_slot_type.getSrc(), components_, rewriter, program_unit,
        read_op.getLoc(), read_op.getOperation());
    if (mlir::failed(queried_unit_result)) {
      return mlir::failure();
    }
    mlir::Value queried_unit = *queried_unit_result;

    // Create dataflow.receive operation
    auto receive_op = mlir::dataflow::ReceiveOp::create(
        rewriter, read_op.getLoc(), vector_type, queried_unit,
        /*dbgName=*/nullptr);

    // If the FIFO was filled by a splat data_transfer, the send side widened
    // its load to the arch's minimum access granularity and shuffled to the
    // full vector width.  The receive side mirrors that with a matching
    // shuffle.
    //
    // read_with_splat = true is set by a pre-pass walk that runs before
    // buildProgramUnits in KTDFLowToDFIRPass::runOnOperation (see
    // KTDFLowToDFIR.cpp), so we read it directly here.
    mlir::Value result = receive_op.getData();
    auto splat_attr = read_op->getDiscardableAttr("read_with_splat");
    bool is_splat =
        splat_attr && mlir::cast<mlir::BoolAttr>(splat_attr).getValue();

    if (is_splat) {
      // The enclosing program_unit names the compute unit.  The
      // shuffle granularity on the receive side must come from that unit's
      // SIMD spec (sub_simd_lanes / shuffle_modes).
      mlir::Attribute compute_kind =
          getEnclosingProgramUnitResourceType(read_op).value_or(
              mlir::Attribute{});

      int64_t fifo_elements = vector_type.getNumElements();
      auto sub_simd_or_err = computeSplatSubSimdElements(
          fifo_elements, vector_type.getElementType(), compute_kind,
          resource_kinds_, read_op.getOperation());
      if (mlir::failed(sub_simd_or_err)) return mlir::failure();
      int64_t granularity_elements = sub_simd_or_err.value();

      // Re-express the splat on the received vector.  The receive already
      // holds a full fifo_elements-wide value.  The send side loaded
      // granularity_elements values and broadcast them; the receive mirrors
      // that by emitting a shuffle with granularity_elements all-zero indices
      // (each pointing at element 0 of the received vector) repeated until
      // the full width is covered:
      //
      //   indices    = [0, 0, ..., 0]  (length = granularity_elements)
      //   repetition = fifo_elements / granularity_elements
      //
      // Example: fifo_elements=64, granularity_elements=8 →
      //   indices = [0,0,0,0,0,0,0,0], repetition = 8
      if (granularity_elements > 1 && granularity_elements <= fifo_elements &&
          fifo_elements % granularity_elements == 0) {
        llvm::SmallVector<mlir::Attribute> index_attrs(
            granularity_elements,
            rewriter.getIntegerAttr(rewriter.getI32Type(), 0));
        int32_t repetition =
            static_cast<int32_t>(fifo_elements / granularity_elements);
        result = mlir::vectorchain::ShuffleOp::create(
                     rewriter, read_op.getLoc(), vector_type, result,
                     /*variable=*/mlir::ValueRange{},
                     /*pad=*/mlir::ValueRange{},
                     /*mask=*/nullptr, /*dbgName=*/nullptr,
                     rewriter.getArrayAttr(index_attrs),
                     rewriter.getI32IntegerAttr(repetition))
                     .getOutput();
      }
    }

    // Replace the read_from_fifo with the (possibly shuffled) result
    rewriter.replaceOp(read_op, result);

    return mlir::success();
  }

 private:
  mlir::ktdf_arch::ResourceKinds& resource_kinds_;
  const ResourceToUnits& components_;
};

struct LowerWriteToFifoPattern
    : public mlir::OpRewritePattern<mlir::ktdf::WriteToFifoOp> {
  LowerWriteToFifoPattern(mlir::MLIRContext* context,
                          mlir::ktdf_arch::ResourceKinds& resource_kinds,
                          const ResourceToUnits& components)
      : OpRewritePattern(context),
        resource_kinds_(resource_kinds),
        components_(components) {}

  mlir::LogicalResult matchAndRewrite(
      mlir::ktdf::WriteToFifoOp write_op,
      mlir::PatternRewriter& rewriter) const override {
    // Get the FIFO slot type
    auto fifo_slot_type =
        llvm::cast<mlir::ktdf::FifoSlotType>(write_op.getFifoSlot().getType());

    // FIXME: Discover compute from op.
    auto compute = resource_kinds_.getDefaultCompute();
    if (!compute) {
      return write_op.emitError(
          "the architecture declares no default compute resource");
    }

    // Convert data type (tensor or vector) to flattened vector type
    auto vector_type =
        getFlattenedVectorType(write_op.getData().getType(), compute);
    if (!vector_type) {
      return mlir::failure();
    }

    // Find the enclosing program_unit
    auto program_unit =
        write_op->getParentOfType<mlir::dataflow::ProgramUnitOp>();
    if (!program_unit) {
      write_op.emitError("write_to_fifo must be inside a program_unit");
      return mlir::failure();
    }

    // Resolve the destination unit from the FIFO dest attribute
    auto queried_unit_result = resolveUnitFromFifoAttr(
        fifo_slot_type.getDest(), components_, rewriter, program_unit,
        write_op.getLoc(), write_op.getOperation());
    if (mlir::failed(queried_unit_result)) {
      return mlir::failure();
    }
    mlir::Value queried_unit = *queried_unit_result;

    // If the data operand is a memref buffer), load it into a vector first.
    mlir::Value send_data = write_op.getData();
    if (mlir::isa<mlir::MemRefType>(send_data.getType())) {
      send_data =
          emitVectorLoad(rewriter, write_op.getLoc(), vector_type, send_data);
    }

    // Create dataflow.send operation
    mlir::dataflow::SendOp::create(rewriter, write_op.getLoc(), queried_unit,
                                   send_data, /*dir=*/nullptr,
                                   /*dbgName=*/nullptr);

    // Erase the write_to_fifo operation
    rewriter.eraseOp(write_op);

    return mlir::success();
  }

 private:
  mlir::ktdf_arch::ResourceKinds& resource_kinds_;
  const ResourceToUnits& components_;
};

/// Helper to find the current unit's query_map from signal operands
/// Returns the query_map whose first mapping result matches a program_unit
/// operand
static llvm::FailureOr<mlir::Value> findCurrentUnitQueryMap(
    llvm::ArrayRef<mlir::Value> signal_units,
    llvm::ArrayRef<mlir::Value> program_unit_operands, mlir::Operation* op) {
  for (auto signal_unit : signal_units) {
    // Get the query_map operation
    auto query_op = signal_unit.getDefiningOp<mlir::uniform::QueryMapOp>();
    if (!query_op) {
      op->emitError("signal operand must be a uniform.query_map result");
      return mlir::failure();
    }

    // Get the def_immutable_mapping
    auto def_map_op =
        query_op.getMap().getDefiningOp<mlir::uniform::DefImmutableMappingOp>();
    if (!def_map_op) {
      op->emitError("query_map must reference a def_immutable_mapping");
      return mlir::failure();
    }

    // Get the first value from the mapping (first result unit)
    auto values = def_map_op.getValues();
    if (values.empty()) {
      op->emitError("def_immutable_mapping must have at least one value");
      return mlir::failure();
    }
    mlir::Value first_unit = values[0];

    // Check if this first unit is in the current program_unit operands
    for (auto pu_unit : program_unit_operands) {
      if (first_unit == pu_unit) {
        return signal_unit;
      }
    }
  }

  op->emitError(
      "signal must include at least one query_map whose first mapping result "
      "is in the current program_unit");
  return mlir::failure();
}
/// Pattern to lower ktdf.tiling.derive_size operations using conditional
/// branching
struct LowerGetTileSizePattern
    : public mlir::OpRewritePattern<mlir::ktdf::TilingDeriveSizeOp> {
  LowerGetTileSizePattern(mlir::MLIRContext* context,
                          const SchedulerExtContext& /*scheduler_ctx*/,
                          const ResourceToUnits& /*components*/)
      : OpRewritePattern(context) {}

  mlir::LogicalResult matchAndRewrite(
      mlir::ktdf::TilingDeriveSizeOp derive_size_op,
      mlir::PatternRewriter& rewriter) const override {
    auto loc = derive_size_op.getLoc();
    auto ivs = derive_size_op.getIvs();
    auto tile_sizes = derive_size_op.getTileSizes();
    auto total_size = derive_size_op.getTotalSize();

    // We only handle the single-level case (one [iv : tile_size] pair)
    if (ivs.size() != 1) {
      return mlir::failure();
    }

    auto iv = ivs[0];
    auto tile_size = tile_sizes[0];

    // Find the enclosing scf.for loop that uses this IV
    auto iv_block_arg = mlir::dyn_cast<mlir::BlockArgument>(iv);
    if (!iv_block_arg) {
      derive_size_op.emitError("IV must be a block argument");
      return mlir::failure();
    }

    auto for_op = iv_block_arg.getOwner()->getParentOp();
    auto scf_for = mlir::dyn_cast<mlir::scf::ForOp>(for_op);

    if (!scf_for) {
      derive_size_op.emitError("IV must be from an scf.for loop");
      return mlir::failure();
    }
    // Extract constant values - both must be constants
    auto tile_size_const =
        tile_size.getDefiningOp<mlir::arith::ConstantIndexOp>();
    auto total_size_const =
        total_size.getDefiningOp<mlir::arith::ConstantIndexOp>();

    if (!tile_size_const || !total_size_const) {
      derive_size_op.emitError("tile_size and total_size must be constants");
      return mlir::failure();
    }

    int64_t tile_size_val = tile_size_const.value();
    int64_t total_size_val = total_size_const.value();

    // Epilogue size is the remainder; zero means total_size divides tile_size
    // evenly and every iteration — including the last — uses the steady-state
    // tile size.
    int64_t epilogue_size_val = total_size_val % tile_size_val;

    mlir::Value result;
    if (epilogue_size_val == 0) {
      // No epilogue: every tile has the steady-state size; no conditional
      // needed.
      result = tile_size;
    } else {
      // Epilogue exists: last iteration gets epilogue_size, others get
      // tile_size.
      auto upper_bound = scf_for.getUpperBound();
      auto c1 = mlir::arith::ConstantIndexOp::create(rewriter, loc, 1);
      auto upper_bound_minus_1 =
          mlir::arith::SubIOp::create(rewriter, loc, upper_bound, c1);
      auto is_not_last = mlir::arith::CmpIOp::create(
          rewriter, loc, mlir::arith::CmpIPredicate::slt, iv,
          upper_bound_minus_1);
      auto epilogue_size = mlir::arith::ConstantIndexOp::create(
          rewriter, loc, epilogue_size_val);
      result = mlir::arith::SelectOp::create(rewriter, loc, is_not_last,
                                             tile_size, epilogue_size);
    }

    rewriter.replaceOp(derive_size_op, result);

    return mlir::success();
  }
};

/// Pattern to lower ktdf_lowering.signal operations
struct LowerSignalPattern
    : public mlir::OpRewritePattern<mlir::ktdf_lowering::SignalOp> {
  LowerSignalPattern(mlir::MLIRContext* context,
                     const SchedulerExtContext& /*scheduler_ctx*/,
                     const ResourceToUnits& /*components*/)
      : OpRewritePattern(context) {}

  mlir::LogicalResult matchAndRewrite(
      mlir::ktdf_lowering::SignalOp signal_op,
      mlir::PatternRewriter& rewriter) const override {
    // Signal operations must have at least 2 units
    if (signal_op.getNumUnits() < 2) {
      signal_op.emitError(
          "signal operation must have at least 2 unit operands");
      return mlir::failure();
    }

    // Find the enclosing program_unit
    auto program_unit =
        signal_op->getParentOfType<mlir::dataflow::ProgramUnitOp>();
    if (!program_unit) {
      signal_op.emitError("signal must be inside a program_unit");
      return mlir::failure();
    }

    auto signal_units = signal_op.getUnits();
    auto program_unit_operands = program_unit.getUnits();

    // Convert to vectors for the helper function
    llvm::SmallVector<mlir::Value, 8> signal_units_vec(signal_units.begin(),
                                                       signal_units.end());
    llvm::SmallVector<mlir::Value, 8> program_unit_vec(
        program_unit_operands.begin(), program_unit_operands.end());

    // Find which signal operand corresponds to the current program_unit
    auto curr_unit_query_map_result = findCurrentUnitQueryMap(
        signal_units_vec, program_unit_vec, signal_op.getOperation());
    if (mlir::failed(curr_unit_query_map_result)) {
      return mlir::failure();
    }
    mlir::Value curr_unit_query_map = *curr_unit_query_map_result;

    auto curr_resource_type =
        scheduler::getUnitTypeFromQueryMap(curr_unit_query_map);
    assert(!curr_resource_type.empty() && "getUnitTypeFromQueryMap failed");

    // Collect peer signal units: different resource type from the current unit.
    llvm::SmallVector<mlir::Value, 8> other_signal_units;
    for (auto signal_unit : signal_units) {
      if (signal_unit == curr_unit_query_map) continue;
      auto other_resource_type =
          scheduler::getUnitTypeFromQueryMap(signal_unit);
      assert(!other_resource_type.empty() && "getUnitTypeFromQueryMap failed");
      if (other_resource_type != curr_resource_type)
        other_signal_units.push_back(signal_unit);
    }

    // Decide which lowering path to use.
    //
    // Corelet-aware path: both the current PU's units AND the peer units carry
    // corelet attributes.  In this case, all peer operands of the same resource
    // type must be grouped into a single query map so that each corelet of the
    // current unit communicates only with the matching corelet of the peer:
    //   [pu_CL0 -> peer_CL0,  pu_CL1 -> peer_CL1]  queried by iter_arg.
    //
    // Other path: at least one side has no corelet attribute.  Fall through
    // to buildSignalQueryMap (core-index matching), one entry per peer operand.

    // A PU is genuinely multi-corelet only when its units carry more than one
    // distinct corelet value (e.g. CL0 and CL1).  Units that have corelet=0
    // set by the scheduler but no real second corelet all share the same value,
    // so the distinct-count check correctly returns false for them.
    bool curr_has_corelets = false;
    {
      llvm::DenseSet<int> corelet_ids;
      for (mlir::Value u : program_unit.getUnits()) {
        auto gu = mlir::dyn_cast_or_null<mlir::dataflow::GetUnitOp>(
            u.getDefiningOp());
        if (gu) corelet_ids.insert(mlir::dataflow::getCoreletId(gu));
      }
      curr_has_corelets = corelet_ids.size() > 1;
    }

    // A peer resource type is genuinely multi-corelet only when the get_unit
    // values it maps to carry more than one distinct corelet value across all
    // operands of that type.  Multiple operands due to multiple cores (all with
    // corelet=0) do not qualify — the distinct-corelet-value check handles
    // that. type_corelet_ids maps each peer resource type to the set of
    // distinct corelet values seen across all signal operands of that type.
    bool peers_have_corelets = false;
    {
      llvm::DenseMap<mlir::StringAttr, llvm::DenseSet<int>> type_corelet_ids;
      for (mlir::Value su : other_signal_units) {
        auto rt = scheduler::getUnitResourceType(su);
        if (!rt) continue;
        auto key = mlir::cast<mlir::StringAttr>(*rt);
        auto qop = su.getDefiningOp<mlir::uniform::QueryMapOp>();
        auto dop =
            qop ? qop.getMap()
                      .getDefiningOp<mlir::uniform::DefImmutableMappingOp>()
                : nullptr;
        if (!dop) continue;
        for (mlir::Value val : dop.getValues()) {
          auto gu = mlir::dyn_cast_or_null<mlir::dataflow::GetUnitOp>(
              val.getDefiningOp());
          if (gu)
            type_corelet_ids[key].insert(mlir::dataflow::getCoreletId(gu));
        }
      }
      for (auto& kv : type_corelet_ids) {
        if (kv.second.size() > 1) {
          peers_have_corelets = true;
          break;
        }
      }
    }

    llvm::SmallVector<mlir::Value, 8> other_query_maps;

    if (curr_has_corelets && peers_have_corelets) {
      // --- Corelet-aware path ---
      // The signal contains one query_map per corelet of each peer resource
      // type (e.g. %30=l1su-CL0, %32=l1su-CL1 for a two-corelet peer).
      // We must NOT lower each one to a separate sync_send/recv — that would
      // make every corelet signal every peer corelet.  Instead, group the peer
      // operands by resource type, then build one merged query_map per group
      // that dispatches to the matching corelet at runtime:

      // Bucket peer operands by their resource type.
      llvm::SmallVector<
          std::pair<mlir::StringAttr, llvm::SmallVector<mlir::Value, 4>>, 4>
          by_resource;
      for (mlir::Value su : other_signal_units) {
        auto key =
            mlir::cast<mlir::StringAttr>(*scheduler::getUnitResourceType(su));
        auto it =
            llvm::find_if(by_resource, [&](auto& p) { return p.first == key; });
        if (it == by_resource.end())
          by_resource.push_back({key, {su}});
        else
          it->second.push_back(su);
      }

      // iter_arg (%arg0) is the program_unit's block argument: at runtime it
      // holds the get_unit value of whichever corelet is currently executing.
      // It is used as the query key so each corelet resolves to its own peer.
      mlir::Value iter_arg = program_unit.getRegion().front().getArgument(0);

      for (auto& [rtype, group] : by_resource) {
        // For each peer signal operand in this group, extract corelet ids and
        // get_unit results from its def_immutable_mapping. An operand may cover
        // one corelet (single-value map) or all corelets (multi-value map).
        llvm::SmallVector<std::pair<int, mlir::Value>, 4> cl_to_peer;
        for (mlir::Value su : group) {
          auto qop = su.getDefiningOp<mlir::uniform::QueryMapOp>();
          auto dop = qop.getMap()
                         .getDefiningOp<mlir::uniform::DefImmutableMappingOp>();
          for (mlir::Value val : dop.getValues()) {
            auto peer_gu =
                mlir::cast<mlir::dataflow::GetUnitOp>(val.getDefiningOp());
            cl_to_peer.push_back(
                {mlir::dataflow::getCoreletId(peer_gu), peer_gu.getResult(0)});
          }
        }

        // Check that the peer group's corelets fully cover every PU corelet.
        // If not (e.g. the peer is a single unit whose corelet attribute is 0
        // but which doesn't actually mirror the PU's multi-corelet layout),
        // the per-corelet pairing is undefined — fall back to
        // buildSignalQueryMap (core-index matching) for each operand in the
        // group instead.
        bool group_covers_all_pu_corelets = true;
        for (mlir::Value pu_op : program_unit.getUnits()) {
          auto pu_gu =
              mlir::cast<mlir::dataflow::GetUnitOp>(pu_op.getDefiningOp());
          int cl = mlir::dataflow::getCoreletId(pu_gu);
          if (!llvm::any_of(cl_to_peer,
                            [cl](auto& p) { return p.first == cl; })) {
            group_covers_all_pu_corelets = false;
            break;
          }
        }

        if (!group_covers_all_pu_corelets) {
          for (mlir::Value signal_unit : group) {
            auto result = UniformInfra::buildSignalQueryMap(
                signal_unit, program_unit, rewriter, signal_op.getLoc());
            if (mlir::failed(result)) {
              signal_op.emitError(
                  "failed to build query map for signal operand");
              return mlir::failure();
            }
            other_query_maps.push_back(*result);
          }
          continue;
        }

        // Build the merged mapping keys (current PU units) and values (peer
        // units at the same corelet index).
        // e.g. keys=[l1lu-CL0, l1lu-CL1], values=[l1su-CL0, l1su-CL1]
        llvm::SmallVector<mlir::Value> new_keys, new_values;
        for (mlir::Value pu_op : program_unit.getUnits()) {
          auto pu_gu =
              mlir::cast<mlir::dataflow::GetUnitOp>(pu_op.getDefiningOp());
          int cl = mlir::dataflow::getCoreletId(pu_gu);
          auto it = llvm::find_if(cl_to_peer,
                                  [cl](auto& p) { return p.first == cl; });
          new_keys.push_back(pu_op);
          new_values.push_back(it->second);
        }

        // Emit the merged def_immutable_mapping + query_map.
        // At runtime, querying with iter_arg returns the peer corelet unit
        // that corresponds to the currently-executing corelet of this PU.
        auto new_map = mlir::uniform::DefImmutableMappingOp::create(
            rewriter, signal_op.getLoc(), rewriter.getIndexType(), new_keys,
            new_values);
        auto new_query = mlir::uniform::QueryMapOp::create(
            rewriter, signal_op.getLoc(), rewriter.getIndexType(),
            new_map.getResult(), iter_arg);
        other_query_maps.push_back(new_query.getResult());
      }
    } else {
      // --- at least one set of units has no corelets ---
      // buildSignalQueryMap rebuilds the query_map substituting PU operands as
      // keys (core-index matching).
      for (mlir::Value signal_unit : other_signal_units) {
        auto result = UniformInfra::buildSignalQueryMap(
            signal_unit, program_unit, rewriter, signal_op.getLoc());
        if (mlir::failed(result)) {
          signal_op.emitError("failed to build query map for signal operand");
          return mlir::failure();
        }
        other_query_maps.push_back(*result);
      }
    }

    // Create all sync_send operations (from current unit to other units)
    bool wait_immediately_for_async_transfer = true;
    for (mlir::Value other_query_map : other_query_maps) {
      mlir::dataflow::SyncSendOp::create(
          rewriter, signal_op.getLoc(), other_query_map, /*dbgName=*/nullptr,
          rewriter.getBoolAttr(wait_immediately_for_async_transfer));
    }

    // Create all sync_recv operations (from other units to current unit)
    for (mlir::Value other_query_map : other_query_maps) {
      mlir::dataflow::SyncRecvOp::create(rewriter, signal_op.getLoc(),
                                         other_query_map, /*dbgName=*/nullptr);
    }

    // Erase the signal operation
    rewriter.eraseOp(signal_op);

    return mlir::success();
  }
};

/// Lower a memref.copy that MapReductionPartials emitted.
///
/// Two sources reach here. A fifo read is already a value, so it is stored
/// straight into the destination:
///   %r  = ktdf.read_from_fifo ... -> memref<...>
///   memref.copy %r, %dest
///
/// A buffer is not, so it is loaded first -- which is how a reduction's
/// accumulator is copied from the buffer holding its identity:
///   memref.copy %identity, %accumulator
///
/// Runs at higher benefit (2) than FoldEmptyCopy (1, a canonicalization
/// pattern) so it fires first, preventing FoldEmptyCopy from crashing on
/// opaque memory-space attributes.
/// Lowers the copy to agen.vector_store of the source value into the dest.
struct LowerMemRefCopyFromFifoPattern
    : public mlir::OpRewritePattern<mlir::memref::CopyOp> {
  LowerMemRefCopyFromFifoPattern(
      mlir::MLIRContext* context,
      mlir::ktdf_arch::ResourceKinds& /*resource_kinds*/)
      : OpRewritePattern(context, /*benefit=*/2) {}

  mlir::LogicalResult matchAndRewrite(
      mlir::memref::CopyOp copy_op,
      mlir::PatternRewriter& rewriter) const override {
    rewriter.setInsertionPoint(copy_op);
    const mlir::Location loc = copy_op.getLoc();

    mlir::Value stored = copy_op.getSource();
    if (!stored.getDefiningOp<mlir::ktdf::ReadFromFifoOp>()) {
      const auto buffer = llvm::dyn_cast<mlir::MemRefType>(stored.getType());
      if (!buffer) {
        return rewriter.notifyMatchFailure(
            copy_op, "source is neither a fifo read nor a memref");
      }
      if (!buffer.hasStaticShape()) {
        return rewriter.notifyMatchFailure(
            copy_op, "source has no static shape to load as one vector");
      }
      stored = emitVectorLoad(rewriter, loc,
                              mlir::VectorType::get({buffer.getNumElements()},
                                                    buffer.getElementType()),
                              stored);
    }

    emitVectorStore(rewriter, loc, stored, copy_op.getTarget());
    rewriter.eraseOp(copy_op);
    return mlir::success();
  }
};

struct LowerOpaquePattern : mlir::OpRewritePattern<mlir::ktdf::OpaqueOp> {
  static constexpr llvm::StringLiteral kRegisterNamesAttrName =
      "dataflow_scheduler.register_names";

  struct NoneAttr : mlir::TypeAttr {
    [[nodiscard]] static auto classof(Attribute attr) -> bool {
      const auto typed = llvm::dyn_cast<TypeAttr>(attr);
      return classof(typed);
    }
    [[nodiscard]] static auto classof(TypeAttr attr) -> bool {
      return llvm::isa<mlir::NoneType>(attr.getValue());
    }

    using TypeAttr::TypeAttr;
  };

  struct RegisterNameAttr : mlir::Attribute {
    [[nodiscard]] static auto classof(Attribute attr) -> bool {
      return llvm::isa<mlir::StringAttr, NoneAttr>(attr);
    }
    [[nodiscard]] static auto classof(mlir::StringAttr /*attr*/) -> bool {
      return true;
    }
    [[nodiscard]] static auto classof(NoneAttr /*attr*/) -> bool {
      return true;
    }

    using Attribute::Attribute;

    using ValueType = mlir::StringAttr;

    [[nodiscard]] auto getValue() const -> mlir::StringAttr {
      return llvm::dyn_cast<mlir::StringAttr>(*this);
    }
  };

  using RegisterNamesAttr = mlir::ktdf_arch::TypedArrayAttr<RegisterNameAttr>;

  explicit LowerOpaquePattern(mlir::MLIRContext* context)
      : OpRewritePattern(context) {
    register_names_attr_name_ =
        mlir::StringAttr::get(context, kRegisterNamesAttrName);
  }

  auto matchAndRewrite(mlir::ktdf::OpaqueOp opaque,
                       mlir::PatternRewriter& rewriter) const
      -> mlir::LogicalResult override {
    if (!opaque.hasPureBufferSemantics()) {
      // We can only lower operations that only use buffers, since there is
      // no way to materialize the values into memories at this point.
      return rewriter.notifyMatchFailure(opaque,
                                         "operation has pure semantics");
    }

    // Map the register names to IDs.
    mlir::NamedAttrList read_only;
    mlir::NamedAttrList read_write;
    if (failed(mapRegisters(opaque, read_only, read_write))) {
      return rewriter.notifyMatchFailure(opaque, "unable to map registers");
    }

    // Extract the inherent attributes values from the discardable dict.
    mlir::NamedAttrList attributes(opaque->getRawDictionaryAttrs());
    mlir::OperationName op_name(mlir::dataflow::OpaqueOp::getOperationName(),
                                rewriter.getContext());

    // func_name
    const auto func_name =
        llvm::dyn_cast_if_present<mlir::StringAttr>(attributes.erase(
            mlir::dataflow::OpaqueOp::getFuncNameAttrName(op_name)));
    if (!func_name) {
      return rewriter.notifyMatchFailure(opaque,
                                         "missing 'func_name' attribute");
    }
    // read_only_register_dictionary
    if (const auto attr =
            llvm::dyn_cast_if_present<mlir::DictionaryAttr>(attributes.erase(
                mlir::dataflow::OpaqueOp::getReadOnlyRegisterDictionaryAttrName(
                    op_name)))) {
      read_only.append(attr);
    }
    // read_write_register_dictionary
    if (const auto attr = llvm::dyn_cast_if_present<
            mlir::DictionaryAttr>(attributes.erase(
            mlir::dataflow::OpaqueOp::getReadWriteRegisterDictionaryAttrName(
                op_name)))) {
      read_write.append(attr);
    }
    // parameter_dictionary
    mlir::NamedAttrList parameters;
    if (const auto attr =
            llvm::dyn_cast_if_present<mlir::DictionaryAttr>(attributes.erase(
                mlir::dataflow::OpaqueOp::getParameterDictionaryAttrName(
                    op_name)))) {
      parameters.append(attr);
    }

    auto df_opaque = rewriter.replaceOpWithNewOp<mlir::dataflow::OpaqueOp>(
        opaque, mlir::StringAttr{}, func_name,
        read_write.getDictionary(rewriter.getContext()),
        read_only.getDictionary(rewriter.getContext()),
        parameters.getDictionary(rewriter.getContext()));
    // Apply all the remaining discardable attributes.
    df_opaque->setDiscardableAttrs(attributes);
    return llvm::success();
  }

 private:
  [[nodiscard]] auto getRegisterNames(mlir::ktdf::OpaqueOp opaque) const
      -> RegisterNamesAttr {
    const auto register_names = llvm::dyn_cast_if_present<RegisterNamesAttr>(
        opaque->getDiscardableAttr(register_names_attr_name_));
    if (!register_names) {
      return nullptr;
    }

    // [inputs, ..., outputs, ...]
    const auto num_expected =
        opaque.getInputs().size() + opaque.getOutputs().size();
    if (register_names.size() != num_expected) {
      opaque->emitError("number of register names (")
          << register_names.size()
          << ") does not match number of operands and results (" << num_expected
          << ")";
      return nullptr;
    }

    return register_names;
  }

  /// Gets the address \p value is viewed at, in elements, when it is constant.
  ///
  /// Taken off the view rather than from address assignment, so it is in the
  /// granularity the view was built in: elements for a register file, and
  /// nullopt for anything the walk cannot reduce to a constant.
  [[nodiscard]] static auto getConstantElementAddress(mlir::Value value)
      -> std::optional<size_t> {
    while (auto* const definition = value.getDefiningOp()) {
      mlir::IntegerAttr constant;
      if (mlir::m_Constant(&constant).match(definition)) {
        return constant.getValue().getZExtValue();
      }

      if (auto view = llvm::dyn_cast<mlir::dataflow::GetLogicalMemoryViewOp>(
              definition);
          view) {
        value = view.getStartAddress();
        continue;
      }

      // Unknown source.
      break;
    }

    return std::nullopt;
  }

  [[nodiscard]] auto mapRegisters(mlir::ktdf::OpaqueOp opaque,
                                  mlir::NamedAttrList& read_only,
                                  mlir::NamedAttrList& read_write) const
      -> mlir::LogicalResult {
    const auto register_names = getRegisterNames(opaque);
    if (!register_names) {
      return llvm::failure();
    }

    llvm::SmallString<4> register_id_buffer;

    // [inputs, ..., outputs, ...]
    const auto values =
        llvm::concat<mlir::Value>(opaque.getInputs(), opaque.getOutputs());
    for (auto [maybe_name, value] :
         llvm::zip_equal(register_names.getAsValueRange(), values)) {
      if (!maybe_name) {
        continue;
      }

      // The address passed in this slot must be a constant.
      const auto address = getConstantElementAddress(value);
      if (!address) {
        opaque.emitError("slot ")
            << value << " does not have a constant address";
        return llvm::failure();
      }

      // FIXME: Query this from the arch graph (per memory space).
      const auto register_size = 128;

      // An address counts elements, so the divisor is how many of them a
      // register holds rather than its size in bytes. Dividing by the bytes
      // instead lands a narrower element element_bytes registers too far
      // along.
      const auto shaped = llvm::dyn_cast<mlir::ShapedType>(value.getType());
      if (!shaped) {
        opaque.emitError("slot ") << value << " is not a buffer";
        return llvm::failure();
      }
      const auto maybe_size = tryGetSizeInBytes(shaped.getElementType());
      if (!maybe_size || register_size % *maybe_size != 0) {
        opaque.emitError("slot ")
            << value << " element does not divide a register of "
            << register_size << " bytes";
        return llvm::failure();
      }
      const auto register_elements = register_size / *maybe_size;

      // Compute the index of the register addressed by the operand.
      if (*address % register_elements != 0) {
        opaque.emitError("slot ")
            << value << " address (" << *address
            << " elements) is not a multiple of the register size ("
            << register_elements << " elements)";
        return llvm::failure();
      }
      const auto index = *address / register_elements;

      // Create the register ID string and put it into the map.
      // FIXME: Is it OK to assume "Rn" as the naming scheme here?
      register_id_buffer.clear();
      const auto register_id = mlir::StringAttr::get(
          getContext(), llvm::Twine("R")
                            .concat(llvm::Twine(index))
                            .toStringRef(register_id_buffer));

      // Add the register ID to the correct map.
      if (llvm::is_contained(opaque.getOutputs(), value)) {
        read_write.set(maybe_name, register_id);
      } else {
        read_only.set(maybe_name, register_id);
      }
    }

    return llvm::success();
  }

  mlir::StringAttr register_names_attr_name_;
};

}  // namespace

mlir::LogicalResult scheduler::runOperationLowerings(
    mlir::func::FuncOp func,
    const scheduler::SchedulerExtContext& scheduler_ctx,
    const ResourceToUnits& components,
    mlir::ktdf_arch::ResourceKinds& resource_kinds, SymbolAllocator& symbols) {
  // Lower linalg.generic compute operations and FIFO operations
  mlir::RewritePatternSet patterns(func.getContext());
  populateLinalgLoweringPatterns(patterns, resource_kinds, symbols);
  patterns.add<LowerMemRefCopyFromFifoPattern>(func.getContext(),
                                               resource_kinds);
  patterns.add<LowerReadFromFifoPattern>(func.getContext(), resource_kinds,
                                         components);
  patterns.add<LowerWriteToFifoPattern>(func.getContext(), resource_kinds,
                                        components);
  populateDataTransferLoweringPatterns(patterns, components, resource_kinds);
  patterns.add<LowerSignalPattern>(func.getContext(), scheduler_ctx,
                                   components);
  patterns.add<LowerGetTileSizePattern>(func.getContext(), scheduler_ctx,
                                        components);
  patterns.add<LowerOpaquePattern>(patterns.getContext());
  if (mlir::failed(mlir::applyPatternsGreedily(func, std::move(patterns)))) {
    return mlir::failure();
  }

  // Clone loops for buffer phase tracking and replace operations
  if (mlir::failed(lowerDoubleBuffering(func, components, resource_kinds))) {
    return mlir::failure();
  }

  // Lower ktdf.parallel operations after all other lowerings are complete
  if (mlir::failed(lowerParallelOps(func))) {
    return mlir::failure();
  }

  // After all lowering helpers have run, no ktdf or ktdf_lowering op should
  // survive. Use an empty conversion with both dialects marked illegal so any
  // survivor produces an explicit error rather than silently passing through.
  mlir::ConversionTarget target(*func.getContext());
  target.addIllegalDialect<mlir::ktdf::KTDFDialect,
                           mlir::ktdf_lowering::KTDFLoweringDialect>();
  target.markUnknownOpDynamicallyLegal([](mlir::Operation*) { return true; });
  if (mlir::failed(mlir::applyPartialConversion(
          func, target,
          mlir::FrozenRewritePatternSet(
              mlir::RewritePatternSet(func.getContext()))))) {
    return mlir::failure();
  }

  return mlir::success();
}

// Made with Bob
