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

#include "dataflow-scheduler/Conversion/backend/ScheduleIRToDFIR/KTDFLowToDFIR/Utils.h"

#include <mlir/IR/BuiltinTypeInterfaces.h>

#include "dataflow-scheduler/Conversion/Utils/Utils.h"
#include "dataflow-scheduler/Dialect/Agen/Agen.h"
#include "dataflow-scheduler/Dialect/Dataflow/Dataflow.h"
#include "dataflow-scheduler/Dialect/Dataflow/Utils.h"
#include "dataflow-scheduler/Dialect/KTDF/Utils/Utils.h"
#include "dataflow-scheduler/Dialect/KTDFArch/Analysis/ResourceKinds.h"
#include "dataflow-scheduler/Dialect/KTDFArch/KTDFArch.h"
#include "dataflow-scheduler/Dialect/KTDFArch/KTDFArchIntrinsics.h"
#include "dataflow-scheduler/Dialect/Uniform/Uniform.h"
#include "dataflow-scheduler/Dialect/VectorChain/VectorChain.h"
#include "dataflow-scheduler/Utils/SchedulerExtContext.h"
#include "ktir/Dialect/KTDP/KTDP.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/IntegerSet.h"
#include "mlir/IR/PatternMatch.h"

using namespace scheduler;

std::optional<scheduler::ResourceType>
scheduler::getEnclosingProgramUnitResourceType(mlir::Operation* op) {
  auto pu = op->getParentOfType<mlir::dataflow::ProgramUnitOp>();
  if (!pu || pu.getUnits().empty()) return std::nullopt;
  return scheduler::getUnitResourceType(pu.getUnits().front());
}

int64_t scheduler::getVectorLanes(mlir::Type elem_type,
                                  mlir::ktdf_arch::ExecutionUnitOp compute) {
  return std::max(
      compute.getFeature<mlir::ktdf_arch::feature::SIMD>().getLanes(elem_type),
      static_cast<int64_t>(1));
}

mlir::IntegerSet scheduler::buildIntegerSetFromSizes(
    mlir::MLIRContext* ctx, llvm::ArrayRef<int64_t> sizes) {
  llvm::SmallVector<mlir::AffineExpr> exprs;
  llvm::SmallVector<bool> eq_flags;
  for (unsigned i = 0; i < sizes.size(); ++i) {
    auto dim = mlir::getAffineDimExpr(i, ctx);
    int64_t size = sizes[i];
    if (size == 1) {
      exprs.push_back(dim);
      eq_flags.push_back(/*equality=*/true);
    } else {
      exprs.push_back(dim);
      eq_flags.push_back(false);
      exprs.push_back(mlir::getAffineConstantExpr(size - 1, ctx) - dim);
      eq_flags.push_back(false);
    }
  }
  return mlir::IntegerSet::get(sizes.size(), 0, exprs, eq_flags);
}

mlir::Value scheduler::emitVectorLoad(mlir::OpBuilder& builder,
                                      mlir::Location loc,
                                      mlir::VectorType vec_type,
                                      mlir::Value memref) {
  auto memref_type = mlir::cast<mlir::MemRefType>(memref.getType());
  unsigned rank = memref_type.getRank();
  mlir::MLIRContext* ctx = builder.getContext();
  auto map = mlir::AffineMap::getMultiDimIdentityMap(rank, ctx);
  auto load_set = buildIntegerSetFromSizes(ctx, memref_type.getShape());
  llvm::SmallVector<mlir::Value> zero_indices(
      rank, mlir::arith::ConstantIndexOp::create(builder, loc, 0).getResult());
  return mlir::agen::VectorLoadOp::create(builder, loc, vec_type, memref,
                                          /*dbgName=*/nullptr, map,
                                          zero_indices, load_set, map)
      .getResult();
}

void scheduler::emitVectorStore(mlir::OpBuilder& builder, mlir::Location loc,
                                mlir::Value value, mlir::Value memref) {
  auto memref_type = mlir::cast<mlir::MemRefType>(memref.getType());
  unsigned rank = memref_type.getRank();
  mlir::MLIRContext* ctx = builder.getContext();
  auto map = mlir::AffineMap::getMultiDimIdentityMap(rank, ctx);
  auto store_set = buildIntegerSetFromSizes(ctx, memref_type.getShape());
  llvm::SmallVector<mlir::Value> zero_indices(
      rank, mlir::arith::ConstantIndexOp::create(builder, loc, 0).getResult());
  mlir::agen::VectorStoreOp::create(builder, loc, value, memref,
                                    /*dbgName=*/nullptr, map, zero_indices,
                                    store_set, map);
}

mlir::VectorType scheduler::getFlattenedVectorType(
    mlir::ShapedType type, mlir::ktdf_arch::ExecutionUnitOp compute) {
  if (!type.hasStaticShape()) {
    return nullptr;
  }

  const auto total_elements = type.getNumElements();
  const auto max_vector_length = getVectorLanes(type.getElementType(), compute);
  assert(total_elements <= max_vector_length &&
         "Flattened tensor/memref size exceeds maximum vector length");

  return mlir::VectorType::get({total_elements}, type.getElementType());
}

mlir::Value scheduler::createQueryMapForComponent(
    mlir::OpBuilder& builder, mlir::dataflow::ProgramUnitOp program_unit,
    const llvm::SmallVector<mlir::Value, 4>& target_units, mlir::Location loc) {
  mlir::ValueRange pu_operands = program_unit.getUnits();
  mlir::Block& body = program_unit.getRegion().front();
  mlir::Value iter_arg = body.getArgument(0);

  llvm::SmallVector<mlir::Value> keys;
  llvm::SmallVector<mlir::Value> values;

  for (mlir::Value pu_op : pu_operands) {
    auto get_unit = mlir::dyn_cast_or_null<mlir::dataflow::GetUnitOp>(
        pu_op.getDefiningOp());
    assert(get_unit &&
           "program_unit operand must be defined by dataflow.get_unit");
    auto core_attr = get_unit->getAttrOfType<mlir::IntegerAttr>("core");
    assert(core_attr && "dataflow.get_unit must have 'core' attribute");
    int core = static_cast<int>(core_attr.getInt());
    int corelet = mlir::dataflow::getCoreletId(get_unit);

    mlir::Value matching_target;
    for (mlir::Value target : target_units) {
      auto target_get_unit = mlir::dyn_cast_or_null<mlir::dataflow::GetUnitOp>(
          target.getDefiningOp());
      assert(target_get_unit &&
             "target unit must be defined by dataflow.get_unit");
      auto target_core_attr =
          target_get_unit->getAttrOfType<mlir::IntegerAttr>("core");
      assert(target_core_attr &&
             "target dataflow.get_unit must have 'core' attribute");
      if (static_cast<int>(target_core_attr.getInt()) != core) continue;
      // When the source unit has a corelet, the target must match it so that
      // each src corelet maps to its own dst corelet (not always corelet 0).
      int target_corelet = mlir::dataflow::getCoreletId(target_get_unit);
      if (corelet >= 0 && target_corelet != corelet) continue;
      matching_target = target;
      break;
    }

    assert(matching_target &&
           "must find matching target unit for each program_unit operand");
    keys.push_back(pu_op);
    values.push_back(matching_target);
  }

  assert(!keys.empty() && "must have at least one key-value pair");

  auto map_op = mlir::uniform::DefImmutableMappingOp::create(
      builder, loc, builder.getIndexType(), keys, values);
  auto query_op = mlir::uniform::QueryMapOp::create(
      builder, loc, builder.getIndexType(), map_op.getResult(), iter_arg);

  return query_op.getResult();
}

mlir::LogicalResult scheduler::replaceComputeTileIdWithCoreQuery(
    mlir::dataflow::ProgramUnitOp program_unit,
    llvm::DenseMap<int64_t, mlir::Value>& core_id_consts,
    mlir::OpBuilder& const_builder) {
  mlir::Region& region = program_unit.getRegion();
  mlir::Block& body = region.front();

  // Collect single-result get_compute_tile_id ops with a use inside this
  // program_unit's region. The op is typically defined outside the region (in
  // the enclosing function) and captured in. Walk the enclosing function body
  // (the program_unit's parent op) — this avoids needing ModuleOp and is the
  // scope where the captured tile-id is defined.
  llvm::SmallVector<mlir::ktdp::GetComputeTileIdOp> tile_ids;
  mlir::Operation* parent = program_unit->getParentOp();
  parent->walk([&](mlir::ktdp::GetComputeTileIdOp tid) {
    if (tid->getNumResults() != 1) return;  // single-result only
    for (mlir::OpOperand& use : tid->getResult(0).getUses()) {
      if (region.isAncestor(use.getOwner()->getParentRegion())) {
        tile_ids.push_back(tid);
        break;
      }
    }
  });
  if (tile_ids.empty()) return mlir::success();  // nothing to do for this unit

  // Build the core map: keys = program_unit operands, values = SHARED constant
  // flat core ids read from each operand get_unit's 'core' attribute. Constants
  // are materialized once at function scope (const_builder) and cached in
  // core_id_consts, then captured into this body (units are not
  // IsolatedFromAbove).
  mlir::OpBuilder builder(&body, body.begin());
  auto loc = program_unit.getLoc();
  llvm::SmallVector<mlir::Value> keys;
  llvm::SmallVector<mlir::Value> values;
  for (mlir::Value pu_op : program_unit.getUnits()) {
    auto get_unit = mlir::dyn_cast_or_null<mlir::dataflow::GetUnitOp>(
        pu_op.getDefiningOp());
    if (!get_unit) {
      return program_unit.emitError(
          "program_unit operand must be defined by dataflow.get_unit");
    }
    auto core_attr = get_unit->getAttrOfType<mlir::IntegerAttr>("core");
    if (!core_attr) {
      return program_unit.emitError(
          "dataflow.get_unit must have 'core' attribute");
    }
    int64_t core = core_attr.getInt();
    auto cached = core_id_consts.find(core);
    mlir::Value core_const;
    if (cached != core_id_consts.end()) {
      core_const = cached->second;
    } else {
      core_const =
          mlir::arith::ConstantIndexOp::create(const_builder, loc, core);
      core_id_consts[core] = core_const;
    }
    keys.push_back(pu_op);
    values.push_back(core_const);
  }

  auto map_op = mlir::uniform::DefImmutableMappingOp::create(
      builder, loc, builder.getIndexType(), keys, values);
  mlir::Value iter_arg = body.getArgument(0);
  auto query_op = mlir::uniform::QueryMapOp::create(
      builder, loc, builder.getIndexType(), map_op.getResult(), iter_arg);
  mlir::Value core_id = query_op.getResult();

  // NOTE: in-region uses include any NESTED program_unit bodies. This assumes
  // nested program_units share the same unit set (hence the same core) as
  // this outer unit, which holds in the current pipeline; a nested unit with
  // a different unit set would need its own per-unit core query instead.
  // Redirect every in-region use of each tile-id result to the query result.
  for (mlir::ktdp::GetComputeTileIdOp tid : tile_ids) {
    tid->getResult(0).replaceUsesWithIf(core_id, [&](mlir::OpOperand& use) {
      return region.isAncestor(use.getOwner()->getParentRegion());
    });
  }

  return mlir::success();
}

llvm::FailureOr<mlir::Value> scheduler::resolveUnitFromFifoAttr(
    mlir::Attribute fifo_attr, const ResourceToUnits& components,
    mlir::PatternRewriter& rewriter, mlir::dataflow::ProgramUnitOp program_unit,
    mlir::Location loc, mlir::Operation* op_for_errors) {
  // Step 1: cast the raw attribute to a StringAttr and upper-case it.
  auto str_attr = mlir::dyn_cast<mlir::StringAttr>(fifo_attr);
  if (!str_attr) {
    op_for_errors->emitError("unsupported FIFO endpoint attribute type");
    return mlir::failure();
  }
  scheduler::ResourceType component_type =
      mlir::StringAttr::get(str_attr.getContext(), str_attr.getValue().upper());

  // Step 2: look up the resource type in the components map.
  auto it = components.find(component_type);
  if (it == components.end()) {
    op_for_errors->emitError()
        << "no units found for FIFO component type: " << component_type;
    return mlir::failure();
  }

  // Step 3: build and return the query_map for that component.
  return createQueryMapForComponent(rewriter, program_unit, it->second, loc);
}

llvm::FailureOr<scheduler::DataTransferType> scheduler::getDataTransferType(
    bool src_is_fifo, bool dst_is_fifo) {
  // Case 1: Both source and destination are memrefs (memory to memory)
  if (!src_is_fifo && !dst_is_fifo) {
    return scheduler::DataTransferType::kLoadAndStore;
  }

  // Case 2: Source is memref, destination is FIFO slot
  if (!src_is_fifo && dst_is_fifo) {
    return scheduler::DataTransferType::kLoadAndSend;
  }

  // Case 3: Source is FIFO slot, destination is memref
  if (src_is_fifo && !dst_is_fifo) {
    return scheduler::DataTransferType::kReceiveAndStore;
  }

  // Both source and destination are FIFO slots - unsupported
  return llvm::failure();
}

mlir::Value scheduler::insertSplatShuffle(mlir::OpBuilder& builder,
                                          mlir::Location loc,
                                          mlir::Value src_vec,
                                          int64_t src_elements,
                                          int64_t dst_elements) {
  assert(src_elements > 0 && "splat source width must be positive");
  assert(dst_elements % src_elements == 0 &&
         "splat destination width must be a multiple of the source width");

  auto src_vec_type = mlir::cast<mlir::VectorType>(src_vec.getType());
  auto elem_type = src_vec_type.getElementType();

  llvm::SmallVector<mlir::Attribute> index_attrs;
  for (int64_t i = 0; i < src_elements; ++i) {
    index_attrs.push_back(builder.getIntegerAttr(builder.getI32Type(), i));
  }
  auto indices_attr = builder.getArrayAttr(index_attrs);
  int32_t repetition = static_cast<int32_t>(dst_elements / src_elements);
  auto result_type = mlir::VectorType::get({dst_elements}, elem_type);

  return mlir::vectorchain::ShuffleOp::create(
             builder, loc, result_type, src_vec,
             /*variable=*/mlir::ValueRange{}, /*pad=*/mlir::ValueRange{},
             /*mask=*/nullptr, /*dbgName=*/nullptr, indices_attr,
             builder.getI32IntegerAttr(repetition))
      .getOutput();
}

llvm::FailureOr<int64_t> scheduler::computeSplatSubSimdElements(
    int64_t dst_total_elements, mlir::Type elem_type, mlir::Attribute kind,
    const mlir::ktdf_arch::ResourceKinds& resource_kinds,
    mlir::Operation* op_for_errors) {
  if (!kind) return dst_total_elements;

  auto simd_feature =
      resource_kinds.getFeature<mlir::ktdf_arch::feature::SIMD>(kind);
  if (!simd_feature) return dst_total_elements;

  // Verify the compute unit declares FirstSubSimdLaneToEachSubSimd.
  auto shuffle_modes =
      simd_feature.getAttr<mlir::DictionaryAttr>("shuffle_modes");
  if (!shuffle_modes || !shuffle_modes.get("FirstSubSimdLaneToEachSubSimd")) {
    op_for_errors->emitError(
        "splat receive requires the compute unit to declare "
        "shuffle_modes = { FirstSubSimdLaneToEachSubSimd } in "
        "ktdf_arch.feature.simd");
    return mlir::failure();
  }

  // Read sub_simd_lanes for elem_type.
  auto sub_simd_lanes =
      simd_feature.getAttr<mlir::ktdf_arch::feature::SIMD::LanesAttr>(
          "sub_simd_lanes");
  if (!sub_simd_lanes) return dst_total_elements;

  int64_t lane_count =
      sub_simd_lanes.getValue(mlir::TypeAttr::get(elem_type)).value_or(0);
  if (lane_count <= 0) return dst_total_elements;

  return lane_count;
}

int64_t scheduler::computeSplatGranularityElements(
    int64_t src_total_elements, mlir::Type elem_type, mlir::Attribute kind,
    const mlir::ktdf_arch::ResourceKinds& resource_kinds) {
  if (!kind) return src_total_elements;

  auto load_feature =
      resource_kinds.getFeature<mlir::ktdf_arch::feature::Load>(kind);
  if (!load_feature) return src_total_elements;

  // The memory space is not available at lowering time (stripped by
  // buildLogicalMemoryViews).  Iterate all memory spaces declared in the Load
  // feature and return the smallest fitting granularity across all of them.
  auto tryWithSpace = [&](mlir::Attribute space) -> int64_t {
    auto granularity_list = load_feature.getAccessGranularity(space);
    if (!granularity_list) return src_total_elements;

    size_t elem_bytes =
        (static_cast<size_t>(elem_type.getIntOrFloatBitWidth()) + 7) / 8;
    size_t word_size = load_feature.getWordSize(space);
    if (word_size == 0) return src_total_elements;

    // elem_words: number of words one element occupies (always >= 1).
    size_t elem_words = (elem_bytes + word_size - 1) / word_size;

    // Minimum load size in words to cover all src elements.
    size_t min_words = static_cast<size_t>(src_total_elements) * elem_words;

    auto best = granularity_list.fitAccess(min_words);
    if (!best) return src_total_elements;

    int64_t load_elements =
        static_cast<int64_t>(best.getSizeInWords() / elem_words);
    return std::max(load_elements, src_total_elements);
  };

  auto gran_map = load_feature.getAccessGranularity();
  if (!gran_map) return src_total_elements;

  // Walk every declared memory space and keep the smallest fitting granularity
  // found across all of them.  The first result wins as the initial best;
  // subsequent results only replace it when they are strictly smaller.
  int64_t best_result = src_total_elements;
  bool found_any = false;
  for (auto [space_attr, _] : gran_map) {
    int64_t result = tryWithSpace(space_attr);
    if (!found_any || result < best_result) {
      best_result = result;
      found_any = true;
    }
  }
  return best_result;
}
