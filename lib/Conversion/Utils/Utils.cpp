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

#include "dataflow-scheduler/Conversion/Utils/Utils.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <utility>

#include "dataflow-scheduler/Dialect/Agen/Agen.h"
#include "dataflow-scheduler/Dialect/Dataflow/Dataflow.h"
#include "dataflow-scheduler/Dialect/KTDFArch/KTDFArch.h"
#include "dataflow-scheduler/Dialect/KTDFArch/KTDFArchIntrinsics.h"
#include "dataflow-scheduler/Dialect/Uniform/Uniform.h"
#include "dataflow-scheduler/Dialect/VectorChain/VectorChain.h"
#include "mlir/Dialect/Arith/IR/Arith.h"

namespace scheduler {

auto getUnitResourceType(mlir::Value unit_value)
    -> std::optional<scheduler::ResourceType> {
  // Direct dataflow.get_unit result.
  if (auto get_unit = unit_value.getDefiningOp<mlir::dataflow::GetUnitOp>()) {
    return mlir::StringAttr::get(unit_value.getContext(),
                                 get_unit.getType().upper());
  }

  // uniform.query_map → uniform.def_immutable_mapping → dataflow.get_unit.
  auto query_op = unit_value.getDefiningOp<mlir::uniform::QueryMapOp>();
  if (!query_op) return std::nullopt;

  auto def_mapping_op =
      query_op.getMap().getDefiningOp<mlir::uniform::DefImmutableMappingOp>();
  if (!def_mapping_op) return std::nullopt;

  auto values = def_mapping_op.getValues();
  if (values.empty()) return std::nullopt;

  auto get_unit_op = values.front().getDefiningOp<mlir::dataflow::GetUnitOp>();
  if (!get_unit_op) return std::nullopt;

  return mlir::StringAttr::get(unit_value.getContext(),
                               get_unit_op.getType().upper());
}

std::string getUnitTypeFromQueryMap(mlir::Value query_map) {
  auto resource_type = getUnitResourceType(query_map);
  if (!resource_type) {
    query_map.getDefiningOp()->emitError(
        "failed to determine unit resource type from query_map");
    return "";
  }
  return mlir::cast<mlir::StringAttr>(*resource_type).strref().str();
}

namespace {

/// Number of elements a single `access_granularity` entry of `space` covers
/// that still holds all `src_total_elements` source elements, or nullopt when
/// `space` declares nothing that fits.
std::optional<int64_t> fittingGranularityElements(
    mlir::ktdf_arch::feature::Load load_feature, mlir::Attribute space,
    int64_t src_total_elements, mlir::Type elem_type) {
  auto granularity_list = load_feature.getAccessGranularity(space);
  if (!granularity_list) return std::nullopt;

  const size_t word_size = load_feature.getWordSize(space);
  if (word_size == 0) return std::nullopt;

  const size_t elem_bytes =
      (static_cast<size_t>(elem_type.getIntOrFloatBitWidth()) + 7) / 8;
  // Words one element occupies, always >= 1, and from that the smallest load
  // that still covers every source element.
  const size_t elem_words = (elem_bytes + word_size - 1) / word_size;
  auto best = granularity_list.fitAccess(
      static_cast<size_t>(src_total_elements) * elem_words);
  if (!best) return std::nullopt;

  return static_cast<int64_t>(best.getSizeInWords() / elem_words);
}

}  // namespace

int64_t computeSplatGranularityElements(
    int64_t src_total_elements, mlir::Type elem_type, ResourceType kind,
    const mlir::ktdf_arch::ResourceKinds& resource_kinds) {
  if (!kind) return src_total_elements;

  auto load_feature =
      resource_kinds.getFeature<mlir::ktdf_arch::feature::Load>(kind);
  if (!load_feature) return src_total_elements;

  auto granularity_map = load_feature.getAccessGranularity();
  if (!granularity_map) return src_total_elements;

  // The memory space is stripped by buildLogicalMemoryViews, so it is not known
  // here.  Take the narrowest load that fits in any space the feature declares,
  // skipping the spaces that declare nothing usable rather than letting one of
  // them stand in as a result.
  std::optional<int64_t> narrowest;
  for (auto [space, _] : granularity_map) {
    std::optional<int64_t> fitting = fittingGranularityElements(
        load_feature, space, src_total_elements, elem_type);
    if (fitting && (!narrowest || *fitting < *narrowest)) narrowest = fitting;
  }

  // A granularity below the source width would drop elements, so it is no use.
  return std::max(narrowest.value_or(src_total_elements), src_total_elements);
}

mlir::IntegerSet buildIntegerSetFromSizes(mlir::MLIRContext* ctx,
                                          llvm::ArrayRef<int64_t> sizes) {
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

namespace {

/// The addressing a whole-buffer agen.vector_load / agen.vector_store shares:
/// an identity map over `memref`'s rank, the index set covering every element,
/// and a zero index per dimension.
struct WholeBufferAccess {
  mlir::AffineMap map;
  mlir::IntegerSet set;
  llvm::SmallVector<mlir::Value> indices;

  WholeBufferAccess(mlir::OpBuilder& builder, mlir::Location loc,
                    mlir::Value memref) {
    auto memref_type = mlir::cast<mlir::MemRefType>(memref.getType());
    const unsigned rank = memref_type.getRank();
    mlir::MLIRContext* ctx = builder.getContext();

    if (rank > 0)
      indices.assign(
          rank,
          mlir::arith::ConstantIndexOp::create(builder, loc, 0).getResult());

    map = mlir::AffineMap::getMultiDimIdentityMap(rank, ctx);
    set = buildIntegerSetFromSizes(ctx, memref_type.getShape());
  }
};

}  // namespace

mlir::Value emitVectorLoad(mlir::OpBuilder& builder, mlir::Location loc,
                           mlir::VectorType vec_type, mlir::Value memref) {
  WholeBufferAccess access(builder, loc, memref);
  return mlir::agen::VectorLoadOp::create(builder, loc, vec_type, memref,
                                          /*dbgName=*/nullptr, access.map,
                                          access.indices, access.set,
                                          access.map)
      .getResult();
}

void emitVectorStore(mlir::OpBuilder& builder, mlir::Location loc,
                     mlir::Value value, mlir::Value memref) {
  WholeBufferAccess access(builder, loc, memref);
  mlir::agen::VectorStoreOp::create(builder, loc, value, memref,
                                    /*dbgName=*/nullptr, access.map,
                                    access.indices, access.set, access.map);
}

namespace {

/// Emit the vectorchain.shuffle that `indices` repeated `repetition` times
/// describes, yielding `result_type`.
mlir::Value emitShuffle(mlir::OpBuilder& builder, mlir::Location loc,
                        mlir::VectorType result_type, mlir::Value src_vec,
                        llvm::ArrayRef<mlir::Attribute> indices,
                        int64_t repetition) {
  return mlir::vectorchain::ShuffleOp::create(
             builder, loc, result_type, src_vec,
             /*variable=*/mlir::ValueRange{}, /*pad=*/mlir::ValueRange{},
             /*mask=*/nullptr, /*dbgName=*/nullptr,
             builder.getArrayAttr(indices),
             builder.getI32IntegerAttr(static_cast<int32_t>(repetition)))
      .getOutput();
}

}  // namespace

mlir::Value emitSubSimdSplatShuffle(mlir::OpBuilder& builder,
                                    mlir::Location loc, mlir::Value src_vec,
                                    int64_t group_elements) {
  auto vec_type = mlir::cast<mlir::VectorType>(src_vec.getType());
  const int64_t lanes = vec_type.getNumElements();
  assert(group_elements > 0 && "sub-SIMD group width must be positive");
  assert(lanes % group_elements == 0 &&
         "sub-SIMD group width must divide the vector width");

  // A single-lane group is already uniform, so there is nothing to spread.
  if (group_elements == 1) return src_vec;

  llvm::SmallVector<mlir::Attribute> indices(group_elements,
                                             builder.getI32IntegerAttr(0));
  return emitShuffle(builder, loc, vec_type, src_vec, indices,
                     lanes / group_elements);
}

mlir::Value emitSplatShuffle(mlir::OpBuilder& builder, mlir::Location loc,
                             mlir::Value src_vec, int64_t dst_elements) {
  auto vec_type = mlir::cast<mlir::VectorType>(src_vec.getType());
  const int64_t lanes = vec_type.getNumElements();
  assert(lanes > 0 && "splat source width must be positive");
  assert(dst_elements % lanes == 0 &&
         "splat destination width must be a multiple of the source width");

  llvm::SmallVector<mlir::Attribute> indices;
  indices.reserve(lanes);
  for (int64_t i = 0; i < lanes; ++i)
    indices.push_back(builder.getI32IntegerAttr(static_cast<int32_t>(i)));

  auto result_type =
      mlir::VectorType::get({dst_elements}, vec_type.getElementType());
  return emitShuffle(builder, loc, result_type, src_vec, indices,
                     dst_elements / lanes);
}

mlir::MemRefType memRefTypeFor(mlir::Type type, mlir::Attribute memory_space) {
  auto shaped = llvm::dyn_cast<mlir::ShapedType>(type);
  if (!shaped || !shaped.hasStaticShape()) return nullptr;
  return mlir::MemRefType::get(shaped.getShape(), shaped.getElementType(),
                               /*layout=*/mlir::MemRefLayoutAttrInterface{},
                               memory_space);
}

}  // namespace scheduler
