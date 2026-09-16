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

#include "dataflow-scheduler/Dialect/Dataflow/Dataflow.h"
#include "dataflow-scheduler/Dialect/KTDFArch/KTDFArch.h"
#include "dataflow-scheduler/Dialect/KTDFArch/KTDFArchIntrinsics.h"
#include "dataflow-scheduler/Dialect/Uniform/Uniform.h"

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

int64_t computeSplatGranularityElements(
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

llvm::FailureOr<int64_t> computeSplatSubSimdElements(
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

mlir::Attribute getComputeRegisterKind(
    mlir::Attribute compute_kind,
    const mlir::ktdf_arch::ResourceKinds& resource_kinds) {
  if (!compute_kind) return nullptr;

  // The register file is the MemoryOp whose exemplar shares the same parent
  // GroupOp as the compute unit's exemplar.
  const auto& compute_entry = resource_kinds[compute_kind];
  if (!compute_entry) return nullptr;

  mlir::Operation* compute_parent = compute_entry.getExemplar()->getParentOp();
  if (!compute_parent) return nullptr;

  for (const auto& kind : resource_kinds) {
    auto mem_op = mlir::dyn_cast<mlir::ktdf_arch::MemoryOp>(kind.getExemplar());
    if (!mem_op) continue;
    if (mem_op->getParentOp() != compute_parent) continue;
    return kind.getKind();
  }
  return nullptr;
}

}  // namespace scheduler
