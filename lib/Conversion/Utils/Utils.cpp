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

}  // namespace scheduler

// Made with Bob
