//===------------------------------------------------------------*- c++ -*-===//
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

#ifndef DATAFLOW_SCHEDULER_CONVERSION_UTILS_UTILS_H_
#define DATAFLOW_SCHEDULER_CONVERSION_UTILS_UTILS_H_

#include <optional>

#include "dataflow-scheduler/Dialect/KTDF/KTDF.h"
#include "dataflow-scheduler/Utils/SchedulerExtContext.h"
#include "mlir/IR/Value.h"

namespace scheduler {

/// Extract the resource type from a unit SSA value.  Accepts either:
///   - a direct `dataflow.get_unit` result, or
///   - a `uniform.query_map` result (follows query_map →
///     def_immutable_mapping → get_unit).
/// Returns the unit's type name as an upper-cased StringAttr (e.g. "SFU"),
/// or nullopt if the chain cannot be resolved.
auto getUnitResourceType(mlir::Value unit_value)
    -> std::optional<scheduler::ResourceType>;

/// Extract the unit type string from a query_map result by traversing through
/// the query_map -> def_immutable_mapping -> get_unit chain.
/// Emits errors on failure.
/// @param query_map The query_map result value
/// @return The unit type string (uppercase) or empty string if extraction fails
std::string getUnitTypeFromQueryMap(mlir::Value query_map);

}  // namespace scheduler

#endif  // DATAFLOW_SCHEDULER_CONVERSION_UTILS_UTILS_H_

// Made with Bob
