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

#include "dataflow-scheduler/Dialect/KTDFArch/Analysis/ResourceKinds.h"
#include "dataflow-scheduler/Utils/SchedulerExtContext.h"
#include "llvm/Support/LogicalResult.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Types.h"
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

/// Compute the number of elements to load/receive in a splat transfer, rounded
/// up to the smallest access granularity of the load unit that covers all
/// source elements.
///
/// Iterates all memory spaces declared in the Load feature for `kind` and
/// returns the smallest fitting granularity found across all of them.
///
/// Falls back to `src_total_elements` when arch info is unavailable or no
/// fitting granularity exists.
int64_t computeSplatGranularityElements(
    int64_t src_total_elements, mlir::Type elem_type, mlir::Attribute kind,
    const mlir::ktdf_arch::ResourceKinds& resource_kinds);

/// Compute the sub-SIMD lane count for a splat shuffle on a compute unit.
///
/// Reads `sub_simd_lanes` from the SIMD feature of `kind`, looks up
/// `elem_type`, and returns that count as the shuffle granularity.
/// Also verifies that the SIMD feature declares
/// `shuffle_modes = { FirstSubSimdLaneToEachSubSimd }`: if not, emits an
/// error on `op_for_errors` and returns failure.
///
/// Falls back to `dst_total_elements` (no shuffle) when the SIMD feature or
/// `sub_simd_lanes` is absent.
llvm::FailureOr<int64_t> computeSplatSubSimdElements(
    int64_t dst_total_elements, mlir::Type elem_type, mlir::Attribute kind,
    const mlir::ktdf_arch::ResourceKinds& resource_kinds,
    mlir::Operation* op_for_errors);

/// Return the kind attribute of the register-file memory that is co-located
/// with the compute unit of kind `compute_kind` in the arch graph.
///
/// "Co-located" means the register-file memory's exemplar lives in the same
/// parent GroupOp as the compute unit's exemplar.  Returns nullptr when no
/// such memory is found.
mlir::Attribute getComputeRegisterKind(
    mlir::Attribute compute_kind,
    const mlir::ktdf_arch::ResourceKinds& resource_kinds);

}  // namespace scheduler

#endif  // DATAFLOW_SCHEDULER_CONVERSION_UTILS_UTILS_H_
