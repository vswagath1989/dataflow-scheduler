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

#include <mlir/IR/Value.h>

#include <optional>

#include "dataflow-scheduler/Analysis/Mapping.h"
#include "dataflow-scheduler/Dialect/KTDFArch/Analysis/ResourceKinds.h"
#include "dataflow-scheduler/Utils/SchedulerExtContext.h"
#include "llvm/ADT/ArrayRef.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/IntegerSet.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/Types.h"

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
/// The memory space is not available at lowering time, so every space the Load
/// feature of `kind` declares is tried and the smallest fitting granularity
/// found across them wins.  Spaces that declare nothing usable are skipped.
///
/// Falls back to `src_total_elements` -- i.e. no widening -- when no space
/// yields a fitting granularity.
int64_t computeSplatGranularityElements(
    int64_t src_total_elements, mlir::Type elem_type, ResourceType kind,
    const mlir::ktdf_arch::ResourceKinds& resource_kinds);

/// Build an IntegerSet from a size array: each size-1 entry becomes an equality
/// constraint (d_i == 0); each size-N entry (N > 1) becomes a range pair
/// (d_i >= 0, N-1-d_i >= 0).
mlir::IntegerSet buildIntegerSetFromSizes(mlir::MLIRContext* ctx,
                                          llvm::ArrayRef<int64_t> sizes);

/// Emit an agen.vector_load that reads all elements of `memref` into a flat
/// vector.  The insertion point of `builder` must be set by the caller.
mlir::Value emitVectorLoad(mlir::OpBuilder& builder, mlir::Location loc,
                           mlir::VectorType vec_type, mlir::Value memref);

/// Emit an agen.vector_store that writes `value` into `memref` covering all
/// elements.  The insertion point of `builder` must be set by the caller.
void emitVectorStore(mlir::OpBuilder& builder, mlir::Location loc,
                     mlir::Value value, mlir::Value memref);

/// Create a vectorchain.shuffle that fills every lane of each `group_elements`
/// wide group with the value the first lane of that group holds, leaving
/// `src_vec` uniform within each group.
///
/// Emitted as indices [0, 0, ..., 0] of length `group_elements` repeated
/// `lanes / group_elements` times, which is what a compute unit declaring
/// `shuffle_modes = { FirstSubSimdLaneToEachSubSimd }` performs.
///
/// `group_elements` must be positive and must divide the vector width evenly;
/// callers are expected to have validated (and diagnosed) that beforehand.
/// Returns `src_vec` unchanged when `group_elements` is 1, since then every
/// group is already a single lane.
mlir::Value emitSubSimdSplatShuffle(mlir::OpBuilder& builder,
                                    mlir::Location loc, mlir::Value src_vec,
                                    int64_t group_elements);

/// Create a vectorchain.shuffle that broadcasts `src_vec` to a
/// `dst_elements`-wide vector by repeating it end to end, i.e. indices
/// [0 .. lanes(src_vec) - 1] with repetition `dst_elements / lanes(src_vec)`.
///
/// `dst_elements` must be a multiple of `src_vec`'s width; callers are expected
/// to have validated (and diagnosed) that beforehand.
mlir::Value emitSplatShuffle(mlir::OpBuilder& builder, mlir::Location loc,
                             mlir::Value src_vec, int64_t dst_elements);

/// Return a layout-free `MemRefType` with the same shape and element type as
/// the shaped type `type`, in `memory_space` (no memory space when null).
/// Returns nullptr when `type` is not shaped or has no static shape.
///
/// With no memory space this is the type a memref-typed `ktdf.read_from_fifo`
/// hands out, so a rewrite that turns a tensor-typed read into a memref one can
/// ask for it.  With one it is the type of a buffer in a memory the device
/// names -- a register file, say -- which is what lets address assignment place
/// the buffer.  Either way the shape comes from `type`, so a caller need not
/// spell a literal that would only hold for one shape.
mlir::MemRefType memRefTypeFor(mlir::Type type,
                               mlir::Attribute memory_space = nullptr);

}  // namespace scheduler

#endif  // DATAFLOW_SCHEDULER_CONVERSION_UTILS_UTILS_H_
