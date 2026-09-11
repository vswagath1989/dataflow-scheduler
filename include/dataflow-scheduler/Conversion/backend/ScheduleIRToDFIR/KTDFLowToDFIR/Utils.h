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

#ifndef DATAFLOW_SCHEDULER_CONVERSION_KTDFLOWTODFIR_UTILS_H_
#define DATAFLOW_SCHEDULER_CONVERSION_KTDFLOWTODFIR_UTILS_H_

#include <optional>

#include "dataflow-scheduler/Conversion/backend/ScheduleIRToDFIR/KTDFLowToDFIR/UnitTypeDiscovery.h"
#include "dataflow-scheduler/Dialect/Dataflow/Dataflow.h"
#include "dataflow-scheduler/Dialect/KTDFArch/Analysis/ResourceKinds.h"
#include "dataflow-scheduler/Dialect/KTDFArch/KTDFArch.h"
#include "dataflow-scheduler/Utils/SchedulerExtContext.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/LogicalResult.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/IntegerSet.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"

namespace scheduler {

/// Enum representing the type of data transfer operation
enum class DataTransferType {
  kLoadAndStore,    // Memory to memory - use CompositeLoadAndStore
  kLoadAndSend,     // Memory to FIFO - use vector_load and send
  kReceiveAndStore  // FIFO to memory - use receive and vector_store
};

/// Walk up the parent chain of `op` to find the enclosing
/// dataflow::ProgramUnitOp and resolve its resource type from its first unit
/// operand.
/// Returns std::nullopt if no enclosing program_unit exists, it has no unit
/// operands, or the resource type cannot be resolved.
std::optional<scheduler::ResourceType> getEnclosingProgramUnitResourceType(
    mlir::Operation* op);

/// Helper to get flattened vector type from tensor or vector type.
mlir::VectorType getFlattenedVectorType(
    mlir::ShapedType type, mlir::ktdf_arch::ExecutionUnitOp compute);

/// Number of vector lanes the compute resource provides for `elem_type`.
int64_t getVectorLanes(mlir::Type elem_type,
                       mlir::ktdf_arch::ExecutionUnitOp compute);

/// Match units by core ID between program_unit operands and target units,
/// create a def_immutable_mapping + query_map, and return the query result.
mlir::Value createQueryMapForComponent(
    mlir::OpBuilder& builder, mlir::dataflow::ProgramUnitOp program_unit,
    const llvm::SmallVector<mlir::Value, 4>& target_units, mlir::Location loc);

/// For the given program_unit, replace every in-region use of a single-result
/// ktdp.get_compute_tile_id with a uniform.query_map over a core map (keys =
/// the unit's operands, values = each operand get_unit's flat 'core' id),
/// keyed on the program_unit iterator argument. The map + query are built once
/// at the body start. The flat-core-id arith.constant values are shared across
/// units via `core_id_consts` (keyed by core id), materialized at
/// `const_builder`'s position (function scope, before the units) on first need.
/// Returns failure if an operand is not a dataflow.get_unit or lacks a 'core'
/// attribute.
mlir::LogicalResult replaceComputeTileIdWithCoreQuery(
    mlir::dataflow::ProgramUnitOp program_unit,
    llvm::DenseMap<int64_t, mlir::Value>& core_id_consts,
    mlir::OpBuilder& const_builder);

/// Resolve a FIFO endpoint attribute to the corresponding unit query-map Value.
///
/// Given a raw `mlir::Attribute` taken from either FifoSlotType::getSrc() or
/// FifoSlotType::getDest(), this helper:
///   1. Casts it to StringAttr and upper-cases the value to a ResourceType.
///   2. Looks that type up in `components`.
///   3. Calls createQueryMapForComponent to build the uniform.query_map.
///
/// On any failure the error is emitted on `op_for_errors` and the function
/// returns failure.
llvm::FailureOr<mlir::Value> resolveUnitFromFifoAttr(
    mlir::Attribute fifo_attr, const ResourceToUnits& components,
    mlir::PatternRewriter& rewriter, mlir::dataflow::ProgramUnitOp program_unit,
    mlir::Location loc, mlir::Operation* op_for_errors);

/// Build an IntegerSet from a size array: each size-1 entry becomes an equality
/// constraint (d_i == 0); each size-N entry (N > 1) becomes a range pair
/// (d_i >= 0, N-1-d_i >= 0).
mlir::IntegerSet buildIntegerSetFromSizes(mlir::MLIRContext* ctx,
                                          llvm::ArrayRef<int64_t> sizes);

/// Emit an agen.vector_load that reads all elements of `memref` into a flat
/// vector.  The insertion point of `rewriter` must be set by the caller.
mlir::Value emitVectorLoad(mlir::OpBuilder& builder, mlir::Location loc,
                           mlir::VectorType vec_type, mlir::Value memref);

/// Emit an agen.vector_store that writes `value` into `memref` covering all
/// elements.  The insertion point of `rewriter` must be set by the caller.
void emitVectorStore(mlir::OpBuilder& builder, mlir::Location loc,
                     mlir::Value value, mlir::Value memref);

/// Create a vectorchain.shuffle that broadcasts `src_vec`
/// (vector<src_elements x T>) to vector<dst_elements x T> using indices
/// [0..src_elements-1] with repetition = dst_elements / src_elements.
/// `src_elements` must be positive and must divide `dst_elements` evenly;
/// callers are expected to have validated (and diagnosed) that beforehand.
mlir::Value insertSplatShuffle(mlir::OpBuilder& builder, mlir::Location loc,
                               mlir::Value src_vec, int64_t src_elements,
                               int64_t dst_elements);

/// Determine the data transfer type based on source and destination types.
/// @param src_is_fifo True if source is a FIFO slot, false if memref
/// @param dst_is_fifo True if destination is a FIFO slot, false if memref
/// @return The data transfer type (LoadAndStore, LoadAndSend, or
/// ReceiveAndStore), or failure if both sides are FIFOs (unsupported).
llvm::FailureOr<scheduler::DataTransferType> getDataTransferType(
    bool src_is_fifo, bool dst_is_fifo);

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

}  // namespace scheduler

#endif  // DATAFLOW_SCHEDULER_CONVERSION_KTDFLOWTODFIR_UTILS_H_
