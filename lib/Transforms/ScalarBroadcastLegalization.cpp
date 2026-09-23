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
// ScalarBroadcastLegalization: legalize scalar broadcast inputs to linalg ops.
//
// See docs/superpowers/specs/2026-05-30-scalar-broadcast-legalization-design.md
//
//===----------------------------------------------------------------------===//

#include <llvm/ADT/SmallPtrSet.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/TypeSwitch.h>
#include <llvm/Support/LogicalResult.h>
#include <mlir/IR/Attributes.h>
#include <mlir/Support/WalkResult.h>

#include <memory>

#include "dataflow-scheduler/Analysis/Mapping.h"
#include "dataflow-scheduler/Analysis/Utils.h"
#include "dataflow-scheduler/Dialect/KTDF/KTDF.h"
#include "dataflow-scheduler/Dialect/KTDF/Utils/Utils.h"
#include "dataflow-scheduler/Dialect/KTDFArch/Analysis/DeviceManager.h"
#include "dataflow-scheduler/Dialect/KTDFArch/Analysis/NodeLinks.h"
#include "dataflow-scheduler/Dialect/KTDFArch/Analysis/ResourceKinds.h"
#include "dataflow-scheduler/Dialect/KTDFArch/KTDFArch.h"
#include "dataflow-scheduler/Dialect/KTDFArch/KTDFArchIntrinsics.h"
#include "dataflow-scheduler/Transforms/Passes.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/DebugLog.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

#define PASS_NAME "scalar-broadcast-legalization"
#define DEBUG_TYPE PASS_NAME

using namespace scheduler;

namespace scheduler {
#define GEN_PASS_DEF_SCALARBROADCASTLEGALIZATIONPASS
#include "dataflow-scheduler/Transforms/Passes.h.inc"
}  // namespace scheduler

static llvm::cl::opt<bool> DisableThisPass(
    "disable-" PASS_NAME,
    llvm::cl::desc("Disable Scalar Broadcast Legalization pass"),
    llvm::cl::init(false));

namespace {

[[nodiscard]]
auto getResourceKind(mlir::Type type, bool is_source) -> ResourceType {
  return llvm::TypeSwitch<mlir::Type, ResourceType>(type)
      .Case([&](mlir::ktdf::FifoSlotType type) -> ResourceType {
        return llvm::cast<ResourceType>(is_source ? type.getDest()
                                                  : type.getSrc());
      })
      .Case([&](mlir::MemRefType type) -> ResourceType {
        return llvm::cast<ResourceType>(type.getMemorySpace());
      })
      .Default(nullptr);
}

[[nodiscard]]
auto getResourceKind(mlir::ktdf::StageOp stage) -> ResourceType {
  if (const auto maybe_units = stage.getApplicableUnits();
      maybe_units && maybe_units->size() == 1) {
    return llvm::cast<ResourceType>(maybe_units->getValue().front());
  }

  return nullptr;
}

[[nodiscard]]
auto getResourceKind(mlir::Value value, bool is_source) -> ResourceType {
  // Try to get the space from the type of the value.
  if (const auto kind = getResourceKind(value.getType(), is_source); kind) {
    return kind;
  }

  // Infer the space from the stage transfer units.
  auto stage = value.getDefiningOp()->getParentOfType<mlir::ktdf::StageOp>();
  return stage ? getResourceKind(stage) : nullptr;
}

struct Hop {
  mlir::ktdf::DataTransferOp transfer;
  mlir::ktdf_arch::Resource from;
  mlir::ktdf_arch::Resource to;
  mlir::ktdf_arch::Link via;
};

auto getHop(const mlir::ktdf_arch::ResourceKinds& resource_kinds,
            mlir::ktdf::DataTransferOp transfer,
            llvm::SmallVectorImpl<Hop>& hops) -> llvm::LogicalResult {
  // Find the source and target resources.
  auto source_kind = getResourceKind(transfer.getSource(), true);
  auto target_kind = getResourceKind(transfer.getDestination(), false);
  if (!source_kind || !target_kind) {
    return llvm::failure();
  }
  auto source = resource_kinds.getInstance<mlir::ktdf_arch::Node>(source_kind);
  auto target = resource_kinds.getInstance<mlir::ktdf_arch::Node>(target_kind);
  if (!source || !target) {
    return llvm::failure();
  }

  // Find the load_store unit, if any.
  mlir::ktdf_arch::Node load_store;
  if (auto stage = transfer->getParentOfType<mlir::ktdf::StageOp>(); stage) {
    if (const auto kind = getResourceKind(stage); kind) {
      load_store = resource_kinds.getInstance<mlir::ktdf_arch::Node>(kind);
    }
  }

  if (!load_store || source == load_store || target == load_store) {
    // There must be an unambiguous direct link.
    const auto link = mlir::ktdf_arch::getLink(source, target);
    if (!link) {
      return llvm::failure();
    }

    hops.emplace_back(Hop{transfer, source, target, link});
    return llvm::success();
  }

  // There must be two unambiguous direct links.
  auto incoming = mlir::ktdf_arch::getLink(source, load_store);
  auto outgoing = mlir::ktdf_arch::getLink(load_store, target);
  if (!incoming || !outgoing) {
    return llvm::failure();
  }
  hops.emplace_back(Hop{transfer, load_store, target, outgoing});
  hops.emplace_back(Hop{transfer, source, load_store, incoming});
  return llvm::success();
}

auto getHop(const mlir::ktdf_arch::ResourceKinds& resource_kinds,
            mlir::Value value, bool is_load, llvm::SmallVectorImpl<Hop>& hops)
    -> llvm::FailureOr<mlir::Value> {
  // Find the single ktdf.data_transfer targeting the value.
  mlir::ktdf::DataTransferOp transfer;
  for (auto* user : value.getUsers()) {
    auto candidate = mlir::dyn_cast<mlir::ktdf::DataTransferOp>(user);
    if (!candidate) {
      continue;
    }

    if (is_load) {
      if (candidate.getDestination() != value) {
        continue;
      }
    } else {
      if (candidate.getSource() != value) {
        continue;
      }
    }

    if (transfer) {
      auto diag = candidate->emitWarning("multiple transfers for ") << value;
      diag.attachNote(transfer->getLoc()) << "previous transfer is here";
      return diag;
    }
    transfer = candidate;
  }

  if (!transfer) {
    return value;
  }

  if (failed(getHop(resource_kinds, transfer, hops))) {
    return llvm::failure();
  }

  if (is_load) {
    return transfer.getSource();
  }

  return transfer.getDestination();
}

auto getHops(const mlir::ktdf_arch::ResourceKinds& resource_kinds,
             mlir::Value value, bool is_load,
             llvm::SmallVectorImpl<Hop>& result) -> llvm::LogicalResult {
  if (auto read = value.getDefiningOp<mlir::ktdf::ReadFromFifoOp>(); read) {
    value = read.getFifoSlot();
  }

  while (true) {
    const auto next = getHop(resource_kinds, value, is_load, result);
    if (llvm::failed(next)) {
      return llvm::failure();
    }

    if (next == value) {
      break;
    }

    value = *next;
  }

  return llvm::success();
}

[[nodiscard]]
auto mustLegalizeLoad(Hop& hop, size_t size_in_bytes, size_t align_in_bytes)
    -> llvm::FailureOr<bool> {
  const auto unit = hop.to.getFeature<mlir::ktdf_arch::feature::Load>();
  if (!unit) {
    hop.to.emitError() << "unit does not support load";
    return llvm::failure();
  }

  const auto load_accesses = unit.getAccessGranularity(hop.from.getKind());
  if (!load_accesses) {
    hop.to->emitError() << "unit does not support load from "
                        << hop.from.getKind();
    return llvm::failure();
  }

  const auto word_size = unit.getWordSize(hop.from.getKind());
  if (size_in_bytes % word_size != 0 || align_in_bytes % word_size != 0) {
    return true;
  }
  const auto size_in_words = size_in_bytes / word_size;
  const auto align_in_words = align_in_bytes / word_size;

  // Must have a perfect-fit access.
  const auto access = load_accesses.fitAccess(size_in_words, align_in_words);
  return access && access.getSizeInWords() == size_in_words;
}

struct BroadcastLegalizationSite {
  mlir::linalg::GenericOp generic_op;
  unsigned operand_index;
  llvm::SmallVector<Hop> hops;
  int64_t vector_width;
  mlir::AffineMap new_indexing_map;
};

static bool isBroadcastOverVectorDim(mlir::AffineMap m) {
  if (m.getNumResults() == 0) return false;
  return mlir::isa<mlir::AffineConstantExpr>(m.getResults().back());
}

static mlir::LogicalResult collectSites(
    mlir::ktdf::PipelineOp pipeline,
    llvm::SmallVector<BroadcastLegalizationSite>& sites,
    mlir::ktdf_arch::ResourceKinds& resource_kinds) {
  const auto walk_result = pipeline.walk([&](mlir::linalg::GenericOp
                                                 generic_op) {
    auto stage = generic_op->getParentOfType<mlir::ktdf::StageOp>();
    if (!stage) return mlir::WalkResult::advance();
    auto compute_kind = getResourceKind(stage);
    if (!compute_kind) {
      return mlir::WalkResult::advance();
    }
    auto compute = resource_kinds.getInstance(compute_kind);
    if (!compute) {
      return mlir::WalkResult::advance();
    }

    auto inputs = generic_op.getDpsInputs();
    if (inputs.empty()) return mlir::WalkResult::advance();
    mlir::Type elem_type =
        mlir::cast<mlir::RankedTensorType>(inputs[0].getType())
            .getElementType();
    int64_t vector_width =
        compute.getFeature<mlir::ktdf_arch::feature::SIMD>().getLanes(
            elem_type);
    const auto maybe_size = tryGetSizeInBytes(elem_type);
    if (vector_width == 0 || !maybe_size) {
      generic_op->emitWarning("unsupported vector scalar type ") << elem_type;
      return mlir::WalkResult::advance();
    }

    auto indexing_maps = generic_op.getIndexingMapsArray();
    unsigned num_inputs = generic_op.getNumDpsInputs();

    for (unsigned i = 0; i < num_inputs; ++i) {
      mlir::AffineMap m = indexing_maps[i];
      if (!isBroadcastOverVectorDim(m)) continue;

      mlir::Value input = inputs[i];
      llvm::SmallVector<Hop> hops;
      if (llvm::failed(getHops(resource_kinds, input, true, hops))) {
        generic_op.emitWarning() << "scalar broadcast operand " << i
                                 << " has unexpected path structure — skipping";
        continue;
      }
      std::reverse(hops.begin(), hops.end());

      Hop* load_hop = nullptr;
      Hop* fifo_hop = nullptr;
      if (!hops.empty()) {
        load_hop = &hops.front();

        if (hops.front().via.getFeature<mlir::ktdf_arch::feature::Queue>()) {
          fifo_hop = &hops.back();
        }
      }

      bool needs_legalize = false;
      if (load_hop) {
        // FIXME: What alignment can we assume here?
        if (const auto maybe_legalize =
                mustLegalizeLoad(*load_hop, *maybe_size, *maybe_size);
            succeeded(maybe_legalize)) {
          needs_legalize |= *maybe_legalize;
        } else {
          return mlir::WalkResult::interrupt();
        }
      }
      needs_legalize |=
          !compute.getFeature<mlir::ktdf_arch::feature::SIMD>().canSplat();

      if (!needs_legalize) continue;

      if (fifo_hop) {
        const auto fifo_granularity =
            fifo_hop->via
                .getProperty<mlir::ktdf_arch::TransferGranularityAttr>();
        if (fifo_granularity && !fifo_granularity.contains(*maybe_size)) {
          generic_op.emitError()
              << "transfer unit does not support sub-vector reads; "
                 "cannot legalize scalar broadcast operand "
              << i;
          return mlir::WalkResult::interrupt();
        }
      }
      if (fifo_hop &&
          !fifo_hop->via.getFeature<mlir::ktdf_arch::feature::SIMD>()
               .canSplat()) {
        generic_op.emitError() << "transfer unit does not support splat; "
                                  "cannot legalize scalar broadcast operand "
                               << i;
        return mlir::WalkResult::interrupt();
      }

      llvm::SmallVector<mlir::AffineExpr> new_results(m.getResults().begin(),
                                                      m.getResults().end());
      new_results.back() =
          mlir::getAffineDimExpr(m.getNumDims() - 1, generic_op.getContext());
      mlir::AffineMap new_map =
          mlir::AffineMap::get(m.getNumDims(), m.getNumSymbols(), new_results,
                               generic_op.getContext());

      sites.push_back({generic_op, i, std::move(hops), vector_width, new_map});
    }
    return mlir::WalkResult::advance();
  });

  return mlir::success(!walk_result.wasInterrupted());
}

/// Widen the last dimension of the memref alloc backing `private_result`
/// to `new_last_dim`. Creates a new alloc, replaces all uses, updates the
/// ktdf.private result type, erases old alloc.
static mlir::Value widenMemrefAlloc(mlir::Value private_result,
                                    int64_t new_last_dim,
                                    mlir::OpBuilder& builder) {
  auto priv_op = private_result.getDefiningOp<mlir::ktdf::PrivateOp>();
  assert(priv_op && "private_result must come from ktdf.private");
  auto result_idx =
      mlir::cast<mlir::OpResult>(private_result).getResultNumber();
  mlir::ktdf::PrivateYieldOp yield = priv_op.getYieldOp();
  mlir::Value alloc_val = yield.getOperand(result_idx);
  auto alloc_op = alloc_val.getDefiningOp<mlir::memref::AllocOp>();
  assert(alloc_op && "must trace back to memref.alloc");

  auto orig_type = alloc_op.getType();
  llvm::SmallVector<int64_t> new_shape(orig_type.getShape());
  new_shape.back() = new_last_dim;
  auto new_type = mlir::MemRefType::get(new_shape, orig_type.getElementType(),
                                        mlir::MemRefLayoutAttrInterface(),
                                        orig_type.getMemorySpace());

  builder.setInsertionPoint(alloc_op);
  auto new_alloc =
      mlir::memref::AllocOp::create(builder, alloc_op.getLoc(), new_type);

  alloc_op.getResult().replaceAllUsesWith(new_alloc.getResult());
  alloc_op.erase();

  // Update the ktdf.private result type in-place.
  private_result.setType(new_type);

  return private_result;
}

/// Widens the slot \p private_result hands out to \p new_num_elements, and
/// returns it.
///
/// A slot's width is part of its type, carried by the allocation it comes from
/// and by the result the pipeline hands it out as, so both are re-typed. What
/// arrives is still one element -- the transfer splats it -- but the read that
/// takes it is the compute's width, and a read has to match the slot it reads.
static mlir::Value widenFifoSlot(mlir::Value private_result,
                                 int64_t new_num_elements,
                                 mlir::OpBuilder& builder) {
  auto priv_op = private_result.getDefiningOp<mlir::ktdf::PrivateOp>();
  if (!priv_op) return private_result;

  const auto result_idx =
      mlir::cast<mlir::OpResult>(private_result).getResultNumber();
  mlir::ktdf::PrivateYieldOp yield = priv_op.getYieldOp();
  auto allocated = mlir::dyn_cast<mlir::OpResult>(yield.getOperand(result_idx));
  if (!allocated) return private_result;
  auto alloc_op = allocated.getDefiningOp<mlir::ktdf::FifoAllocateOp>();
  if (!alloc_op) return private_result;

  auto slot = mlir::cast<mlir::ktdf::FifoSlotType>(allocated.getType());
  if (slot.getNumElements() == new_num_elements) return private_result;
  auto wider = mlir::ktdf::FifoSlotType::get(slot.getContext(), slot.getSrc(),
                                             slot.getDest(), new_num_elements,
                                             slot.getElementType());

  // The allocation again, that slot wider. Only the one result changes type;
  // the others are handed on as they were.
  llvm::SmallVector<mlir::Type> slots(alloc_op->getResultTypes());
  slots[allocated.getResultNumber()] = wider;
  builder.setInsertionPoint(alloc_op);
  auto new_alloc = mlir::ktdf::FifoAllocateOp::create(
      builder, alloc_op.getLoc(), slots, alloc_op.getOperands());
  alloc_op->replaceAllUsesWith(new_alloc->getResults());
  alloc_op.erase();

  private_result.setType(wider);
  return private_result;
}

/// Apply all mutations for one BroadcastLegalizationSite.
static void applyTransformation(const BroadcastLegalizationSite& site,
                                mlir::OpBuilder& builder) {
  mlir::linalg::GenericOp generic_op = site.generic_op;
  mlir::Value input = generic_op.getDpsInputs()[site.operand_index];

  // Step 1: Traverse hops source→target.
  llvm::SmallPtrSet<mlir::Operation*, 4> seen_transfers;
  for (auto hop : site.hops) {
    if (!seen_transfers.insert(hop.transfer.getOperation()).second) continue;

    bool is_fifo_hop =
        hop.via.getFeature<mlir::ktdf_arch::feature::Queue>() != nullptr;

    if (!is_fifo_hop) {
      // Memref hop: widen the alloc and update transfer sizes.
      widenMemrefAlloc(hop.transfer.getDestination(), site.vector_width,
                       builder);

      // Update static_dest_sizes last dim.
      auto dest_sizes_opt = hop.transfer.getStaticDestSizesArray();
      if (dest_sizes_opt) {
        llvm::SmallVector<int64_t> new_dest_static(*dest_sizes_opt);
        new_dest_static.back() = site.vector_width;
        hop.transfer.setStaticDestSizes(
            llvm::ArrayRef<int64_t>(new_dest_static));
      }

      // Update static_source_sizes last dim.
      mlir::ktdf::setInnermostStaticSourceSize(hop.transfer, site.vector_width);
    } else {
      // FIFO hop: mark with transfer_mode="splat", widen the slot it splats
      // into, and say so on the transfer. The source stays the one element that
      // arrives; it is the destination that is filled.
      mlir::ktdf::markSplatTransfer(hop.transfer);
      widenFifoSlot(hop.transfer.getDestination(), site.vector_width, builder);

      auto dest_sizes_opt = hop.transfer.getStaticDestSizesArray();
      if (dest_sizes_opt) {
        llvm::SmallVector<int64_t> new_dest_static(*dest_sizes_opt);
        new_dest_static.back() = site.vector_width;
        hop.transfer.setStaticDestSizes(
            llvm::ArrayRef<int64_t>(new_dest_static));
      }
    }
  }

  // Step 2: Update read_from_fifo result type + linalg.generic map together.
  auto read_op = input.getDefiningOp<mlir::ktdf::ReadFromFifoOp>();
  assert(read_op && "target must be read_from_fifo");

  auto old_result_type =
      mlir::cast<mlir::RankedTensorType>(read_op.getResult().getType());
  llvm::SmallVector<int64_t> new_shape(old_result_type.getShape());
  new_shape.back() = site.vector_width;
  auto new_result_type =
      mlir::RankedTensorType::get(new_shape, old_result_type.getElementType());

  builder.setInsertionPoint(read_op);
  auto new_read = mlir::ktdf::ReadFromFifoOp::create(
      builder, read_op.getLoc(), new_result_type, read_op.getFifoSlot());

  read_op.getResult().replaceAllUsesWith(new_read.getResult());
  read_op.erase();

  // Update the linalg.generic indexing map for this operand.
  auto maps = generic_op.getIndexingMapsArray();
  maps[site.operand_index] = site.new_indexing_map;
  generic_op.setIndexingMapsAttr(builder.getAffineMapArrayAttr(maps));
}

struct ScalarBroadcastLegalizationPass
    : public impl::ScalarBroadcastLegalizationPassBase<
          ScalarBroadcastLegalizationPass> {
  void runOnOperation() override {
    if (DisableThisPass) return;
    LDBG(1) << "========= " PASS_NAME " =========";
    mlir::ModuleOp module_op = getOperation();
    llvm::SmallVector<BroadcastLegalizationSite> sites;

    auto& device_manager = getAnalysis<mlir::ktdf_arch::DeviceManager>();
    auto* const device = device_manager.getOrImportDevice();
    if (!device) {
      module_op->emitError(
          "Unable to import the device specification. This could happen if the "
          "device spec file is empty or contains multiple devices");
      signalPassFailure();
      return;
    }
    auto& resource_kinds =
        device_manager.getOrCreateView<mlir::ktdf_arch::ResourceKinds>(*device);

    const auto walk_result =
        module_op.walk([&](mlir::ktdf::PipelineOp pipeline) {
          if (mlir::failed(collectSites(pipeline, sites, resource_kinds)))
            return mlir::WalkResult::interrupt();
          return mlir::WalkResult::advance();
        });

    if (walk_result.wasInterrupted()) {
      signalPassFailure();
      return;
    }

    // Transformation phase.
    mlir::OpBuilder builder(&getContext());
    for (const auto& site : sites) {
      applyTransformation(site, builder);
    }
  }
};

}  // namespace

std::unique_ptr<mlir::Pass> scheduler::createScalarBroadcastLegalizationPass() {
  return std::make_unique<ScalarBroadcastLegalizationPass>();
}
