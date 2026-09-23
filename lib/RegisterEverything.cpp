//===-- RegisterEverything.cpp ----------------------------------*- c++ -*-===//
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
// This file implements the main registration entry point for the scheduler.
//
// IMPORTANT:
// When you update this file, make sure that any new dialects, passes or
// extensions that you add below, you also declare as link dependencies in the
// accompanying `CMakeLists.txt` file!
//
//===----------------------------------------------------------------------===//

#include "dataflow-scheduler/RegisterEverything.h"

#include <mlir/Dialect/Affine/IR/AffineOps.h>
#include <mlir/Dialect/Affine/IR/ValueBoundsOpInterfaceImpl.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Arith/IR/ValueBoundsOpInterfaceImpl.h>
#include <mlir/Dialect/Bufferization/IR/Bufferization.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/Linalg/IR/Linalg.h>
#include <mlir/Dialect/Math/IR/Math.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/Dialect/SCF/IR/ValueBoundsOpInterfaceImpl.h>
#include <mlir/Dialect/SCF/Transforms/Passes.h>
#include <mlir/Dialect/Tensor/IR/Tensor.h>
#include <mlir/IR/DialectRegistry.h>
#include <mlir/Transforms/Passes.h>

#include "dataflow-scheduler/Conversion/backend/ScheduleIRToDFIR/Passes.h"
#include "dataflow-scheduler/Conversion/frontend/KTIRToScheduleIR/Passes.h"
#include "dataflow-scheduler/Dialect/Agen/Agen.h"
#include "dataflow-scheduler/Dialect/Dataflow/DataflowDialect.h"
#include "dataflow-scheduler/Dialect/Dataflow/Transforms/MappableExtensions.h"
#include "dataflow-scheduler/Dialect/KTDF/KTDFDialect.h"
#include "dataflow-scheduler/Dialect/KTDF/Transforms/MappableExtensions.h"
#include "dataflow-scheduler/Dialect/KTDF/Transforms/Passes.h"
#include "dataflow-scheduler/Dialect/KTDFArch/KTDFArchDialect.h"
#include "dataflow-scheduler/Dialect/KTDFArch/Transforms/Passes.h"
#include "dataflow-scheduler/Dialect/KTDFLowering/KTDFLoweringDialect.h"
#include "dataflow-scheduler/Dialect/KTDPLowering/KTDPLoweringDialect.h"
#include "dataflow-scheduler/Dialect/Symbol/Symbol.h"
#include "dataflow-scheduler/Dialect/Uniform/Uniform.h"
#include "dataflow-scheduler/Dialect/VectorChain/VectorChain.h"
#include "dataflow-scheduler/Transforms/Passes.h"
#include "ktir/Dialect/KTDP/KTDPDialect.h"
#include "ktir/Dialect/SpyreOp/SpyreOpDialect.h"

using namespace scheduler;

//===----------------------------------------------------------------------===//
// Exported Only
//===----------------------------------------------------------------------===//

void scheduler::registerPasses() {
  // Register the passes required from KTIR.

  // Register the passes defined in the scheduler.
  mlir::ktdf_arch::registerKTDFArchPasses();
  registerKTIRToScheduleIRPasses();
  registerScheduleIRToDFIRPasses();
  mlir::ktdf::registerKTDFTransformsPasses();
  registerSchedulerTransformsPasses();
}

void scheduler::registerDialects(mlir::DialectRegistry& registry) {
  // clang-format off

  // Register the dialects required from KTIR.
  registry.insert<mlir::ktdp::KtdpDialect, mlir::spyreop::SpyreOpDialect>();

  // Register the dialects defined in the scheduler.
  registry.insert<mlir::agen::AgenDialect,
                  mlir::dataflow::DataflowDialect,
                  mlir::ktdf::KTDFDialect,
                  mlir::ktdf_arch::KTDFArchDialect,
                  mlir::ktdf_lowering::KTDFLoweringDialect,
                  mlir::ktdp_lowering::KTDPLoweringDialect,
                  mlir::symbol::SymbolDialect,
                  mlir::uniform::UniformDialect,
                  mlir::vectorchain::VectorChainDialect>();

  // clang-format on
}

void scheduler::registerExtensions(mlir::DialectRegistry& registry) {
  // Register the extensions required from KTIR.

  // Register the extensions provided by the scheduler.
  mlir::dataflow::registerMappableInterfaceExternalModels(registry);
  mlir::ktdf::registerMappableInterfaceExternalModels(registry);
}

//===----------------------------------------------------------------------===//
// Imported And Exported
//===----------------------------------------------------------------------===//

void scheduler::registerAllPasses() {
  // Register the passes required from MLIR.
  mlir::registerCSEPass();
  mlir::registerSCCPPass();
  mlir::registerSymbolDCEPass();
  mlir::registerCanonicalizerPass();
  mlir::registerLoopInvariantCodeMotionPass();
  mlir::registerSCFPasses();

  // Register our own passes.
  registerPasses();
}

void scheduler::registerAllDialects(mlir::DialectRegistry& registry) {
  // clang-format off

  // Register the dialects required from MLIR.
  registry.insert<mlir::affine::AffineDialect,
                  mlir::arith::ArithDialect,
                  mlir::bufferization::BufferizationDialect,
                  mlir::func::FuncDialect,
                  mlir::linalg::LinalgDialect,
                  mlir::math::MathDialect,
                  mlir::memref::MemRefDialect,
                  mlir::scf::SCFDialect,
                  mlir::tensor::TensorDialect>();

  // clang-format on

  // Register our own dialects.
  registerDialects(registry);
}

void scheduler::registerAllExtensions(mlir::DialectRegistry& registry) {
  // Register the extensions required from MLIR.
  mlir::affine::registerValueBoundsOpInterfaceExternalModels(registry);
  mlir::arith::registerValueBoundsOpInterfaceExternalModels(registry);
  mlir::scf::registerValueBoundsOpInterfaceExternalModels(registry);

  // Register our own extensions.
  registerExtensions(registry);
}
