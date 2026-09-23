//===-- dataflow-scheduler-main.cpp -----------------------------*- c++ -*-===//
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

#include "dataflow-scheduler-main.h"

#include <mlir/IR/AsmState.h>
#include <mlir/IR/DialectRegistry.h>
#include <mlir/IR/Location.h>

#include <filesystem>

#include "dataflow-scheduler/Dialect/KTDF/KTDFDialect.h"
#include "dataflow-scheduler/Dialect/KTDFArch/KTDFArch.h"
#include "dataflow-scheduler/Dialect/KTDFArch/KTDFArchDialect.h"
#include "ktir/Dialect/KTDP/KTDPDialect.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/Process.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/ThreadPool.h"
#include "llvm/Support/ToolOutputFile.h"
#include "mlir/Debug/CLOptionsSetup.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Support/ToolUtilities.h"

using namespace mlir;
using namespace llvm;

namespace {
llvm::cl::opt<std::string> schedulerInputFilename(
    llvm::cl::Positional, llvm::cl::desc("<input file>"), llvm::cl::init("-"));
llvm::cl::opt<std::string> schedulerOutputFilename(
    "o", llvm::cl::desc("Output filename"), llvm::cl::value_desc("filename"),
    llvm::cl::init("-"));
llvm::cl::opt<std::string> schedulerDeviceFilename(
    "device", llvm::cl::desc("Device architecture filename"),
    llvm::cl::value_desc("filename"), llvm::cl::init(""));
llvm::cl::opt<bool> schedulerAllowUnregisteredDialects(
    "allow-unregistered-dialect",
    llvm::cl::desc("Allow operation with no registered dialects"),
    llvm::cl::init(false));
llvm::cl::opt<bool> schedulerShowDialects(
    "show-dialects", llvm::cl::desc("Print the list of registered dialects"),
    llvm::cl::init(false));
llvm::cl::opt<bool> schedulerVerifyPasses(
    "verify-each",
    llvm::cl::desc("Run the verifier after each transformation pass"),
    llvm::cl::init(true));
llvm::cl::opt<bool> schedulerPrintFinalIr(
    "print-ir-after-all",
    llvm::cl::desc("Print the final IR after executing the pass pipeline"),
    llvm::cl::init(true));
llvm::cl::opt<std::string> schedulerSplitInputFile(
    "split-input-file",
    llvm::cl::desc("Split the input file into pieces and process each chunk "
                   "independently"),
    llvm::cl::init(""));
llvm::cl::opt<std::string> schedulerOutputSplitMarker(
    "output-split-marker",
    llvm::cl::desc("Marker to use when merging split output chunks"),
    llvm::cl::init(kDefaultSplitMarker));

// Config class to hold CL options including PassPipelineCLParser
struct SchedulerOptMainConfigCLOptions
    : public scheduler::SchedulerOptMainConfig {
  SchedulerOptMainConfigCLOptions() {
    static PassPipelineCLParser passPipeline("", "Compiler passes to run", "p");
    setPassPipelineParser(passPipeline);
  }
};

auto getDeviceName(const std::filesystem::path& path) -> StringRef {
  DialectRegistry registry;
  registry.insert<ktdf_arch::KTDFArchDialect>();
  // Device specs may reference #ktdp.memory_space<...> in memory `kind`
  // attributes, so the ktdp dialect must be available when parsing the file.
  registry.insert<mlir::ktdp::KtdpDialect>();
  // A device's patterns may match an attribute the scheduler's own ops carry,
  // such as #ktdf.splat<...>, which is only an attribute here if ktdf is too.
  registry.insert<ktdf::KTDFDialect>();
  MLIRContext context(registry);

  std::string error_message;
  auto maybe_file = openInputFile(path.native(), &error_message);
  if (!maybe_file) {
    emitError(UnknownLoc::get(&context), "unable to import device: ")
        << error_message;
    std::exit(1);
  }

  auto source_mgr = std::make_shared<llvm::SourceMgr>();
  source_mgr->AddNewSourceBuffer(std::move(maybe_file), {});
  SourceMgrDiagnosticHandler import_handler(*source_mgr, &context);

  ParserConfig config(&context, true);
  auto module = parseSourceFile<ModuleOp>(*source_mgr, config);
  if (!module) {
    // Diagnostic was already emitted.
    std::exit(1);
  }

  auto devices = module->getOps<ktdf_arch::DeviceOp>();
  if (devices.empty()) {
    emitError(module->getLoc(), "no device in import file");
    std::exit(1);
  }
  if (std::next(devices.begin()) != devices.end()) {
    emitError(module->getLoc(), "multiple devices in import file");
    std::exit(1);
  }

  return (*devices.begin()).getName();
}

void injectDevice(ModuleOp module) {
  auto devices = module.getOps<ktdf_arch::DeviceOp>();
  if (devices.empty()) {
    if (schedulerDeviceFilename.empty()) {
      auto diag = emitWarning(module.getLoc())
                  << "no 'ktdf_arch.device' is present in the module";
      diag.attachNote()
          << "consider using --device to specify a path to a device";
      return;
    }

    const std::filesystem::path filename(schedulerDeviceFilename.c_str());
    const auto import_path = std::filesystem::absolute(filename);
    const auto device_name = getDeviceName(import_path);
    OpBuilder builder(module.getBody(), module.getBody()->begin());
    ktdf_arch::DeviceOp::create(builder, builder.getUnknownLoc(), device_name,
                                import_path.native());
    return;
  }

  if (!schedulerDeviceFilename.empty()) {
    emitWarning((*devices.begin()).getLoc())
        << "a 'ktdf_arch.device' is present in the module, ignoring --device "
           "option";
  }
}

}  // namespace

namespace scheduler {

ManagedStatic<SchedulerOptMainConfigCLOptions> clOptionsConfig;

void SchedulerOptMainConfig::registerCLOptions(
    DialectRegistry& dialectRegistry) {
  (void)*clOptionsConfig;
  tracing::DebugConfig::registerCLOptions();
}

SchedulerOptMainConfig& SchedulerOptMainConfig::createFromCLOptions() {
  clOptionsConfig->inputFileName = schedulerInputFilename;
  clOptionsConfig->outputFileName = schedulerOutputFilename;
  clOptionsConfig->allowUnregisteredDialectsFlag =
      schedulerAllowUnregisteredDialects;
  clOptionsConfig->showDialectsFlag = schedulerShowDialects;
  clOptionsConfig->verifyPassesFlag = schedulerVerifyPasses;
  clOptionsConfig->printFinalIRFlag = schedulerPrintFinalIr;
  clOptionsConfig->splitInputFileFlag = schedulerSplitInputFile;
  clOptionsConfig->outputSplitMarkerFlag = schedulerOutputSplitMarker;
  clOptionsConfig->setDebugConfig(tracing::DebugConfig::createFromCLOptions());
  return *clOptionsConfig;
}

SchedulerOptMainConfig& SchedulerOptMainConfig::setPassPipelineParser(
    const PassPipelineCLParser& parser) {
  passPipelineCallback = [&](PassManager& pm) {
    auto errorHandler = [&](const Twine& msg) {
      emitError(UnknownLoc::get(pm.getContext())) << msg;
      return failure();
    };
    if (failed(parser.addToPipeline(pm, errorHandler))) return failure();
    if (shouldDumpPassPipeline()) {
      pm.dump();
      llvm::errs() << "\n";
    }
    return success();
  };
  return *this;
}

static LogicalResult printRegisteredDialects(DialectRegistry& registry) {
  llvm::outs() << "Available Dialects: ";
  interleave(registry.getDialectNames(), llvm::outs(), ",");
  llvm::outs() << "\n";
  return success();
}

static LogicalResult processBuffer(raw_ostream& outputStream,
                                   std::unique_ptr<MemoryBuffer> ownedBuffer,
                                   SchedulerOptMainConfig& config,
                                   DialectRegistry& registry,
                                   llvm::ThreadPoolInterface* threadPool) {
  auto sourceMgr = std::make_shared<SourceMgr>();
  sourceMgr->AddNewSourceBuffer(std::move(ownedBuffer), SMLoc());

  MLIRContext context(registry, MLIRContext::Threading::DISABLED);
  if (threadPool) context.setThreadPool(*threadPool);

  context.loadAllAvailableDialects();
  context.allowUnregisteredDialects(config.shouldAllowUnregisteredDialects());

  SourceMgrDiagnosticHandler sourceMgrHandler(*sourceMgr, &context);

  DefaultTimingManager tm;
  applyDefaultTimingManagerCLOptions(tm);
  TimingScope timing = tm.getRootScope();

  OwningOpRef<ModuleOp> module =
      parseSourceFile<ModuleOp>(*sourceMgr, &context);
  if (!module) return failure();

  injectDevice(*module);

  PassManager pm(&context);
  pm.enableVerifier(config.shouldVerifyPasses());
  if (failed(applyPassManagerCLOptions(pm))) return failure();
  pm.enableTiming(timing);
  if (failed(config.setupPassPipeline(pm))) return failure();
  if (failed(pm.run(*module))) return failure();

  if (config.shouldPrintFinalIr()) {
    module->print(outputStream);
    outputStream << '\n';
  }

  return success();
}

LogicalResult SchedulerOptMain(raw_ostream& outputStream,
                               std::unique_ptr<MemoryBuffer> buffer,
                               DialectRegistry& registry,
                               const SchedulerOptMainConfig& config) {
  if (config.shouldShowDialects()) return printRegisteredDialects(registry);

  ThreadPoolInterface* threadPool = nullptr;
  MLIRContext threadPoolCtx;
  if (threadPoolCtx.isMultithreadingEnabled())
    threadPool = &threadPoolCtx.getThreadPool();

  SchedulerOptMainConfig& mutableConfig =
      const_cast<SchedulerOptMainConfig&>(config);
  auto chunkFn = [&](std::unique_ptr<MemoryBuffer> chunkBuffer,
                     raw_ostream& os) {
    return processBuffer(os, std::move(chunkBuffer), mutableConfig, registry,
                         threadPool);
  };

  return splitAndProcessBuffer(std::move(buffer), chunkFn, outputStream,
                               config.inputSplitMarker(),
                               config.getOutputSplitMarkerFlag());
}

void registerAndParseCLIOptions(int argc, char** argv, llvm::StringRef toolName,
                                DialectRegistry& registry) {
  SchedulerOptMainConfig::registerCLOptions(registry);
  registerAsmPrinterCLOptions();
  registerMLIRContextCLOptions();
  registerPassManagerCLOptions();
  registerDefaultTimingManagerCLOptions();
  tracing::DebugCounter::registerCLOptions();

  std::string helpHeader = (toolName + "\nAvailable Dialects: ").str();
  {
    llvm::raw_string_ostream os(helpHeader);
    interleaveComma(registry.getDialectNames(), os,
                    [&](auto name) { os << name; });
  }
  cl::ParseCommandLineOptions(argc, argv, helpHeader);
}

LogicalResult SchedulerOptMain(int argc, char** argv, StringRef toolName,
                               DialectRegistry& registry) {
  scheduler::registerAndParseCLIOptions(argc, argv, toolName, registry);

  InitLLVM y(argc, argv);

  SchedulerOptMainConfig config = SchedulerOptMainConfig::createFromCLOptions();

  if (config.shouldShowDialects()) return printRegisteredDialects(registry);

  if (config.getInputFileName() == "-" &&
      sys::Process::FileDescriptorIsDisplayed(fileno(stdin)))
    llvm::errs() << "(processing input from stdin now, hit ctrl-c/ctrl-d to "
                    "interrupt)\n";

  std::string errorMessage;
  auto file = openInputFile(config.getInputFileName(), &errorMessage);
  if (!file) {
    llvm::errs() << errorMessage << "\n";
    return failure();
  }

  auto output = openOutputFile(config.getOutputFileName(), &errorMessage);
  if (!output) {
    llvm::errs() << errorMessage << "\n";
    return failure();
  }

  if (failed(SchedulerOptMain(output->os(), std::move(file), registry, config)))
    return failure();

  output->keep();
  return success();
}

}  // namespace scheduler
