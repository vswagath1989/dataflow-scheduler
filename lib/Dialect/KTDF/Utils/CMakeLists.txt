add_dataflow_scheduler_library(DataflowSchedulerKTDFUtils
    Utils.cpp
  
  ADDITIONAL_HEADER_DIRS
    ${DataflowScheduler_SOURCE_DIR}/include/dataflow-scheduler/Dialect/KTDF/Utils

  GROUP DIALECT
  EXPORT_NAME KTDFUtils

  LINK_LIBS PUBLIC
    KTIR::KtdpDialect
    DataflowSchedulerDialects::DataflowDialect
    DataflowSchedulerDialects::KTDFDialect
    DataflowSchedulerDialects::UniformDialect

  LINK_MLIR_LIBS
    MLIRLinalgDialect
    MLIRMemRefDialect
)
