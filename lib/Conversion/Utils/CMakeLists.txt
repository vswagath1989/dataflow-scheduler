add_dataflow_scheduler_library(DataflowSchedulerConversionUtils
    Utils.cpp

  ADDITIONAL_HEADER_DIRS
    ${DataflowScheduler_SOURCE_DIR}/include/dataflow-scheduler/Conversion/Utils

  GROUP CONVERSION
  EXPORT_NAME ConversionUtils

  LINK_LIBS PUBLIC
    DataflowSchedulerDialects::AgenDialect
    DataflowSchedulerDialects::DataflowDialect
    DataflowSchedulerDialects::KTDFDialect
    DataflowSchedulerDialects::KTDFArchAnalysis
    DataflowSchedulerDialects::UniformDialect
    DataflowSchedulerDialects::VectorChainDialect

  LINK_MLIR_LIBS
    MLIRArithDialect
)
