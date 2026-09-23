add_subdirectory(Analysis)
add_subdirectory(CAPI)
add_subdirectory(Conversion)
add_subdirectory(Dialect)
add_subdirectory(Transforms)
add_subdirectory(Utils)

#-------------------------------------------------------------------------------
# SchedulerRegisterEverything
#-------------------------------------------------------------------------------

# IMPORTANT: Make sure to keep this in sync with the `RegisterEverything.cpp`
#            source file! If a new dialect, pass or extension is added, its
#            owning library must be added to one of the following lists.

# List all relevant exported MLIR targets.
set(mlir_dialect_libs
  MLIRAffineDialect
  MLIRArithDialect
  MLIRBufferizationDialect
  MLIRFuncDialect
  MLIRLinalgDialect
  MLIRMathDialect
  MLIRMemRefDialect
  MLIRSCFDialect
  MLIRSCFTransforms
  MLIRTensorDialect
)
set(mlir_extension_libs
  MLIRArithValueBoundsOpInterfaceImpl
)
set(mlir_conversion_libs)

# Get the names of all relevant exported KTIR targets.
get_property(ktir_dialect_libs GLOBAL PROPERTY KTIR_DIALECT_LIBS)
get_property(ktir_extension_libs GLOBAL PROPERTY KTIR_EXTENSION_LIBS)
get_property(ktir_conversion_libs GLOBAL PROPERTY KTIR_CONVERSION_LIBS)

# Get the names of all relevant exported DataflowSchedulerDialects targets.
get_property(external_dialect_libs GLOBAL PROPERTY DataflowSchedulerDialects_DIALECT_LIBS)
get_property(external_extension_libs GLOBAL PROPERTY DataflowSchedulerDialects_EXTENSION_LIBS)
get_property(external_conversion_libs GLOBAL PROPERTY DataflowSchedulerDialects_CONVERSION_LIBS)

# Get the names of all relevant exported DataflowScheduler targets.
get_property(scheduler_dialect_libs GLOBAL PROPERTY DataflowScheduler_DIALECT_LIBS)
get_property(scheduler_extension_libs GLOBAL PROPERTY DataflowScheduler_EXTENSION_LIBS)
get_property(scheduler_conversion_libs GLOBAL PROPERTY DataflowScheduler_CONVERSION_LIBS)

add_dataflow_scheduler_library(DataflowSchedulerRegisterEverything
    RegisterEverything.cpp

  PARTIAL_SOURCES_INTENDED

  EXPORT_NAME RegisterEverything

  LINK_LIBS PUBLIC
    ${ktir_dialect_libs}
    ${ktir_extension_libs}
    ${ktir_conversion_libs}

    ${external_dialect_libs}
    ${external_extension_libs}
    ${external_conversion_libs}

    ${scheduler_dialect_libs}
    ${scheduler_extension_libs}
    ${scheduler_conversion_libs}

  LINK_MLIR_LIBS
    ${mlir_dialect_libs}
    ${mlir_extension_libs}
    ${mlir_conversion_libs}
)

#-------------------------------------------------------------------------------
# SchedulerPipeline
#-------------------------------------------------------------------------------

add_dataflow_scheduler_library(DataflowSchedulerPipeline
    Pipeline.cpp

  EXPORT_NAME Pipeline

  PARTIAL_SOURCES_INTENDED

  LINK_LIBS PUBLIC
    DataflowScheduler::RegisterEverything
)
