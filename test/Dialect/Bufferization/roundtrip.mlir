// RUN: dataflow-scheduler-opt %s | dataflow-scheduler-opt | FileCheck %s

// The bufferization dialect is registered so a device pattern can hand a
// memref to a tensor-typed consumer without rewriting that consumer, which is
// a rewrite PDL cannot express.  It is only reachable from the drivers if the
// dialect is in the registry, so pin that here.

// CHECK-LABEL: func.func @to_tensor(
// CHECK-SAME:    %[[BUF:.*]]: memref<1x1x32xf32>) -> tensor<1x1x32xf32>
func.func @to_tensor(%buffer: memref<1x1x32xf32>) -> tensor<1x1x32xf32> {
  // CHECK:         %[[T:.*]] = bufferization.to_tensor %[[BUF]] : memref<1x1x32xf32> to tensor<1x1x32xf32>
  %0 = bufferization.to_tensor %buffer : memref<1x1x32xf32> to tensor<1x1x32xf32>
  // CHECK:         return %[[T]] : tensor<1x1x32xf32>
  return %0 : tensor<1x1x32xf32>
}

// The buffer normally sits in a register file, and the memory space plays no
// part in the type match -- only shape and element type do.
// CHECK-LABEL: func.func @to_tensor_from_register_file(
// CHECK-SAME:    %[[BUF:.*]]: memref<1x1x32xf32, "SFU_REG">) -> tensor<1x1x32xf32>
func.func @to_tensor_from_register_file(
    %buffer: memref<1x1x32xf32, "SFU_REG">) -> tensor<1x1x32xf32> {
  // CHECK:         %[[T:.*]] = bufferization.to_tensor %[[BUF]] : memref<1x1x32xf32, "SFU_REG"> to tensor<1x1x32xf32>
  %0 = bufferization.to_tensor %buffer
    : memref<1x1x32xf32, "SFU_REG"> to tensor<1x1x32xf32>
  // CHECK:         return %[[T]] : tensor<1x1x32xf32>
  return %0 : tensor<1x1x32xf32>
}
