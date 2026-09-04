// RUN: dataflow-scheduler-opt --map-reduction-partials %s | FileCheck %s

// CHECK: #[[$ATTR_0:.+]] = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
// CHECK: #[[$ATTR_1:.+]] = affine_map<(d0, d1, d2) -> (d1, d2)>
// CHECK: #[[$ATTR_2:.+]] = affine_map<(d0, d1, d2) -> (d0, d1)>
// CHECK: #[[$ATTR_3:.+]] = affine_map<(d0, d1, d2) -> (d0, d2)>
// CHECK: #[[$ATTR_4:.+]] = affine_set<(d0, d1, d2) : (d0 >= 0, -d0 + 1 >= 0, d1 >= 0, -d1 + 255 >= 0, d2 >= 0, -d2 + 63 >= 0)>
// CHECK: #[[$ATTR_5:.+]] = affine_set<(d0, d1) : (d0 >= 0, -d0 + 255 >= 0, d1 >= 0, -d1 + 63 >= 0)>
// CHECK-LABEL:   module {
// CHECK:     func.func @sum_1core() attributes {grid = [1]} {
// CHECK:       call @local_schedule_0() : () -> ()
// CHECK:       return
// CHECK:     }
// CHECK:     func.func private @local_schedule_0()
// CHECK:   }
// CHECK:   ktdf_arch.device @sample_device import("../../Dialect/KTDFArch/sample_device.mlir")

// CHECK-LABEL:   module @local_schedule_0 {
// CHECK-NEXT:     func.func @local_schedule_0() attributes {grid = [1]} {
// CHECK-NEXT:       %[[CONSTANT_0:.*]] = arith.constant 0 : index
// CHECK-NEXT:       %[[CONSTANT_1:.*]] = arith.constant 1 : index
// CHECK-NEXT:       %[[CONSTANT_2:.*]] = arith.constant 8589934592 : index
// CHECK-NEXT:       %[[CONSTANT_3:.*]] = arith.constant 256 : index
// CHECK-NEXT:       %[[CONSTRUCT_MEMORY_VIEW_0:.*]] = ktdp.construct_memory_view %[[CONSTANT_0]], sizes: [2, 256, 64], strides: [16384, 64, 1] {coordinate_set = #[[$ATTR_4]], memory_space = #ktdp.memory_space<global>} : memref<2x256x64xf16>
// CHECK-NEXT:       %[[CONSTRUCT_MEMORY_VIEW_1:.*]] = ktdp.construct_memory_view %[[CONSTANT_2]], sizes: [256, 64], strides: [64, 1] {coordinate_set = #[[$ATTR_5]], memory_space = #ktdp.memory_space<global>} : memref<256x64xf16>
// CHECK-NEXT:       %[[MEMORY_SPACE_CAST_0:.*]] = memref.memory_space_cast %[[CONSTRUCT_MEMORY_VIEW_0]] : memref<2x256x64xf16> to memref<2x256x64xf16, #ktdp.memory_space<global>>
// CHECK-NEXT:       %[[REINTERPRET_CAST_0:.*]] = memref.reinterpret_cast %[[MEMORY_SPACE_CAST_0]] to offset: [0], sizes: [2, 256, 64], strides: [16384, 64, 1] : memref<2x256x64xf16, #ktdp.memory_space<global>> to memref<2x256x64xf16, strided<[16384, 64, 1]>, #ktdp.memory_space<global>>
// CHECK-NEXT:       %[[CAST_0:.*]] = memref.cast %[[REINTERPRET_CAST_0]] : memref<2x256x64xf16, strided<[16384, 64, 1]>, #ktdp.memory_space<global>> to memref<2x256x64xf16, strided<[16384, 64, 1], offset: ?>, #ktdp.memory_space<global>>
// CHECK-NEXT:       %[[MEMORY_SPACE_CAST_1:.*]] = memref.memory_space_cast %[[CONSTRUCT_MEMORY_VIEW_1]] : memref<256x64xf16> to memref<256x64xf16, #ktdp.memory_space<global>>
// CHECK-NEXT:       %[[REINTERPRET_CAST_1:.*]] = memref.reinterpret_cast %[[MEMORY_SPACE_CAST_1]] to offset: [0], sizes: [256, 64], strides: [64, 1] : memref<256x64xf16, #ktdp.memory_space<global>> to memref<256x64xf16, strided<[64, 1]>, #ktdp.memory_space<global>>
// CHECK-NEXT:       %[[CAST_1:.*]] = memref.cast %[[REINTERPRET_CAST_1]] : memref<256x64xf16, strided<[64, 1]>, #ktdp.memory_space<global>> to memref<256x64xf16, strided<[64, 1], offset: ?>, #ktdp.memory_space<global>>
// CHECK-NEXT:       ktdf.pipeline {
// CHECK-NEXT:         %[[PRIVATE_0:.*]]:4 = ktdf.private -> (memref<256x2x1x64xf16, #ktdp.memory_space<ct_local>>, memref<256x1x64xf16, #ktdp.memory_space<ct_local>>, !ktdf.token, !ktdf.token) {
// CHECK-NEXT:           %[[ALLOC_0:.*]] = memref.alloc() : memref<256x2x1x64xf16, #ktdp.memory_space<ct_local>>
// CHECK-NEXT:           %[[ALLOC_1:.*]] = memref.alloc() : memref<256x1x64xf16, #ktdp.memory_space<ct_local>>
// CHECK-NEXT:           %[[CREATE_TOKEN_0:.*]] = ktdf.create_token : !ktdf.token
// CHECK-NEXT:           %[[CREATE_TOKEN_1:.*]] = ktdf.create_token : !ktdf.token
// CHECK-NEXT:           ktdf.private_yield %[[ALLOC_0]], %[[ALLOC_1]], %[[CREATE_TOKEN_0]], %[[CREATE_TOKEN_1]] : memref<256x2x1x64xf16, #ktdp.memory_space<ct_local>>, memref<256x1x64xf16, #ktdp.memory_space<ct_local>>, !ktdf.token, !ktdf.token
// CHECK-NEXT:         }
// CHECK-NEXT:         ktdf.stage depends_in(none) depends_out(%[[VAL_0:.*]]#2) {
// CHECK-NEXT:           scf.for %[[VAL_1:.*]] = %[[CONSTANT_0]] to %[[CONSTANT_3]] step %[[CONSTANT_1]] {
// CHECK-NEXT:             %[[SUBI_0:.*]] = arith.subi %[[VAL_1]], %[[CONSTANT_0]] : index
// CHECK-NEXT:             %[[DIVSI_0:.*]] = arith.divsi %[[SUBI_0]], %[[CONSTANT_1]] : index
// CHECK-NEXT:             ktdf.data_transfer from %[[CAST_0]][0, %[[VAL_1]], 0] size [2, 1, 64] to %[[VAL_0]]#0{{\[}}%[[DIVSI_0]], 0, 0, 0] size [1, 2, 1, 64] : memref<2x256x64xf16, strided<[16384, 64, 1], offset: ?>, #ktdp.memory_space<global>>, memref<256x2x1x64xf16, #ktdp.memory_space<ct_local>>
// CHECK-NEXT:           } {loop_type = #ktdf.loop_type<parallel_loop>}
// CHECK-NEXT:         } {applicable_units = ["MNILU"]}
// CHECK-NEXT:         ktdf.stage depends_in(%[[VAL_2:.*]]#2) depends_out(%[[VAL_2]]#3) {
// CHECK-NEXT:           scf.for %[[VAL_3:.*]] = %[[CONSTANT_0]] to %[[CONSTANT_3]] step %[[CONSTANT_1]] {
// CHECK-NEXT:             %[[CONSTANT_4:.*]] = arith.constant 0 : index
// CHECK-NEXT:             %[[CONSTANT_5:.*]] = arith.constant 1 : index
// CHECK-NEXT:             %[[CONSTANT_6:.*]] = arith.constant 1 : index
// CHECK-NEXT:             ktdf.pipeline {
// CHECK-NEXT:               %[[PRIVATE_1:.*]]:4 = ktdf.private -> (!ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>, !ktdf.fifo.slot<"SFU" -> "L1SU", 64xf16>, !ktdf.token, !ktdf.token) {
// CHECK-NEXT:                 %[[FIFO_0:.*]] = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>
// CHECK-NEXT:                 %[[FIFO_1:.*]] = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"SFU" -> "L1SU", 64xf16>
// CHECK-NEXT:                 %[[CREATE_TOKEN_2:.*]] = ktdf.create_token : !ktdf.token
// CHECK-NEXT:                 %[[CREATE_TOKEN_3:.*]] = ktdf.create_token : !ktdf.token
// CHECK-NEXT:                 ktdf.private_yield %[[FIFO_0]], %[[FIFO_1]], %[[CREATE_TOKEN_2]], %[[CREATE_TOKEN_3]] : !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>, !ktdf.fifo.slot<"SFU" -> "L1SU", 64xf16>, !ktdf.token, !ktdf.token
// CHECK-NEXT:               }
// CHECK-NEXT:               ktdf.stage depends_in(none) depends_out(%[[VAL_4:.*]]#2) {
// CHECK-NEXT:                 %[[SUBI_1:.*]] = arith.subi %[[VAL_3]], %[[CONSTANT_0]] : index
// CHECK-NEXT:                 %[[DIVSI_1:.*]] = arith.divsi %[[SUBI_1]], %[[CONSTANT_1]] : index
// CHECK-NEXT:                 %[[CONSTANT_7:.*]] = arith.constant 2 : index
// CHECK-NEXT:                 scf.for %[[VAL_5:.*]] = %[[CONSTANT_4]] to %[[CONSTANT_7]] step %[[CONSTANT_5]] {
// CHECK-NEXT:                   ktdf.data_transfer from %[[VAL_2]]#0{{\[}}%[[DIVSI_1]], %[[VAL_5]], 0, 0] size [1, 1, 1, 64] to %[[VAL_4]]#0 size [64] : memref<256x2x1x64xf16, #ktdp.memory_space<ct_local>>, !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>
// CHECK-NEXT:                 } {loop_type = #ktdf.loop_type<reduction_loop>}
// CHECK-NEXT:               } {applicable_units = ["L1LU"]}
// CHECK-NEXT:               ktdf.stage depends_in(%[[VAL_6:.*]]#2) depends_out(%[[VAL_6]]#3) {
// CHECK-NEXT:                 %[[ALLOC_2:.*]] = memref.alloc() : memref<1x64xf16, "SFU_REG">
// CHECK-NEXT:                 %[[CONSTANT_8:.*]] = arith.constant 0.000000e+00 : f16
// CHECK-NEXT:                 %[[IDENTITY:.*]] = memref.alloc() : memref<1x64xf16, "SFU_REG">
// CHECK-NEXT:                 linalg.fill ins(%[[CONSTANT_8]] : f16) outs(%[[IDENTITY]] : memref<1x64xf16, "SFU_REG">)
// CHECK-NEXT:                 memref.copy %[[IDENTITY]], %[[ALLOC_2]] : memref<1x64xf16, "SFU_REG"> to memref<1x64xf16, "SFU_REG">
// CHECK-NEXT:                 %[[CONSTANT_9:.*]] = arith.constant 2 : index
// CHECK-NEXT:                 scf.for %[[VAL_7:.*]] = %[[CONSTANT_4]] to %[[CONSTANT_9]] step %[[CONSTANT_5]] {
// CHECK-NEXT:                   %[[READ_FROM_FIFO_0:.*]] = ktdf.read_from_fifo %[[VAL_6]]#0 : <"L1LU" -> "SFU", 64xf16> -> memref<1x1x64xf16>
// CHECK-NEXT:                   linalg.generic {indexing_maps = [#[[$ATTR_0]], #[[$ATTR_1]]], iterator_types = ["reduction", "parallel", "parallel"]} ins(%[[READ_FROM_FIFO_0]] : memref<1x1x64xf16>) outs(%[[ALLOC_2]] : memref<1x64xf16, "SFU_REG">) {
// CHECK-NEXT:                   ^bb0(%[[VAL_8:.*]]: f16, %[[VAL_9:.*]]: f16):
// CHECK-NEXT:                     %[[ADDF_0:.*]] = arith.addf %[[VAL_8]], %[[VAL_9]] : f16
// CHECK-NEXT:                     linalg.yield %[[ADDF_0]] : f16
// CHECK-NEXT:                   }
// CHECK-NEXT:                 } {loop_type = #ktdf.loop_type<reduction_loop>}
// CHECK-NEXT:                 %[[SUBVIEW_0:.*]] = memref.subview %[[ALLOC_2]][0, 0] [1, 1] [1, 1] : memref<1x64xf16, "SFU_REG"> to memref<1x1xf16, strided<[64, 1]>, "SFU_REG">
// CHECK-NEXT:                 linalg.generic {indexing_maps = [#[[$ATTR_2]], #[[$ATTR_3]]], iterator_types = ["parallel", "reduction", "parallel"]} ins(%[[ALLOC_2]] : memref<1x64xf16, "SFU_REG">) outs(%[[SUBVIEW_0]] : memref<1x1xf16, strided<[64, 1]>, "SFU_REG">) {
// CHECK-NEXT:                 ^bb0(%[[VAL_10:.*]]: f16, %[[VAL_11:.*]]: f16):
// CHECK-NEXT:                   %[[ADDF_1:.*]] = arith.addf %[[VAL_10]], %[[VAL_11]] : f16
// CHECK-NEXT:                   linalg.yield %[[ADDF_1]] : f16
// CHECK-NEXT:                 }
// CHECK-NEXT:                 ktdf.write_to_fifo %[[ALLOC_2]], %[[VAL_6]]#1 : memref<1x64xf16, "SFU_REG">, <"SFU" -> "L1SU", 64xf16>
// CHECK-NEXT:               } {applicable_units = ["SFU"]}
// CHECK-NEXT:               ktdf.stage depends_in(%[[VAL_12:.*]]#3) depends_out(none) {
// CHECK-NEXT:                 %[[CONSTANT_10:.*]] = arith.constant 2 : index
// CHECK-NEXT:                 scf.for %[[VAL_13:.*]] = %[[CONSTANT_4]] to %[[CONSTANT_10]] step %[[CONSTANT_5]] {
// CHECK-NEXT:                   %[[SUBI_2:.*]] = arith.subi %[[VAL_3]], %[[CONSTANT_0]] : index
// CHECK-NEXT:                   %[[DIVSI_2:.*]] = arith.divsi %[[SUBI_2]], %[[CONSTANT_1]] : index
// CHECK-NEXT:                   %[[CMPI_0:.*]] = arith.cmpi eq, %[[VAL_13]], %[[CONSTANT_6]] : index
// CHECK-NEXT:                   scf.if %[[CMPI_0]] {
// CHECK-NEXT:                     ktdf.data_transfer from %[[VAL_12]]#1 size [1, 64] to %[[VAL_2]]#1{{\[}}%[[DIVSI_2]], 0, 0] size [1, 1, 64] : !ktdf.fifo.slot<"SFU" -> "L1SU", 64xf16>, memref<256x1x64xf16, #ktdp.memory_space<ct_local>>
// CHECK-NEXT:                   }
// CHECK-NEXT:                 } {loop_type = #ktdf.loop_type<reduction_loop>}
// CHECK-NEXT:               } {applicable_units = ["L1SU"]}
// CHECK-NEXT:             }
// CHECK-NEXT:           } {loop_type = #ktdf.loop_type<parallel_loop>}
// CHECK-NEXT:         } {applicable_units = ["L1LU", "SFU", "L1SU"]}
// CHECK-NEXT:         ktdf.stage depends_in(%[[VAL_14:.*]]#3) depends_out(none) {
// CHECK-NEXT:           scf.for %[[VAL_15:.*]] = %[[CONSTANT_0]] to %[[CONSTANT_3]] step %[[CONSTANT_1]] {
// CHECK-NEXT:             %[[SUBI_3:.*]] = arith.subi %[[VAL_15]], %[[CONSTANT_0]] : index
// CHECK-NEXT:             %[[DIVSI_3:.*]] = arith.divsi %[[SUBI_3]], %[[CONSTANT_1]] : index
// CHECK-NEXT:             ktdf.data_transfer from %[[VAL_14]]#1{{\[}}%[[DIVSI_3]], 0, 0] size [1, 1, 64] to %[[CAST_1]]{{\[}}%[[VAL_15]], %[[CONSTANT_0]] * 64] size [1, 64] : memref<256x1x64xf16, #ktdp.memory_space<ct_local>>, memref<256x64xf16, strided<[64, 1], offset: ?>, #ktdp.memory_space<global>>
// CHECK-NEXT:           } {loop_type = #ktdf.loop_type<parallel_loop>}
// CHECK-NEXT:         } {applicable_units = ["MNISU"]}
// CHECK-NEXT:       }
// CHECK-NEXT:       return
// CHECK-NEXT:     }
// CHECK-NEXT:   }



// Tests the inner-dim generic path: after SplitReductionInnerOuterDim +
// ReductionLoopExposure the SFU compute stage contains two linalg.generic ops:
//   G1 (outer-dim, loop-exposed): tensor<1x1x64xf16> -> tensor<1x64xf16>
//   G2 (inner-dim, plain):        tensor<1x64xf16>   -> tensor<1x64xf16>
//
// MapReductionPartials must:
//   - Lower G1 into a memref.alloc + linalg.fill + scf.for with buffer generic.
//   - Lower G2 using the G1 alloc as ins, a rank-reducing subview as outs,
//     and write the subview to the FIFO.



#map = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d1, d2)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
#map3 = affine_map<(d0, d1, d2) -> (d0, d2)>
#set = affine_set<(d0, d1, d2) : (d0 >= 0, -d0 + 1 >= 0, d1 >= 0, -d1 + 255 >= 0, d2 >= 0, -d2 + 63 >= 0)>
#set1 = affine_set<(d0, d1) : (d0 >= 0, -d0 + 255 >= 0, d1 >= 0, -d1 + 63 >= 0)>
module {
  module {
    func.func @sum_1core() attributes {grid = [1]} {
      call @local_schedule_0() : () -> ()
      return
    }
    func.func private @local_schedule_0()
  }
  ktdf_arch.device @sample_device import("../../Dialect/KTDFArch/sample_device.mlir")
  module @local_schedule_0 {
    func.func @local_schedule_0() attributes {grid = [1]} {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c8589934592 = arith.constant 8589934592 : index
      %c256 = arith.constant 256 : index
      %0 = ktdp.construct_memory_view %c0, sizes: [2, 256, 64], strides: [16384, 64, 1] {coordinate_set = #set, memory_space = #ktdp.memory_space<global>} : memref<2x256x64xf16>
      %1 = ktdp.construct_memory_view %c8589934592, sizes: [256, 64], strides: [64, 1] {coordinate_set = #set1, memory_space = #ktdp.memory_space<global>} : memref<256x64xf16>
      %memspacecast = memref.memory_space_cast %0 : memref<2x256x64xf16> to memref<2x256x64xf16, #ktdp.memory_space<global>>
      %reinterpret_cast = memref.reinterpret_cast %memspacecast to offset: [0], sizes: [2, 256, 64], strides: [16384, 64, 1] : memref<2x256x64xf16, #ktdp.memory_space<global>> to memref<2x256x64xf16, strided<[16384, 64, 1]>, #ktdp.memory_space<global>>
      %cast = memref.cast %reinterpret_cast : memref<2x256x64xf16, strided<[16384, 64, 1]>, #ktdp.memory_space<global>> to memref<2x256x64xf16, strided<[16384, 64, 1], offset: ?>, #ktdp.memory_space<global>>
      %memspacecast_0 = memref.memory_space_cast %1 : memref<256x64xf16> to memref<256x64xf16, #ktdp.memory_space<global>>
      %reinterpret_cast_1 = memref.reinterpret_cast %memspacecast_0 to offset: [0], sizes: [256, 64], strides: [64, 1] : memref<256x64xf16, #ktdp.memory_space<global>> to memref<256x64xf16, strided<[64, 1]>, #ktdp.memory_space<global>>
      %cast_2 = memref.cast %reinterpret_cast_1 : memref<256x64xf16, strided<[64, 1]>, #ktdp.memory_space<global>> to memref<256x64xf16, strided<[64, 1], offset: ?>, #ktdp.memory_space<global>>
      ktdf.pipeline {
        %2:4 = ktdf.private -> (memref<256x2x1x64xf16, #ktdp.memory_space<ct_local>>, memref<256x1x64xf16, #ktdp.memory_space<ct_local>>, !ktdf.token, !ktdf.token) {
          %alloc = memref.alloc() : memref<256x2x1x64xf16, #ktdp.memory_space<ct_local>>
          %alloc_3 = memref.alloc() : memref<256x1x64xf16, #ktdp.memory_space<ct_local>>
          %3 = ktdf.create_token : !ktdf.token
          %4 = ktdf.create_token : !ktdf.token
          ktdf.private_yield %alloc, %alloc_3, %3, %4 : memref<256x2x1x64xf16, #ktdp.memory_space<ct_local>>, memref<256x1x64xf16, #ktdp.memory_space<ct_local>>, !ktdf.token, !ktdf.token
        }
        ktdf.stage depends_in(none) depends_out(%2#2) {
          scf.for %arg0 = %c0 to %c256 step %c1 {
            %3 = arith.subi %arg0, %c0 : index
            %4 = arith.divsi %3, %c1 : index
            ktdf.data_transfer from %cast[0, %arg0, 0] size [2, 1, 64] to %2#0[%4, 0, 0, 0] size [1, 2, 1, 64] : memref<2x256x64xf16, strided<[16384, 64, 1], offset: ?>, #ktdp.memory_space<global>>, memref<256x2x1x64xf16, #ktdp.memory_space<ct_local>>
          } {loop_type = #ktdf.loop_type<parallel_loop>}
        } {applicable_units = ["MNILU"]}
        ktdf.stage depends_in(%2#2) depends_out(%2#3) {
          scf.for %arg0 = %c0 to %c256 step %c1 {
            %c0_3 = arith.constant 0 : index
            %c1_4 = arith.constant 1 : index
            %c1_5 = arith.constant 1 : index
            ktdf.pipeline {
              %3:4 = ktdf.private -> (!ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>, !ktdf.fifo.slot<"SFU" -> "L1SU", 64xf16>, !ktdf.token, !ktdf.token) {
                %4 = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>
                %5 = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"SFU" -> "L1SU", 64xf16>
                %6 = ktdf.create_token : !ktdf.token
                %7 = ktdf.create_token : !ktdf.token
                ktdf.private_yield %4, %5, %6, %7 : !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>, !ktdf.fifo.slot<"SFU" -> "L1SU", 64xf16>, !ktdf.token, !ktdf.token
              }
              ktdf.stage depends_in(none) depends_out(%3#2) {
                %4 = arith.subi %arg0, %c0 : index
                %5 = arith.divsi %4, %c1 : index
                %c2 = arith.constant 2 : index
                scf.for %arg1 = %c0_3 to %c2 step %c1_4 {
                  ktdf.data_transfer from %2#0[%5, %arg1, 0, 0] size [1, 1, 1, 64] to %3#0 size [64] : memref<256x2x1x64xf16, #ktdp.memory_space<ct_local>>, !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>
                } {loop_type = #ktdf.loop_type<reduction_loop>}
              } {applicable_units = ["L1LU"]}
              ktdf.stage depends_in(%3#2) depends_out(%3#3) {
                %4 = tensor.empty() : tensor<1x64xf16>
                %c2 = arith.constant 2 : index
                %5 = scf.for %arg1 = %c0_3 to %c2 step %c1_4 iter_args(%arg2 = %4) -> (tensor<1x64xf16>) {
                  %8 = ktdf.read_from_fifo %3#0 : <"L1LU" -> "SFU", 64xf16> -> tensor<1x1x64xf16>
                  %9 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction", "parallel", "parallel"]} ins(%8 : tensor<1x1x64xf16>) outs(%arg2 : tensor<1x64xf16>) {
                  ^bb0(%in: f16, %out: f16):
                    %10 = arith.addf %in, %out : f16
                    linalg.yield %10 : f16
                  } -> tensor<1x64xf16>
                  scf.yield %9 : tensor<1x64xf16>
                } {loop_type = #ktdf.loop_type<reduction_loop>}
                %6 = tensor.empty() : tensor<1x64xf16>
                %7 = linalg.generic {indexing_maps = [#map2, #map3], iterator_types = ["parallel", "reduction", "parallel"]} ins(%5 : tensor<1x64xf16>) outs(%6 : tensor<1x64xf16>) {
                ^bb0(%in: f16, %out: f16):
                  %8 = arith.addf %in, %out : f16
                  linalg.yield %8 : f16
                } -> tensor<1x64xf16>
                ktdf.write_to_fifo %7, %3#1 : tensor<1x64xf16>, <"SFU" -> "L1SU", 64xf16>
              } {applicable_units = ["SFU"]}
              ktdf.stage depends_in(%3#3) depends_out(none) {
                %c2 = arith.constant 2 : index
                scf.for %arg1 = %c0_3 to %c2 step %c1_4 {
                  %4 = arith.subi %arg0, %c0 : index
                  %5 = arith.divsi %4, %c1 : index
                  %6 = arith.cmpi eq, %arg1, %c1_5 : index
                  scf.if %6 {
                    ktdf.data_transfer from %3#1 size [64] to %2#1[%5, 0, 0] size [1, 1, 64] : !ktdf.fifo.slot<"SFU" -> "L1SU", 64xf16>, memref<256x1x64xf16, #ktdp.memory_space<ct_local>>
                  }
                } {loop_type = #ktdf.loop_type<reduction_loop>}
              } {applicable_units = ["L1SU"]}
            }
          } {loop_type = #ktdf.loop_type<parallel_loop>}
        } {applicable_units = ["L1LU", "SFU", "L1SU"]}
        ktdf.stage depends_in(%2#3) depends_out(none) {
          scf.for %arg0 = %c0 to %c256 step %c1 {
            %3 = arith.subi %arg0, %c0 : index
            %4 = arith.divsi %3, %c1 : index
            ktdf.data_transfer from %2#1[%4, 0, 0] size [1, 1, 64] to %cast_2[%arg0, %c0 * 64] size [1, 64] : memref<256x1x64xf16, #ktdp.memory_space<ct_local>>, memref<256x64xf16, strided<[64, 1], offset: ?>, #ktdp.memory_space<global>>
          } {loop_type = #ktdf.loop_type<parallel_loop>}
        } {applicable_units = ["MNISU"]}
      }
      return
    }
  }
}
