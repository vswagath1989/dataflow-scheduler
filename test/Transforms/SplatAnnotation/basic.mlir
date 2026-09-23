// RUN: dataflow-scheduler-opt --splat-annotation %s | FileCheck %s

// Verify that SplatAnnotationPass widens a splat load to the arch's access
// granularity and marks the reads a sub-SIMD splat still has to finish.
//
// The LXLU / SFP naming used in production is mapped to the sample_device
// equivalents L1LU / SFU so this test is self-contained.
//
// === f32 case ===
// L1LU word_size=1 byte, access_granularity for "L1": [64, 8, 2] words.
// For f32 (4 bytes/elem): elem_words=4, min_words=4*1=4, fitAccess(4)->8 words
//   load_elements = 8 / 4 = 2   (widening occurs: 2 > 1)
//   => splat source size widened from [1,1,1] to [1,1,2]
//   => the consuming read carries the splat mode that finishes the splat
//
// === f16 case ===
// For f16 (2 bytes/elem): elem_words=2, min_words=2, fitAccess(2)->2 words
//   load_elements = 2 / 2 = 1   (no widening)
//   => source size unchanged, and no read carries a mode
//
// Nothing here converts the receive.  The shuffle the mode names is emitted by
// the lowering to DFIR, and where the buffer it works on lives is the device's
// to decide in its patterns ["post_scheduling"] block.

ktdf_arch.device @sample_device attributes {mem_space_mapping = #ktdf_arch.map<"DDR" = "DDR", "L1" = "L1">} import("../../Dialect/KTDFArch/sample_device.mlir")

// ---------------------------------------------------------------------------
// f32: widening occurs — the splat load grows and the read names the shuffle.
// ---------------------------------------------------------------------------
// CHECK-LABEL: func.func @splat_f32
// Load stage: non-splat transfer unchanged; splat transfer source widened.
// CHECK:       ktdf.data_transfer from %{{.*}}[0, 0, 0] size [1, 1, 32] to
// CHECK:       ktdf.data_transfer from %{{.*}}[0, 0, 0] size [1, 1, 2] to
// CHECK-SAME:    transfer_mode = "splat"

// Compute stage: the splat read carries the mode and both stay tensor-typed.
// CHECK:       %[[R0:.*]] = ktdf.read_from_fifo %{{.*}}#0
// CHECK-SAME:    -> tensor<1x1x32xf32>
// CHECK:       %[[R1:.*]] = ktdf.read_from_fifo %{{.*}}#1
// CHECK-SAME:    {splat = #ktdf.splat<first_subsimd_lane_to_all_subsimd_lanes>}
// CHECK-SAME:    -> tensor<1x1x32xf32>

// The generic is untouched: still tensors, still the map legalization left.
// CHECK:       linalg.generic
// CHECK-SAME:    ins(%[[R0]], %[[R1]] : tensor<1x1x32xf32>, tensor<1x1x32xf32>)
// CHECK-NOT:   memref.copy

#map_f32  = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map1_f32 = affine_map<(d0, d1, d2) -> (0, d1, d2)>

module {
  func.func @splat_f32(%alloc_a: memref<1x1x32xf32, "L1">,
                       %alloc_b: memref<1x1x1xf32, "L1">) {
    %out = tensor.empty() : tensor<1x1x32xf32>
    ktdf.pipeline {
      %slots:5 = ktdf.private -> (
          !ktdf.fifo.slot<"L1LU" -> "SFU", 32xf32>,
          !ktdf.fifo.slot<"L1LU" -> "SFU", 32xf32>,
          !ktdf.fifo.slot<"SFU" -> "L1SU", 32xf32>,
          !ktdf.token,
          !ktdf.token) {
        %f0:2 = ktdf.fifo.allocate() ->
            !ktdf.fifo.slot<"L1LU" -> "SFU", 32xf32>,
            !ktdf.fifo.slot<"L1LU" -> "SFU", 32xf32>
        %f1 = ktdf.fifo.allocate() ->
            !ktdf.fifo.slot<"SFU" -> "L1SU", 32xf32>
        %t0 = ktdf.create_token : !ktdf.token
        %t1 = ktdf.create_token : !ktdf.token
        ktdf.private_yield %f0#0, %f0#1, %f1, %t0, %t1 :
            !ktdf.fifo.slot<"L1LU" -> "SFU", 32xf32>,
            !ktdf.fifo.slot<"L1LU" -> "SFU", 32xf32>,
            !ktdf.fifo.slot<"SFU" -> "L1SU", 32xf32>,
            !ktdf.token, !ktdf.token
      }
      // Load stage: non-splat transfer (full row) + splat transfer (scalar).
      ktdf.stage depends_in(none) depends_out(%slots#3) {
        ktdf.data_transfer from %alloc_a[0, 0, 0] size [1, 1, 32]
                           to %slots#0 size [32]
            : memref<1x1x32xf32, "L1">, !ktdf.fifo.slot<"L1LU" -> "SFU", 32xf32>
        ktdf.data_transfer from %alloc_b[0, 0, 0] size [1, 1, 1]
                           to %slots#1 size [32]
                           {transfer_mode = "splat"}
            : memref<1x1x1xf32, "L1">, !ktdf.fifo.slot<"L1LU" -> "SFU", 32xf32>
      } {applicable_units = ["L1LU"]}
      // Compute stage.
      ktdf.stage depends_in(%slots#3) depends_out(%slots#4) {
        %r0 = ktdf.read_from_fifo %slots#0 :
            <"L1LU" -> "SFU", 32xf32> -> tensor<1x1x32xf32>
        %r1 = ktdf.read_from_fifo %slots#1 :
            <"L1LU" -> "SFU", 32xf32> -> tensor<1x1x32xf32>
        %res = linalg.generic {
            indexing_maps = [#map_f32, #map1_f32, #map_f32],
            iterator_types = ["parallel", "parallel", "parallel"]}
            ins(%r0, %r1 : tensor<1x1x32xf32>, tensor<1x1x32xf32>)
            outs(%out : tensor<1x1x32xf32>) {
          ^bb0(%in: f32, %in_b: f32, %o: f32):
            %add = arith.addf %in, %in_b : f32
            linalg.yield %add : f32
        } -> tensor<1x1x32xf32>
        ktdf.write_to_fifo %res, %slots#2 :
            tensor<1x1x32xf32>, <"SFU" -> "L1SU", 32xf32>
      } {applicable_units = ["SFU"]}
    }
    return
  }
}

// ---------------------------------------------------------------------------
// f16: no widening — nothing to finish, so no read carries a mode.
// ---------------------------------------------------------------------------

// CHECK-LABEL: func.func @splat_f16
// Source size unchanged.
// CHECK:       ktdf.data_transfer from %{{.*}}[0, 0, 0] size [1, 1, 1] to
// CHECK-SAME:    transfer_mode = "splat"
// The load already splats a whole vector, so no sub-SIMD step is left over and
// there is no shuffle to name.
// CHECK-NOT:   splat = #ktdf.splat
// CHECK:       linalg.generic
// CHECK-SAME:    ins(%{{.*}}, %{{.*}} : tensor<1x1x64xf16>, tensor<1x1x64xf16>)

#map_f16  = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map1_f16 = affine_map<(d0, d1, d2) -> (0, d1, d2)>

module {
  func.func @splat_f16(%alloc_a: memref<1x1x64xf16, "L1">,
                       %alloc_b: memref<1x1x1xf16, "L1">) {
    %out = tensor.empty() : tensor<1x1x64xf16>
    ktdf.pipeline {
      %slots:4 = ktdf.private -> (
          !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>,
          !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>,
          !ktdf.token,
          !ktdf.token) {
        %f0:2 = ktdf.fifo.allocate() ->
            !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>,
            !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>
        %t0 = ktdf.create_token : !ktdf.token
        %t1 = ktdf.create_token : !ktdf.token
        ktdf.private_yield %f0#0, %f0#1, %t0, %t1 :
            !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>,
            !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>,
            !ktdf.token, !ktdf.token
      }
      ktdf.stage depends_in(none) depends_out(%slots#2) {
        ktdf.data_transfer from %alloc_a[0, 0, 0] size [1, 1, 64]
                           to %slots#0 size [64]
            : memref<1x1x64xf16, "L1">, !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>
        ktdf.data_transfer from %alloc_b[0, 0, 0] size [1, 1, 1]
                           to %slots#1 size [64]
                           {transfer_mode = "splat"}
            : memref<1x1x1xf16, "L1">, !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>
      } {applicable_units = ["L1LU"]}
      ktdf.stage depends_in(%slots#2) depends_out(%slots#3) {
        %r0 = ktdf.read_from_fifo %slots#0 :
            <"L1LU" -> "SFU", 64xf16> -> tensor<1x1x64xf16>
        %r1 = ktdf.read_from_fifo %slots#1 :
            <"L1LU" -> "SFU", 64xf16> -> tensor<1x1x64xf16>
        %res = linalg.generic {
            indexing_maps = [#map_f16, #map1_f16, #map_f16],
            iterator_types = ["parallel", "parallel", "parallel"]}
            ins(%r0, %r1 : tensor<1x1x64xf16>, tensor<1x1x64xf16>)
            outs(%out : tensor<1x1x64xf16>) {
          ^bb0(%in: f16, %in_b: f16, %o: f16):
            %add = arith.addf %in, %in_b : f16
            linalg.yield %add : f16
        } -> tensor<1x1x64xf16>
      } {applicable_units = ["SFU"]}
    }
    return
  }
}
