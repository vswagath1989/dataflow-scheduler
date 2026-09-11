// RUN: dataflow-scheduler-opt -pass-pipeline="builtin.module(ktdflowering-to-dfir)" -allow-unregistered-dialect %s | FileCheck %s

// Verify hardware-aware splat widening for f16.
//
// Send side — L1LU load granularity (access_granularity-aligned):
//   L1LU word_size=1 (byte), access granularities [64, 8, 2] words for "L1".
//   For f16 (2 bytes/elem): elem_words=2, min_words=2, fitAccess(2)→2 words
//   load_elements = 2 / 2 = 1   (smallest fitting granularity equals 1 elem)
//   agen.vector_load vector<1xf16>
//   vectorchain.shuffle indices=[0] rep=64 → vector<64xf16>
//   Only one shuffle is emitted on the L1LU side; no intermediate widening
//   is possible because the minimum granularity already covers just 1 element.
//   Because load_elements == 1 the FIFO carries a pure scalar broadcast;
//   the SFU receive needs no shuffle and read_with_splat is NOT set.
//
// Receive side: no shuffle — dataflow.receive result is used directly.

// CHECK-LABEL: func.func @splat_granularity_f16
// --- load-unit program_unit (L1LU) ---
// CHECK:       dataflow.program_unit
// CHECK:         %[[LOAD:.+]] = agen.vector_load
// CHECK-SAME:      vector<1xf16>
// CHECK-NEXT:    %[[SEND_SHUF:.+]] = vectorchain.shuffle input(%[[LOAD]]) {indices = [0 : i32], repetition = 64 : i32} : vector<1xf16>, vector<64xf16>
// CHECK:         dataflow.send %{{.*}}, %[[SEND_SHUF]] : vector<64xf16>
// --- compute-unit program_unit (SFU): receive result used directly ---
// CHECK:       dataflow.program_unit
// CHECK:         %[[RECV:.+]] = dataflow.receive
// CHECK-SAME:      vector<64xf16>
// CHECK-NOT:     vectorchain.shuffle
// CHECK:         "test.use"(%[[RECV]])

module {
  ktdf_arch.device @sample_device attributes {} import("../../../../Dialect/KTDFArch/sample_device.mlir")
  func.func @splat_granularity_f16() attributes {grid = [2]} {
    %0 = dataflow.get_unit {core = 0 : i32, name = "C0-L1LU", type = "L1LU"} : index
    %1 = dataflow.get_unit {core = 1 : i32, name = "C1-L1LU", type = "L1LU"} : index
    %2 = dataflow.get_unit {core = 0 : i32, name = "C0-SFU", type = "SFU"} : index
    %3 = dataflow.get_unit {core = 1 : i32, name = "C1-SFU", type = "SFU"} : index
    %tile_id = ktdp.get_compute_tile_id : index
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c12 = arith.constant 12 : index
    %map_l1lu = uniform.def_immutable_mapping([%c0 -> %0], [%c1 -> %1]):index
    %u_l1lu = uniform.query_map(map:%map_l1lu, key:%tile_id) : index
    %map_sfu  = uniform.def_immutable_mapping([%c0 -> %2], [%c1 -> %3]):index
    %u_sfu  = uniform.query_map(map:%map_sfu,  key:%tile_id) : index
    ktdf_lowering.execute_on %u_l1lu, %u_sfu {
      %alloc = memref.alloc() : memref<256xf16, "L1">
      %fifo:1 = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>
      ktdf_lowering.execute_on %u_l1lu {
        scf.for %i = %c0 to %c12 step %c1 {
          // Source size [1], dest size [64]: splat 1 f16 element to 64.
          // With word_size=1 and fitAccess(2)=2 words → loads 1 element,
          // single shuffle rep=64.  No widening is possible since the minimum
          // granularity covers exactly 1 f16 element.
          ktdf.data_transfer from %alloc[%c0] size [1]
                             to %fifo#0 size [64]
                             {transfer_mode = "splat"}
              : memref<256xf16, "L1">, !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>
        } {loop_type = #ktdf.loop_type<parallel_loop>}
      }
      ktdf_lowering.execute_on %u_sfu {
        scf.for %i = %c0 to %c12 step %c1 {
          %result = ktdf.read_from_fifo %fifo#0 : <"L1LU" -> "SFU", 64xf16> -> tensor<1x64xf16>
          "test.use"(%result) : (tensor<1x64xf16>) -> ()
        } {loop_type = #ktdf.loop_type<parallel_loop>}
      }
    }
    return
  }
}
