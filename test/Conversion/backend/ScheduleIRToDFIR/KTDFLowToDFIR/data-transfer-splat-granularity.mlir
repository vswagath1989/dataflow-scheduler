// RUN: dataflow-scheduler-opt -pass-pipeline="builtin.module(ktdflowering-to-dfir)" -allow-unregistered-dialect %s | FileCheck %s

// Verify hardware-aware splat widening for f32.
//
// Send side — L1LU load granularity (access_granularity-aligned):
//   L1LU word_size=1 (byte), access granularities [64, 8, 2] words for "L1".
//   For f32 (4 bytes/elem): elem_words=4, min_words=4, fitAccess(4)→8 words
//   load_elements = 8 / 4 = 2   repetition = 64 / 2 = 32
//   agen.vector_load vector<2xf32>
//   vectorchain.shuffle indices=[0,1] rep=32 → vector<64xf32>
//
//   When the source memref is multi-dimensional (e.g. memref<?x1x1x32xf32>
//   with src_static_sizes=[1,1,1,1]), widening replaces only the innermost
//   dimension: effective_sizes=[1,1,1,4], producing a 4D load_set that keeps
//   all outer dims equality-constrained (d0==0, d1==0, d2==0) and gives the
//   inner dim a range (d3>=0, -d3+3>=0).  The 1D source in this test
//   (src_static_sizes=[1]) yields effective_sizes=[2] with the same logic.
//
// Receive side — SFU sub_simd_lanes (arch-declared shuffle granularity):
//   SFU sub_simd_lanes for f32 = 8  →  repetition = 64 / 8 = 8
//   dataflow.receive vector<64xf32>
//   vectorchain.shuffle indices=[0,0,0,0,0,0,0,0] rep=8 → vector<64xf32>

// CHECK-DAG: #[[MAP:.+]] = affine_map<(d0) -> (d0)>
// CHECK-DAG: #[[SET:.+]] = affine_set<(d0) : (d0 >= 0, -d0 + 1 >= 0)>
// CHECK-DAG: #[[REG_SET:.+]] = affine_set<(d0) : (d0 >= 0, -d0 + 63 >= 0)>

// CHECK-LABEL: func.func @splat_granularity_f32
// CHECK:         %[[C0:.+]] = arith.constant 0 : index
// CHECK:         %[[U0:.+]] = dataflow.get_unit {core = 0 : i32, name = "C0-L1LU", type = "L1LU"} : index
// CHECK:         %[[U1:.+]] = dataflow.get_unit {core = 1 : i32, name = "C1-L1LU", type = "L1LU"} : index
// CHECK:         %[[V0:.+]] = dataflow.get_unit {core = 0 : i32, name = "C0-SFU", type = "SFU"} : index
// CHECK:         %[[V1:.+]] = dataflow.get_unit {core = 1 : i32, name = "C1-SFU", type = "SFU"} : index
// CHECK:         %[[REG_UNIT:.+]] = dataflow.get_unit {name = "sfu_reg", type = "sfu_reg"} : index
// --- load-unit program_unit (L1LU) ---
// CHECK:       dataflow.program_unit iter_arg : %[[ITER_L:.+]] -> (%[[U0]], %[[U1]]) :
// CHECK:         %[[ALLOC:.+]] = memref.alloc() : memref<256xf32, "L1">
// CHECK:         scf.for
// CHECK:           %[[LOAD:.+]] = agen.vector_load %[[ALLOC]][%{{.*}}]
// CHECK-SAME:        {load_order = #[[MAP]], load_set = #[[SET]]} : memref<256xf32, "L1">, vector<2xf32>
// CHECK-NEXT:      %[[SEND_SHUF:.+]] = vectorchain.shuffle input(%[[LOAD]]) {indices = [0 : i32, 1 : i32], repetition = 32 : i32} : vector<2xf32>, vector<64xf32>
// CHECK-NEXT:      %{{.*}} = uniform.def_immutable_mapping
// CHECK-NEXT:      %{{.*}} = uniform.query_map
// CHECK-NEXT:      dataflow.send %{{.*}}, %[[SEND_SHUF]] : vector<64xf32>
// --- compute-unit program_unit (SFU) ---
// CHECK:       dataflow.program_unit iter_arg : %[[ITER_C:.+]] -> (%[[V0]], %[[V1]]) :
// CHECK:         scf.for
// CHECK:           %{{.*}} = uniform.def_immutable_mapping
// CHECK-NEXT:      %[[SRC:.+]] = uniform.query_map
// CHECK-NEXT:      %[[RECV:.+]] = dataflow.receive %[[SRC]] : vector<64xf32>
// CHECK-NEXT:      %[[RECV_SHUF:.+]] = vectorchain.shuffle input(%[[RECV]]) {indices = [0 : i32, 0 : i32, 0 : i32, 0 : i32, 0 : i32, 0 : i32, 0 : i32, 0 : i32], repetition = 8 : i32} : vector<64xf32>, vector<64xf32>
// CHECK-NEXT:      %[[REG_VIEW:.+]] = dataflow.get_logical_memory_view %[[REG_UNIT]], %[[C0]] {layout_map = #[[MAP]]} : index, index, memref<64xf32>
// CHECK-NEXT:      agen.vector_store %[[RECV_SHUF]], %[[REG_VIEW]][%{{.*}}] {store_order = #[[MAP]], store_set = #[[REG_SET]]} : memref<64xf32>, vector<64xf32>
// CHECK-NEXT:      %[[RESULT:.+]] = agen.vector_load %[[REG_VIEW]][%{{.*}}] {load_order = #[[MAP]], load_set = #[[REG_SET]]} : memref<64xf32>, vector<64xf32>
// CHECK-NEXT:      "test.use"(%[[RESULT]])

module {
  ktdf_arch.device @sample_device attributes {} import("../../../../Dialect/KTDFArch/sample_device.mlir")
  func.func @splat_granularity_f32() attributes {grid = [2]} {
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
      %alloc = memref.alloc() : memref<256xf32, "L1">
      %fifo:1 = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf32>
      ktdf_lowering.execute_on %u_l1lu {
        scf.for %i = %c0 to %c12 step %c1 {
          // Source size [1], dest size [64]: splat 1 f32 element to 64.
          // With word_size=1 and fitAccess(4)=8 words → loads 2 elements,
          // shuffle rep=32 instead of loading 1 element and rep=64.
          ktdf.data_transfer from %alloc[%c0] size [1]
                             to %fifo#0 size [64]
                             {transfer_mode = "splat"}
              : memref<256xf32, "L1">, !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf32>
        } {loop_type = #ktdf.loop_type<parallel_loop>}
      }
      ktdf_lowering.execute_on %u_sfu {
        scf.for %i = %c0 to %c12 step %c1 {
          %result = ktdf.read_from_fifo %fifo#0 : <"L1LU" -> "SFU", 64xf32> -> tensor<1x64xf32>
          "test.use"(%result) : (tensor<1x64xf32>) -> ()
        } {loop_type = #ktdf.loop_type<parallel_loop>}
      }
    }
    return
  }
}
