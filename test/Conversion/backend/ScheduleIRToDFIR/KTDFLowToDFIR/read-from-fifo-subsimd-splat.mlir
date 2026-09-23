// RUN: dataflow-scheduler-opt -pass-pipeline="builtin.module(ktdflowering-to-dfir)" %s | FileCheck %s

// Verify that a ktdf.read_from_fifo carrying a splat mode is lowered to the
// receive plus the shuffle the mode names.
//
// The mode says the load unit widened the load to its access granularity, so
// only the first lane of every sub-SIMD group holds a live element. The shuffle
// spreads that lane across its group, which is what a compute unit declaring
// shuffle_modes = { FirstSubSimdLaneToEachSubSimd } performs.
//
// The group width comes from the compute unit's arch declaration, not from the
// op: sample_device's compute unit declares lanes f16 = 64 and
// sub_simd_lanes f16 = 8, so the shuffle repeats eight zero indices eight times.
//
// The memref.copy that follows stores the shuffled vector into the register
// buffer, the same way it stores a plain read.

// CHECK-LABEL: func.func @read_from_fifo_subsimd_splat
// CHECK:         dataflow.program_unit
// CHECK:           %[[RECV:.+]] = dataflow.receive
// CHECK-NEXT:      %[[SPLAT:.+]] = vectorchain.shuffle input(%[[RECV]]
// CHECK-SAME:        indices = [0 : i32, 0 : i32, 0 : i32, 0 : i32, 0 : i32, 0 : i32, 0 : i32, 0 : i32]
// CHECK-SAME:        repetition = 8
// CHECK:           agen.vector_store %[[SPLAT]]
// CHECK-NOT:       memref.copy

module {
  ktdf_arch.device @sample_device attributes {} import("../../../../Dialect/KTDFArch/sample_device.mlir")

  func.func @read_from_fifo_subsimd_splat() attributes {grid = [1]} {
    %l1lu = dataflow.get_unit {core = 0 : i32, name = "C0-L1LU", type = "L1LU"} : index
    %sfu  = dataflow.get_unit {core = 0 : i32, name = "C0-SFU",  type = "SFU"}  : index
    %tile_id = ktdp.get_compute_tile_id : index
    %c0 = arith.constant 0 : index

    %map_l1lu = uniform.def_immutable_mapping([%c0 -> %l1lu]) : index
    %u_l1lu   = uniform.query_map(map:%map_l1lu, key:%tile_id) : index
    %map_sfu  = uniform.def_immutable_mapping([%c0 -> %sfu])  : index
    %u_sfu    = uniform.query_map(map:%map_sfu,  key:%tile_id) : index

    %fifo = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>

    // Producer: a splat load widened to the access granularity. The source size
    // is the load width; the destination is the full vector.
    ktdf_lowering.execute_on %u_l1lu {
      %src = memref.alloc() : memref<1x2xf16, "L1">
      ktdf.data_transfer from %src[%c0, %c0] size [1, 2] to %fifo size [64]
        {transfer_mode = "splat"}
        : memref<1x2xf16, "L1">, !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>
    }

    // Consumer: the read owes the sub-SIMD shuffle, and the copy moves the
    // result into a register buffer.
    ktdf_lowering.execute_on %u_sfu {
      %reg = memref.alloc() : memref<1x64xf16, "SFU_REG">
      %read = ktdf.read_from_fifo %fifo
        {splat = #ktdf.splat<first_subsimd_lane_to_all_subsimd_lanes>}
        : !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16> -> memref<1x64xf16>
      memref.copy %read, %reg : memref<1x64xf16> to memref<1x64xf16, "SFU_REG">
    }

    return
  }
}
