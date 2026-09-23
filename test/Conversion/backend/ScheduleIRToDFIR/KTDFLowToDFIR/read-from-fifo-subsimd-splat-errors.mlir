// RUN: dataflow-scheduler-opt -pass-pipeline="builtin.module(ktdflowering-to-dfir)" -verify-diagnostics %s

// A splat mode is only as good as the group width the arch declares for it.  A
// width that does not divide what arrives would make the shuffle drop lanes, so
// it is a diagnostic rather than a best effort.
//
// sample_device's SFU declares sub_simd_lanes f16 = 8, and this slot carries 4
// lanes.

module {
  ktdf_arch.device @sample_device attributes {} import("../../../../Dialect/KTDFArch/sample_device.mlir")

  // expected-error @below {{failed to run operation lowerings for group_wider_than_the_vector}}
  func.func @group_wider_than_the_vector() attributes {grid = [1]} {
    %l1lu = dataflow.get_unit {core = 0 : i32, name = "C0-L1LU", type = "L1LU"} : index
    %sfu  = dataflow.get_unit {core = 0 : i32, name = "C0-SFU",  type = "SFU"}  : index
    %tile_id = ktdp.get_compute_tile_id : index
    %c0 = arith.constant 0 : index

    %map_l1lu = uniform.def_immutable_mapping([%c0 -> %l1lu]) : index
    %u_l1lu   = uniform.query_map(map:%map_l1lu, key:%tile_id) : index
    %map_sfu  = uniform.def_immutable_mapping([%c0 -> %sfu])  : index
    %u_sfu    = uniform.query_map(map:%map_sfu,  key:%tile_id) : index

    // The slot itself is then left with nothing to lower it.
    // expected-error @below {{failed to legalize operation 'ktdf.fifo.allocate'}}
    %fifo = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"L1LU" -> "SFU", 4xf16>

    ktdf_lowering.execute_on %u_l1lu {
      %src = memref.alloc() : memref<1x4xf16, "L1">
      ktdf.data_transfer from %src[%c0, %c0] size [1, 4] to %fifo size [4]
        : memref<1x4xf16, "L1">, !ktdf.fifo.slot<"L1LU" -> "SFU", 4xf16>
    }

    ktdf_lowering.execute_on %u_sfu {
      %reg = memref.alloc() : memref<1x4xf16, "SFU_REG">
      // expected-error @below {{sub-SIMD group width 8 does not divide the 4 lanes received}}
      %read = ktdf.read_from_fifo %fifo
        {splat = #ktdf.splat<first_subsimd_lane_to_all_subsimd_lanes>}
        : !ktdf.fifo.slot<"L1LU" -> "SFU", 4xf16> -> memref<1x4xf16>
      memref.copy %read, %reg : memref<1x4xf16> to memref<1x4xf16, "SFU_REG">
    }

    return
  }
}
