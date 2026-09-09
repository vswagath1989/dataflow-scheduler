// RUN: dataflow-scheduler-opt -pass-pipeline="builtin.module(ktdflowering-to-dfir)" %s | FileCheck %s

// Tests that arith.minnumf lowers to the min binary operator. It was accepted by
// the frontend legality check and had no lowering, so a body using it got as far
// as here and then said only "unsupported operation type in linalg.generic body".
//
// There is no absolute-min operator, so minnumf has no second shape to select the
// way maxnumf picks abs_max for the abs-of-both case.

// CHECK-LABEL: func.func @minnumf_plain
// CHECK: vectorchain.binary
// CHECK-SAME: binary_op = #vectorchain<binary_operator min>

#map_in  = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map_out = affine_map<(d0, d1, d2) -> (d0, d2)>

module {
  ktdf_arch.device @sample_device attributes {} import("../../../../Dialect/KTDFArch/sample_device.mlir")

  // ── minnumf: lowers to min ─────────────────────────────────────────────────
  func.func @minnumf_plain() attributes {grid = [2]} {
    %l1lu0 = dataflow.get_unit {core = 0 : i32, name = "C0-L1LU", type = "L1LU"} : index
    %l1lu1 = dataflow.get_unit {core = 1 : i32, name = "C1-L1LU", type = "L1LU"} : index
    %sfu0  = dataflow.get_unit {core = 0 : i32, name = "C0-SFU",  type = "SFU"}  : index
    %sfu1  = dataflow.get_unit {core = 1 : i32, name = "C1-SFU",  type = "SFU"}  : index
    %tile_id = ktdp.get_compute_tile_id : index
    %c0   = arith.constant 0 : index
    %c1   = arith.constant 1 : index
    %c256 = arith.constant 256 : index
    %map_l1lu = uniform.def_immutable_mapping([%c0 -> %l1lu0], [%c1 -> %l1lu1]) : index
    %u_l1lu   = uniform.query_map(map:%map_l1lu, key:%tile_id) : index
    %map_sfu  = uniform.def_immutable_mapping([%c0 -> %sfu0],  [%c1 -> %sfu1])  : index
    %u_sfu    = uniform.query_map(map:%map_sfu,  key:%tile_id) : index

    %alloc_l1 = memref.alloc() : memref<1x256x64xf16, "L1">
    %fifo = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>

    ktdf_lowering.execute_on %u_l1lu {
      scf.for %i = %c0 to %c256 step %c1 {
        ktdf.data_transfer from %alloc_l1[%c0, %i, %c0] size [1, 1, 64] to %fifo size [64] : memref<1x256x64xf16, "L1">, !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>
      }
    }
    ktdf_lowering.execute_on %u_sfu {
      %alloc = memref.alloc() : memref<1x64xf16, "SFU_REG">
      %neg_inf = arith.constant 0xFF80 : f16
      linalg.fill ins(%neg_inf : f16) outs(%alloc : memref<1x64xf16, "SFU_REG">)
      scf.for %i = %c0 to %c256 step %c1 {
        %input = ktdf.read_from_fifo %fifo : !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16> -> memref<1x1x64xf16>
        linalg.generic {
          indexing_maps = [#map_in, #map_out],
          iterator_types = ["parallel", "reduction", "parallel"]
        } ins(%input : memref<1x1x64xf16>) outs(%alloc : memref<1x64xf16, "SFU_REG">) {
        ^bb0(%in: f16, %out: f16):
          %result = arith.minnumf %in, %out : f16
          linalg.yield %result : f16
        }
      } {loop_type = #ktdf.loop_type<reduction_loop>}
    }
    return
  }

}
