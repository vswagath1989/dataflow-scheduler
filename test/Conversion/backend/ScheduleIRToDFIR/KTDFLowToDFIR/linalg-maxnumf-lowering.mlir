// RUN: dataflow-scheduler-opt -pass-pipeline="builtin.module(ktdflowering-to-dfir)" %s | FileCheck %s

// Tests that absf(x) maxnumf absf(y) lowers to the abs_max binary operator.
//
// That shape is the only one maxnumf lowers in: on its own it is the withdrawn
// 754-2008 operation, whose NaN and signed-zero handling is left to the
// implementation, so linalg-maxnumf-plain-rejected covers it being turned away.

// CHECK-LABEL: func.func @maxnumf_abs_max
// CHECK: vectorchain.binary
// CHECK-SAME: binary_op = #vectorchain<binary_operator abs_max>

// CHECK-LABEL: func.func @maxnumf_abs_max_with_constant
// CHECK: vectorchain.binary
// CHECK-SAME: binary_op = #vectorchain<binary_operator abs_max>

#map_in  = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map_out = affine_map<(d0, d1, d2) -> (d0, d2)>

module {
  ktdf_arch.device @sample_device attributes {} import("../../../../Dialect/KTDFArch/sample_device.mlir")

  // ── abs_max fusion: absf(x) maxnumf absf(y) → abs_max ─────────────────────
  func.func @maxnumf_abs_max() attributes {grid = [2]} {
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
      %zero = arith.constant 0.0 : f16
      linalg.fill ins(%zero : f16) outs(%alloc : memref<1x64xf16, "SFU_REG">)
      scf.for %i = %c0 to %c256 step %c1 {
        %input = ktdf.read_from_fifo %fifo : !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16> -> memref<1x1x64xf16>
        linalg.generic {
          indexing_maps = [#map_in, #map_out],
          iterator_types = ["parallel", "reduction", "parallel"]
        } ins(%input : memref<1x1x64xf16>) outs(%alloc : memref<1x64xf16, "SFU_REG">) {
        ^bb0(%in: f16, %out: f16):
          %abs_in  = math.absf %in  : f16
          %abs_out = math.absf %out : f16
          %result  = arith.maxnumf %abs_in, %abs_out : f16
          linalg.yield %result : f16
        }
      } {loop_type = #ktdf.loop_type<reduction_loop>}
    }
    return
  }

  // ── abs_max with constant: absf(x) maxnumf absf(y) with constant in body ──
  func.func @maxnumf_abs_max_with_constant() attributes {grid = [2]} {
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
      %zero = arith.constant 0.0 : f16
      linalg.fill ins(%zero : f16) outs(%alloc : memref<1x64xf16, "SFU_REG">)
      scf.for %i = %c0 to %c256 step %c1 {
        %input = ktdf.read_from_fifo %fifo : !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16> -> memref<1x1x64xf16>
        linalg.generic {
          indexing_maps = [#map_in, #map_out],
          iterator_types = ["parallel", "reduction", "parallel"]
        } ins(%input : memref<1x1x64xf16>) outs(%alloc : memref<1x64xf16, "SFU_REG">) {
        ^bb0(%in: f16, %out: f16):
          %cst = arith.constant 1.000000e+00 : f16
          %abs_in  = math.absf %in  : f16
          %abs_out = math.absf %out : f16
          %result  = arith.maxnumf %abs_in, %abs_out : f16
          linalg.yield %result : f16
        }
      } {loop_type = #ktdf.loop_type<reduction_loop>}
    }
    return
  }
}
