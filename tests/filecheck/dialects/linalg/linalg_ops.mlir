// RUN: scair-opt %s --print-generic | filecheck %s

// Coverage for the Linalg ops that do not correspond to library calls, ported
// from MLIR's LinalgOps.td.
//
// `linalg.softmax` and the three Winograd transforms use their upstream
// assembly format. One deviation from MLIR is visible below:
//
//   * `fmr` prints as an i32 ordinal (`fmr(0 : i32)`) rather than as the
//     symbolic `fmr(F_2_3)`. ScaIR represents enums as their ordinal
//     IntegerAttr; see LinalgEnums.scala. MLIR accepts the ordinal on input.

%0, %1, %2 = "test.op"() : () -> (tensor<4x8xf32>, tensor<4x8xf32>, f32)

// linalg.index. MLIR requires an enclosing op carrying the LinalgOp interface,
// so it is exercised inside the body of a linalg.generic.
%3 = "linalg.generic"(%0, %1) <{
  indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>],
  iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>],
  operandSegmentSizes = array<i32: 1, 1>
}> ({
^bb0(%4: f32, %5: f32):
  %6 = "linalg.index"() <{dim = 0}> : () -> index
  "linalg.yield"(%4) : (f32) -> ()
}) : (tensor<4x8xf32>, tensor<4x8xf32>) -> tensor<4x8xf32>

// linalg.yield, terminating the body of a structured op. Here in its
// operand-carrying form: the number of yielded values must match the number of
// inits, and linalg.map's body block takes one argument per input plus one for
// the init.
%7 = "linalg.map"(%0, %1) ({
^bb0(%8: f32, %9: f32):
  "linalg.yield"(%8) : (f32) -> ()
}) : (tensor<4x8xf32>, tensor<4x8xf32>) -> tensor<4x8xf32>

// ... and in its operand-less form, which needs an enclosing op with no inits
// at all -- hence a linalg.generic with an empty `outs` group and no results.
"linalg.generic"(%0) <{
  indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>],
  iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>],
  operandSegmentSizes = array<i32: 1, 0>
}> ({
^bb0(%10: f32):
  "linalg.yield"() : () -> ()
}) : (tensor<4x8xf32>) -> ()

// linalg.softmax
%11 = "linalg.softmax"(%0, %1) <{dimension = 1}> : (tensor<4x8xf32>, tensor<4x8xf32>) -> tensor<4x8xf32>

// The result is optional upstream; on memrefs the op has no result at all.
%12, %13 = "test.op"() : () -> (memref<4x8xf32>, memref<4x8xf32>)

"linalg.softmax"(%12, %13) <{dimension = 0}> : (memref<4x8xf32>, memref<4x8xf32>) -> ()

// The Winograd transforms. `fmr` selects the F(m, r) minimal filtering
// algorithm; see the header for why it prints as an ordinal rather than as
// `F_2_3`. Shapes follow from the tile size alpha = m + r - 1.

// fmr = 0 is F_2_3, so alpha = 4. The filter is (F, H, W, C) and the transform
// of it is (alpha, alpha, C, F).
%14, %15 = "test.op"() : () -> (tensor<2x3x3x5xf32>, tensor<4x4x5x2xf32>)
%16 = "linalg.winograd_filter_transform"(%14, %15) <{fmr = 0 : i32}> : (tensor<2x3x3x5xf32>, tensor<4x4x5x2xf32>) -> tensor<4x4x5x2xf32>

// fmr = 1 is F_4_3, so alpha = 6. The input is (N, H, W, C) and the transform
// of it is (alpha, alpha, tileH, tileW, N, C), with tileH = (H - (r - 1)) / m.
%17, %18 = "test.op"() : () -> (tensor<2x6x6x5xf32>, tensor<6x6x1x1x2x5xf32>)
%19 = "linalg.winograd_input_transform"(%17, %18) <{fmr = 1 : i32}> : (tensor<2x6x6x5xf32>, tensor<6x6x1x1x2x5xf32>) -> tensor<6x6x1x1x2x5xf32>

// fmr = 2 is F_2_5, so alpha = 6 again. The value is
// (alpha, alpha, tileH, tileW, N, F) and the output is (N, m * tileH, m * tileW, F).
%20, %21 = "test.op"() : () -> (tensor<6x6x2x2x2x5xf32>, tensor<2x4x4x5xf32>)
%22 = "linalg.winograd_output_transform"(%20, %21) <{fmr = 2 : i32}> : (tensor<6x6x2x2x2x5xf32>, tensor<2x4x4x5xf32>) -> tensor<2x4x4x5xf32>

// CHECK:      #map = affine_map<(d0, d1)[] -> (d0, d1)>
// CHECK-NEXT: "builtin.module"() ({
// CHECK-NEXT:   %0, %1, %2 = "test.op"() : () -> (tensor<4x8xf32>, tensor<4x8xf32>, f32)
// CHECK-NEXT:   %3 = "linalg.generic"(%0, %1) <{indexing_maps = [#map, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operandSegmentSizes = array<i32: 1, 1>}> ({
// CHECK-NEXT:   ^bb0(%4: f32, %5: f32):
// CHECK-NEXT:     %6 = "linalg.index"() <{dim = 0}> : () -> index
// CHECK-NEXT:     "linalg.yield"(%4) : (f32) -> ()
// CHECK-NEXT:   }) : (tensor<4x8xf32>, tensor<4x8xf32>) -> tensor<4x8xf32>
// CHECK-NEXT:   %4 = "linalg.map"(%0, %1) ({
// CHECK-NEXT:   ^bb0(%5: f32, %6: f32):
// CHECK-NEXT:     "linalg.yield"(%5) : (f32) -> ()
// CHECK-NEXT:   }) : (tensor<4x8xf32>, tensor<4x8xf32>) -> tensor<4x8xf32>
// CHECK-NEXT:   "linalg.generic"(%0) <{indexing_maps = [#map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operandSegmentSizes = array<i32: 1, 0>}> ({
// CHECK-NEXT:   ^bb0(%5: f32):
// CHECK-NEXT:     "linalg.yield"() : () -> ()
// CHECK-NEXT:   }) : (tensor<4x8xf32>) -> ()
// CHECK-NEXT:   %5 = "linalg.softmax"(%0, %1) <{dimension = 1}> : (tensor<4x8xf32>, tensor<4x8xf32>) -> tensor<4x8xf32>
// CHECK-NEXT:   %6, %7 = "test.op"() : () -> (memref<4x8xf32>, memref<4x8xf32>)
// CHECK-NEXT:   "linalg.softmax"(%6, %7) <{dimension = 0}> : (memref<4x8xf32>, memref<4x8xf32>) -> ()
// CHECK-NEXT:   %8, %9 = "test.op"() : () -> (tensor<2x3x3x5xf32>, tensor<4x4x5x2xf32>)
// CHECK-NEXT:   %10 = "linalg.winograd_filter_transform"(%8, %9) <{fmr = 0 : i32}> : (tensor<2x3x3x5xf32>, tensor<4x4x5x2xf32>) -> tensor<4x4x5x2xf32>
// CHECK-NEXT:   %11, %12 = "test.op"() : () -> (tensor<2x6x6x5xf32>, tensor<6x6x1x1x2x5xf32>)
// CHECK-NEXT:   %13 = "linalg.winograd_input_transform"(%11, %12) <{fmr = 1 : i32}> : (tensor<2x6x6x5xf32>, tensor<6x6x1x1x2x5xf32>) -> tensor<6x6x1x1x2x5xf32>
// CHECK-NEXT:   %14, %15 = "test.op"() : () -> (tensor<6x6x2x2x2x5xf32>, tensor<2x4x4x5xf32>)
// CHECK-NEXT:   %16 = "linalg.winograd_output_transform"(%14, %15) <{fmr = 2 : i32}> : (tensor<6x6x2x2x2x5xf32>, tensor<2x4x4x5xf32>) -> tensor<2x4x4x5xf32>
// CHECK-NEXT: }) : () -> ()
