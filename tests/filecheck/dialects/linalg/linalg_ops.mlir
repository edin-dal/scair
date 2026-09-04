// RUN: scair-opt %s --print-generic | filecheck %s

// Coverage for the Linalg ops that do not correspond to library calls, ported
// from MLIR's LinalgOps.td.
//
// `linalg.softmax` and the three Winograd transforms use their upstream
// assembly format. One deviation from MLIR are visible below:
//
//   * `fmr` prints as an i32 ordinal (`fmr(0 : i32)`) rather than as the
//     symbolic `fmr(F_2_3)`. ScaIR represents enums as their ordinal
//     IntegerAttr; see LinalgEnums.scala.

%0, %1, %2 = "test.op"() : () -> (tensor<4x8xf32>, tensor<4x8xf32>, f32)

// linalg.index, in the generic form (see the header for why).
%3 = "linalg.index"() <{dim = 0}> : () -> index

// linalg.yield, terminating the body of a structured op. Both the operand-less
// and the operand-carrying forms are covered.
%4 = "linalg.map"(%0, %1) ({
^bb0(%5: f32):
  "linalg.yield"(%5) : (f32) -> ()
}) : (tensor<4x8xf32>, tensor<4x8xf32>) -> tensor<4x8xf32>

%5 = "linalg.map"(%0, %1) ({
^bb0(%6: f32):
  "linalg.yield"() : () -> ()
}) : (tensor<4x8xf32>, tensor<4x8xf32>) -> tensor<4x8xf32>

// linalg.softmax
%6 = "linalg.softmax"(%0, %1) <{dimension = 1}> : (tensor<4x8xf32>, tensor<4x8xf32>) -> tensor<4x8xf32>

// The result is optional upstream; on memrefs the op has no result at all.
%7, %8 = "test.op"() : () -> (memref<4x8xf32>, memref<4x8xf32>)

"linalg.softmax"(%7, %8) <{dimension = 0}> : (memref<4x8xf32>, memref<4x8xf32>) -> ()

// The Winograd transforms. `fmr` selects the F(m, r) minimal filtering
// algorithm; see the header for why it prints as an ordinal rather than as
// `F_2_3`.
%9, %10 = "test.op"() : () -> (tensor<2x3x3x5xf32>, tensor<4x4x3x5xf32>)
%11 = "linalg.winograd_filter_transform"(%9, %10) <{fmr = 0 : i32}> : (tensor<2x3x3x5xf32>, tensor<4x4x3x5xf32>) -> tensor<4x4x3x5xf32>

%12, %13 = "test.op"() : () -> (tensor<2x6x6x5xf32>, tensor<4x4x2x2x2x5xf32>)
%14 = "linalg.winograd_input_transform"(%12, %13) <{fmr = 1 : i32}> : (tensor<2x6x6x5xf32>, tensor<4x4x2x2x2x5xf32>) -> tensor<4x4x2x2x2x5xf32>

%15, %16 = "test.op"() : () -> (tensor<4x4x2x2x2x5xf32>, tensor<2x4x4x5xf32>)
%17 = "linalg.winograd_output_transform"(%15, %16) <{fmr = 2 : i32}> : (tensor<4x4x2x2x2x5xf32>, tensor<2x4x4x5xf32>) -> tensor<2x4x4x5xf32>

// CHECK:      "builtin.module"() ({
// CHECK-NEXT:   %0, %1, %2 = "test.op"() : () -> (tensor<4x8xf32>, tensor<4x8xf32>, f32)
// CHECK-NEXT:   %3 = "linalg.index"() <{dim = 0}> : () -> index
// CHECK-NEXT:   %4 = "linalg.map"(%0, %1) ({
// CHECK-NEXT:   ^bb0(%5: f32):
// CHECK-NEXT:     "linalg.yield"(%5) : (f32) -> ()
// CHECK-NEXT:   }) : (tensor<4x8xf32>, tensor<4x8xf32>) -> tensor<4x8xf32>
// CHECK-NEXT:   %5 = "linalg.map"(%0, %1) ({
// CHECK-NEXT:   ^bb0(%6: f32):
// CHECK-NEXT:     "linalg.yield"() : () -> ()
// CHECK-NEXT:   }) : (tensor<4x8xf32>, tensor<4x8xf32>) -> tensor<4x8xf32>
// CHECK-NEXT:   %6 = "linalg.softmax"(%0, %1) <{dimension = 1}> : (tensor<4x8xf32>, tensor<4x8xf32>) -> tensor<4x8xf32>
// CHECK-NEXT:   %7, %8 = "test.op"() : () -> (memref<4x8xf32>, memref<4x8xf32>)
// CHECK-NEXT:   "linalg.softmax"(%7, %8) <{dimension = 0}> : (memref<4x8xf32>, memref<4x8xf32>) -> ()
// CHECK-NEXT:   %9, %10 = "test.op"() : () -> (tensor<2x3x3x5xf32>, tensor<4x4x3x5xf32>)
// CHECK-NEXT:   %11 = "linalg.winograd_filter_transform"(%9, %10) <{fmr = 0 : i32}> : (tensor<2x3x3x5xf32>, tensor<4x4x3x5xf32>) -> tensor<4x4x3x5xf32>
// CHECK-NEXT:   %12, %13 = "test.op"() : () -> (tensor<2x6x6x5xf32>, tensor<4x4x2x2x2x5xf32>)
// CHECK-NEXT:   %14 = "linalg.winograd_input_transform"(%12, %13) <{fmr = 1 : i32}> : (tensor<2x6x6x5xf32>, tensor<4x4x2x2x2x5xf32>) -> tensor<4x4x2x2x2x5xf32>
// CHECK-NEXT:   %15, %16 = "test.op"() : () -> (tensor<4x4x2x2x2x5xf32>, tensor<2x4x4x5xf32>)
// CHECK-NEXT:   %17 = "linalg.winograd_output_transform"(%15, %16) <{fmr = 2 : i32}> : (tensor<4x4x2x2x2x5xf32>, tensor<2x4x4x5xf32>) -> tensor<2x4x4x5xf32>
// CHECK-NEXT: }) : () -> ()
