// RUN: scair-opt %s | filecheck %s

// Coverage for the Linalg ops that do not correspond to library calls, ported
// from MLIR's LinalgOps.td.
//
// `linalg.softmax` and the three Winograd transforms use their upstream
// assembly format. Two deviations from MLIR are visible below:
//
//   * `linalg.index` prints in the generic form. Its upstream format is
//     `$dim attr-dict `:` type($result)`, i.e. `linalg.index 0 : index`; ScaIR
//     prints that faithfully but cannot parse it back, because its attribute
//     parser reads `0 : index` greedily as a typed IntegerAttr and then finds
//     no `:` left for the result type.
//
//   * `fmr` prints as an i32 ordinal (`fmr(0 : i32)`) rather than as the
//     symbolic `fmr(F_2_3)`. ScaIR represents enums as their ordinal
//     IntegerAttr; see LinalgEnums.scala.

%t, %tb, %f = "test.op"() : () -> (tensor<4x8xf32>, tensor<4x8xf32>, f32)

// linalg.index, in the generic form (see the header for why).
%i = "linalg.index"() <{dim = 0 : i64}> : () -> index

// linalg.yield, terminating the body of a structured op. Both the operand-less
// and the operand-carrying forms are covered.
%mapped = "linalg.map"(%t, %tb) ({
^bb0(%a: f32):
  linalg.yield %a : f32
}) : (tensor<4x8xf32>, tensor<4x8xf32>) -> tensor<4x8xf32>

%filled = "linalg.map"(%t, %tb) ({
^bb0(%a: f32):
  linalg.yield
}) : (tensor<4x8xf32>, tensor<4x8xf32>) -> tensor<4x8xf32>

// linalg.softmax
%sm = linalg.softmax dimension(1 : i64) ins(%t : tensor<4x8xf32>) outs(%tb : tensor<4x8xf32>) -> tensor<4x8xf32>

// The result is optional upstream; on memrefs the op has no result at all.
%m, %mb = "test.op"() : () -> (memref<4x8xf32>, memref<4x8xf32>)
linalg.softmax dimension(0 : i64) ins(%m : memref<4x8xf32>) outs(%mb : memref<4x8xf32>)

// The Winograd transforms. `fmr` selects the F(m, r) minimal filtering
// algorithm; see the header for why it prints as an ordinal rather than as
// `F_2_3`.
%wfi, %wfo = "test.op"() : () -> (tensor<2x3x3x5xf32>, tensor<4x4x3x5xf32>)
%wf = linalg.winograd_filter_transform fmr(0 : i32) ins(%wfi : tensor<2x3x3x5xf32>) outs(%wfo : tensor<4x4x3x5xf32>) -> tensor<4x4x3x5xf32>

%wii, %wio = "test.op"() : () -> (tensor<2x6x6x5xf32>, tensor<4x4x2x2x2x5xf32>)
%wi = linalg.winograd_input_transform fmr(1 : i32) ins(%wii : tensor<2x6x6x5xf32>) outs(%wio : tensor<4x4x2x2x2x5xf32>) -> tensor<4x4x2x2x2x5xf32>

%woi, %woo = "test.op"() : () -> (tensor<4x4x2x2x2x5xf32>, tensor<2x4x4x5xf32>)
%wo = linalg.winograd_output_transform fmr(2 : i32) ins(%woi : tensor<4x4x2x2x2x5xf32>) outs(%woo : tensor<2x4x4x5xf32>) -> tensor<2x4x4x5xf32>

// CHECK:      builtin.module {
// CHECK-NEXT:   %0, %1, %2 = "test.op"() : () -> (tensor<4x8xf32>, tensor<4x8xf32>, f32)
// CHECK-NEXT:   %3 = "linalg.index"() <{dim = 0}> : () -> index
// CHECK-NEXT:   %4 = "linalg.map"(%0, %1) ({
// CHECK-NEXT:   ^bb0(%5: f32):
// CHECK-NEXT:     linalg.yield %5 : f32
// CHECK-NEXT:   }) : (tensor<4x8xf32>, tensor<4x8xf32>) -> tensor<4x8xf32>
// CHECK-NEXT:   %5 = "linalg.map"(%0, %1) ({
// CHECK-NEXT:   ^bb0(%6: f32):
// CHECK-NEXT:     linalg.yield
// CHECK-NEXT:   }) : (tensor<4x8xf32>, tensor<4x8xf32>) -> tensor<4x8xf32>
// CHECK-NEXT:   %6 = linalg.softmax dimension(1) ins(%0 : tensor<4x8xf32>) outs(%1 : tensor<4x8xf32>) -> tensor<4x8xf32>
// CHECK-NEXT:   %7, %8 = "test.op"() : () -> (memref<4x8xf32>, memref<4x8xf32>)
// CHECK-NEXT:   linalg.softmax dimension(0) ins(%7 : memref<4x8xf32>) outs(%8 : memref<4x8xf32>)
// CHECK-NEXT:   %9, %10 = "test.op"() : () -> (tensor<2x3x3x5xf32>, tensor<4x4x3x5xf32>)
// CHECK-NEXT:   %11 = linalg.winograd_filter_transform fmr(0 : i32) ins(%9 : tensor<2x3x3x5xf32>) outs(%10 : tensor<4x4x3x5xf32>) -> tensor<4x4x3x5xf32>
// CHECK-NEXT:   %12, %13 = "test.op"() : () -> (tensor<2x6x6x5xf32>, tensor<4x4x2x2x2x5xf32>)
// CHECK-NEXT:   %14 = linalg.winograd_input_transform fmr(1 : i32) ins(%12 : tensor<2x6x6x5xf32>) outs(%13 : tensor<4x4x2x2x2x5xf32>) -> tensor<4x4x2x2x2x5xf32>
// CHECK-NEXT:   %15, %16 = "test.op"() : () -> (tensor<4x4x2x2x2x5xf32>, tensor<2x4x4x5xf32>)
// CHECK-NEXT:   %17 = linalg.winograd_output_transform fmr(2 : i32) ins(%15 : tensor<4x4x2x2x2x5xf32>) outs(%16 : tensor<2x4x4x5xf32>) -> tensor<2x4x4x5xf32>
// CHECK-NEXT: }
