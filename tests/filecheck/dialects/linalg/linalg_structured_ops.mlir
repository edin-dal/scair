// RUN: scair-opt %s | filecheck %s

// Coverage for the hand-written structured Linalg ops, ported from MLIR's
// LinalgStructuredOps.td. Each op is exercised in the generic form, which is
// what ScaIR prints for them and what MLIR accepts on input.
//
// Two deviations from MLIR to note:
//
//   * `linalg.reduce` carries an `operandSegmentSizes` property here. Upstream
//     it uses SameVariadicOperandSize and so has none; ScaIR requires the
//     property whenever an op has two variadic operand groups.

%t, %tb, %f = "test.op"() : () -> (tensor<4x8xf32>, tensor<4x8xf32>, f32)

// linalg.generic, with both optional string attributes present.
%gen = "linalg.generic"(%t, %tb) <{
  indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>],
  iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>],
  doc = "an element-wise copy",
  library_call = "copy",
  operandSegmentSizes = array<i32: 1, 1>
}> ({
^bb0(%a: f32, %b: f32):
  "linalg.yield"(%a) : (f32) -> ()
}) : (tensor<4x8xf32>, tensor<4x8xf32>) -> tensor<4x8xf32>

// ... and without them.
%gen2 = "linalg.generic"(%t, %tb) <{
  indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>],
  iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<reduction>],
  operandSegmentSizes = array<i32: 1, 1>
}> ({
^bb0(%a: f32, %b: f32):
  "linalg.yield"(%a) : (f32) -> ()
}) : (tensor<4x8xf32>, tensor<4x8xf32>) -> tensor<4x8xf32>

// linalg.generic on memrefs, i.e. with no results at all.
%m, %mb = "test.op"() : () -> (memref<4x8xf32>, memref<4x8xf32>)
"linalg.generic"(%m, %mb) <{
  indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>],
  iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>],
  operandSegmentSizes = array<i32: 1, 1>
}> ({
^bb0(%a: f32, %b: f32):
  "linalg.yield"() : () -> ()
}) : (memref<4x8xf32>, memref<4x8xf32>) -> ()

// linalg.map
%mapped = "linalg.map"(%t, %t, %tb) ({
^bb0(%a: f32, %b: f32):
  "linalg.yield"(%a) : (f32) -> ()
}) : (tensor<4x8xf32>, tensor<4x8xf32>, tensor<4x8xf32>) -> tensor<4x8xf32>

// linalg.reduce. Note the operandSegmentSizes: upstream the op carries
// SameVariadicOperandSize instead, and so has no such property.
%r1d = "test.op"() : () -> tensor<4xf32>
%reduced = "linalg.reduce"(%t, %r1d) <{dimensions = array<i64: 1>, operandSegmentSizes = array<i32: 1, 1>}> ({
^bb0(%a: f32, %b: f32):
  "linalg.yield"(%a) : (f32) -> ()
}) : (tensor<4x8xf32>, tensor<4xf32>) -> tensor<4xf32>

// linalg.transpose
%tt = "test.op"() : () -> tensor<8x4xf32>
%transposed = "linalg.transpose"(%t, %tt) <{permutation = array<i64: 1, 0>}> ({
^bb0(%a: f32, %b: f32):
  "linalg.yield"(%a) : (f32) -> ()
}) : (tensor<4x8xf32>, tensor<8x4xf32>) -> tensor<8x4xf32>

// linalg.broadcast
%bt = "test.op"() : () -> tensor<4x8x2xf32>
%broadcast = "linalg.broadcast"(%t, %bt) <{dimensions = array<i64: 2>}> ({
^bb0(%a: f32, %b: f32):
  "linalg.yield"(%a) : (f32) -> ()
}) : (tensor<4x8xf32>, tensor<4x8x2xf32>) -> tensor<4x8x2xf32>

// linalg.elementwise. `kind` is an ElementwiseKind ordinal: 13 is `add`.
%ew = "linalg.elementwise"(%t, %t, %tb) <{kind = #linalg.elementwise_kind<add>, operandSegmentSizes = array<i32: 2, 1>}> ({
^bb0(%a: f32, %b: f32, %c: f32):
  "linalg.yield"(%a) : (f32) -> ()
}) : (tensor<4x8xf32>, tensor<4x8xf32>, tensor<4x8xf32>) -> tensor<4x8xf32>

// ... and with explicit indexing maps.
%ew2 = "linalg.elementwise"(%t, %tb) <{
  kind = #linalg.elementwise_kind<exp>,
  indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>],
  operandSegmentSizes = array<i32: 1, 1>
}> ({
^bb0(%a: f32, %b: f32):
  "linalg.yield"(%a) : (f32) -> ()
}) : (tensor<4x8xf32>, tensor<4x8xf32>) -> tensor<4x8xf32>

// linalg.matmul, with the default indexing maps and cast left implicit.
%lhs, %rhs, %acc = "test.op"() : () -> (tensor<4x16xf32>, tensor<16x8xf32>, tensor<4x8xf32>)
%mm = "linalg.matmul"(%lhs, %rhs, %acc) <{operandSegmentSizes = array<i32: 2, 1>}> ({
^bb0(%a: f32, %b: f32, %c: f32):
  "linalg.yield"(%a) : (f32) -> ()
}) : (tensor<4x16xf32>, tensor<16x8xf32>, tensor<4x8xf32>) -> tensor<4x8xf32>

// ... and with both spelled out. `cast = 1` is TypeFn::cast_unsigned.
%mm2 = "linalg.matmul"(%lhs, %rhs, %acc) <{
  indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d2, d1)>, affine_map<(d0, d1, d2) -> (d0, d1)>],
  cast = #linalg.type_fn<cast_unsigned>,
  operandSegmentSizes = array<i32: 2, 1>
}> ({
^bb0(%a: f32, %b: f32, %c: f32):
  "linalg.yield"(%a) : (f32) -> ()
}) : (tensor<4x16xf32>, tensor<16x8xf32>, tensor<4x8xf32>) -> tensor<4x8xf32>

// linalg.contract, whose indexing maps are mandatory.
%contracted = "linalg.contract"(%lhs, %rhs, %acc) <{
  indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d2, d1)>, affine_map<(d0, d1, d2) -> (d0, d1)>],
  operandSegmentSizes = array<i32: 2, 1>
}> ({
^bb0(%a: f32, %b: f32, %c: f32):
  "linalg.yield"(%a) : (f32) -> ()
}) : (tensor<4x16xf32>, tensor<16x8xf32>, tensor<4x8xf32>) -> tensor<4x8xf32>

// linalg.batch_matmul
%blhs, %brhs, %bacc = "test.op"() : () -> (tensor<2x4x16xf32>, tensor<2x16x8xf32>, tensor<2x4x8xf32>)
%bmm = "linalg.batch_matmul"(%blhs, %brhs, %bacc) <{operandSegmentSizes = array<i32: 2, 1>}> ({
^bb0(%a: f32, %b: f32, %c: f32):
  "linalg.yield"(%a) : (f32) -> ()
}) : (tensor<2x4x16xf32>, tensor<2x16x8xf32>, tensor<2x4x8xf32>) -> tensor<2x4x8xf32>

// linalg.batch_reduce_matmul, which reduces the batch dimension away.
%brmm = "linalg.batch_reduce_matmul"(%blhs, %brhs, %acc) <{operandSegmentSizes = array<i32: 2, 1>}> ({
^bb0(%a: f32, %b: f32, %c: f32):
  "linalg.yield"(%a) : (f32) -> ()
}) : (tensor<2x4x16xf32>, tensor<2x16x8xf32>, tensor<4x8xf32>) -> tensor<4x8xf32>

// CHECK:      #map = affine_map<(d0, d1)[] -> (d0, d1)>
// CHECK-NEXT: #map1 = affine_map<(d0, d1, d2)[] -> (d0, d2)>
// CHECK-NEXT: #map2 = affine_map<(d0, d1, d2)[] -> (d2, d1)>
// CHECK-NEXT: #map3 = affine_map<(d0, d1, d2)[] -> (d0, d1)>
// CHECK-NEXT: builtin.module {
// CHECK-NEXT:   %0, %1, %2 = "test.op"() : () -> (tensor<4x8xf32>, tensor<4x8xf32>, f32)
// CHECK-NEXT:   %3 = "linalg.generic"(%0, %1) <{iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], indexing_maps = [#map, #map], doc = "an element-wise copy", operandSegmentSizes = array<i32: 1, 1>, library_call = "copy"}> ({
// CHECK-NEXT:   ^bb0(%4: f32, %5: f32):
// CHECK-NEXT:     "linalg.yield"(%4) : (f32) -> ()
// CHECK-NEXT:   }) : (tensor<4x8xf32>, tensor<4x8xf32>) -> tensor<4x8xf32>
// CHECK-NEXT:   %4 = "linalg.generic"(%0, %1) <{indexing_maps = [#map, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<reduction>], operandSegmentSizes = array<i32: 1, 1>}> ({
// CHECK-NEXT:   ^bb0(%5: f32, %6: f32):
// CHECK-NEXT:     "linalg.yield"(%5) : (f32) -> ()
// CHECK-NEXT:   }) : (tensor<4x8xf32>, tensor<4x8xf32>) -> tensor<4x8xf32>
// CHECK-NEXT:   %5, %6 = "test.op"() : () -> (memref<4x8xf32>, memref<4x8xf32>)
// CHECK-NEXT:   "linalg.generic"(%5, %6) <{indexing_maps = [#map, #map], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operandSegmentSizes = array<i32: 1, 1>}> ({
// CHECK-NEXT:   ^bb0(%7: f32, %8: f32):
// CHECK-NEXT:     "linalg.yield"() : () -> ()
// CHECK-NEXT:   }) : (memref<4x8xf32>, memref<4x8xf32>) -> ()
// CHECK-NEXT:   %7 = "linalg.map"(%0, %0, %1) ({
// CHECK-NEXT:   ^bb0(%8: f32, %9: f32):
// CHECK-NEXT:     "linalg.yield"(%8) : (f32) -> ()
// CHECK-NEXT:   }) : (tensor<4x8xf32>, tensor<4x8xf32>, tensor<4x8xf32>) -> tensor<4x8xf32>
// CHECK-NEXT:   %8 = "test.op"() : () -> tensor<4xf32>
// CHECK-NEXT:   %9 = "linalg.reduce"(%0, %8) <{dimensions = array<i64: 1>, operandSegmentSizes = array<i32: 1, 1>}> ({
// CHECK-NEXT:   ^bb0(%10: f32, %11: f32):
// CHECK-NEXT:     "linalg.yield"(%10) : (f32) -> ()
// CHECK-NEXT:   }) : (tensor<4x8xf32>, tensor<4xf32>) -> tensor<4xf32>
// CHECK-NEXT:   %10 = "test.op"() : () -> tensor<8x4xf32>
// CHECK-NEXT:   %11 = "linalg.transpose"(%0, %10) <{permutation = array<i64: 1, 0>}> ({
// CHECK-NEXT:   ^bb0(%12: f32, %13: f32):
// CHECK-NEXT:     "linalg.yield"(%12) : (f32) -> ()
// CHECK-NEXT:   }) : (tensor<4x8xf32>, tensor<8x4xf32>) -> tensor<8x4xf32>
// CHECK-NEXT:   %12 = "test.op"() : () -> tensor<4x8x2xf32>
// CHECK-NEXT:   %13 = "linalg.broadcast"(%0, %12) <{dimensions = array<i64: 2>}> ({
// CHECK-NEXT:   ^bb0(%14: f32, %15: f32):
// CHECK-NEXT:     "linalg.yield"(%14) : (f32) -> ()
// CHECK-NEXT:   }) : (tensor<4x8xf32>, tensor<4x8x2xf32>) -> tensor<4x8x2xf32>
// CHECK-NEXT:   %14 = "linalg.elementwise"(%0, %0, %1) <{kind = #linalg.elementwise_kind<add>, operandSegmentSizes = array<i32: 2, 1>}> ({
// CHECK-NEXT:   ^bb0(%15: f32, %16: f32, %17: f32):
// CHECK-NEXT:     "linalg.yield"(%15) : (f32) -> ()
// CHECK-NEXT:   }) : (tensor<4x8xf32>, tensor<4x8xf32>, tensor<4x8xf32>) -> tensor<4x8xf32>
// CHECK-NEXT:   %15 = "linalg.elementwise"(%0, %1) <{kind = #linalg.elementwise_kind<exp>, operandSegmentSizes = array<i32: 1, 1>, indexing_maps = [#map, #map]}> ({
// CHECK-NEXT:   ^bb0(%16: f32, %17: f32):
// CHECK-NEXT:     "linalg.yield"(%16) : (f32) -> ()
// CHECK-NEXT:   }) : (tensor<4x8xf32>, tensor<4x8xf32>) -> tensor<4x8xf32>
// CHECK-NEXT:   %16, %17, %18 = "test.op"() : () -> (tensor<4x16xf32>, tensor<16x8xf32>, tensor<4x8xf32>)
// CHECK-NEXT:   %19 = "linalg.matmul"(%16, %17, %18) <{operandSegmentSizes = array<i32: 2, 1>}> ({
// CHECK-NEXT:   ^bb0(%20: f32, %21: f32, %22: f32):
// CHECK-NEXT:     "linalg.yield"(%20) : (f32) -> ()
// CHECK-NEXT:   }) : (tensor<4x16xf32>, tensor<16x8xf32>, tensor<4x8xf32>) -> tensor<4x8xf32>
// CHECK-NEXT:   %20 = "linalg.matmul"(%16, %17, %18) <{operandSegmentSizes = array<i32: 2, 1>, indexing_maps = [#map1, #map2, #map3], cast = #linalg.type_fn<cast_unsigned>}> ({
// CHECK-NEXT:   ^bb0(%21: f32, %22: f32, %23: f32):
// CHECK-NEXT:     "linalg.yield"(%21) : (f32) -> ()
// CHECK-NEXT:   }) : (tensor<4x16xf32>, tensor<16x8xf32>, tensor<4x8xf32>) -> tensor<4x8xf32>
// CHECK-NEXT:   %21 = "linalg.contract"(%16, %17, %18) <{indexing_maps = [#map1, #map2, #map3], operandSegmentSizes = array<i32: 2, 1>}> ({
// CHECK-NEXT:   ^bb0(%22: f32, %23: f32, %24: f32):
// CHECK-NEXT:     "linalg.yield"(%22) : (f32) -> ()
// CHECK-NEXT:   }) : (tensor<4x16xf32>, tensor<16x8xf32>, tensor<4x8xf32>) -> tensor<4x8xf32>
// CHECK-NEXT:   %22, %23, %24 = "test.op"() : () -> (tensor<2x4x16xf32>, tensor<2x16x8xf32>, tensor<2x4x8xf32>)
// CHECK-NEXT:   %25 = "linalg.batch_matmul"(%22, %23, %24) <{operandSegmentSizes = array<i32: 2, 1>}> ({
// CHECK-NEXT:   ^bb0(%26: f32, %27: f32, %28: f32):
// CHECK-NEXT:     "linalg.yield"(%26) : (f32) -> ()
// CHECK-NEXT:   }) : (tensor<2x4x16xf32>, tensor<2x16x8xf32>, tensor<2x4x8xf32>) -> tensor<2x4x8xf32>
// CHECK-NEXT:   %26 = "linalg.batch_reduce_matmul"(%22, %23, %18) <{operandSegmentSizes = array<i32: 2, 1>}> ({
// CHECK-NEXT:   ^bb0(%27: f32, %28: f32, %29: f32):
// CHECK-NEXT:     "linalg.yield"(%27) : (f32) -> ()
// CHECK-NEXT:   }) : (tensor<2x4x16xf32>, tensor<2x16x8xf32>, tensor<4x8xf32>) -> tensor<4x8xf32>
// CHECK-NEXT: }
