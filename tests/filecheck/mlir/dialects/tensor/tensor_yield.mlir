// RUN: scair-opt %s | mlir-opt --allow-unregistered-dialect --mlir-print-op-generic | scair-opt | filecheck %s

%i, %j = "test.op"() : () -> (index, index)
// CHECK:       builtin.module {
// CHECK-NEXT:    %0, %1 = "test.op"() : () -> (index, index)
%f, %n = "test.op"() : () -> (f32, i32)
// CHECK-NEXT:    %2, %3 = "test.op"() : () -> (f32, i32)
%t = "test.op"() : () -> tensor<3x4xf32>

// CHECK-NEXT:    %4 = "test.op"() : () -> tensor<3x4xf32>

%g0 = "tensor.generate"(%i, %j) ({
^bb0(%a: index, %b: index, %c: index):
  tensor.yield %f : f32
}) : (index, index) -> tensor<?x3x?xf32>

// CHECK-NEXT:    %5 = "tensor.generate"(%0, %1) ({
// CHECK-NEXT:    ^bb0(%6: index, %7: index, %8: index):
// CHECK-NEXT:      tensor.yield %2 : f32
// CHECK-NEXT:    }) : (index, index) -> tensor<?x3x?xf32>
%g1 = "tensor.generate"() ({
^bb0(%a: index):
  tensor.yield %n : i32
}) : () -> tensor<3xi32>

// CHECK-NEXT:    %6 = "tensor.generate"() ({
// CHECK-NEXT:    ^bb0(%7: index):
// CHECK-NEXT:      tensor.yield %3 : i32
// CHECK-NEXT:    }) : () -> tensor<3xi32>

%p0 = "tensor.pad"(%t) <{operandSegmentSizes = array<i32: 1, 0, 0>, static_high = array<i64: 2, 3>, static_low = array<i64: 1, 2>}> ({
^bb0(%a: index, %b: index):
  tensor.yield %f : f32
}) : (tensor<3x4xf32>) -> tensor<6x9xf32>
// CHECK-NEXT:    %7 = "tensor.pad"(%4) <{static_low = array<i64: 1, 2>, static_high = array<i64: 2, 3>, operandSegmentSizes = array<i32: 1, 0, 0>}> ({
// CHECK-NEXT:    ^bb0(%8: index, %9: index):
// CHECK-NEXT:      tensor.yield %2 : f32
// CHECK-NEXT:    }) : (tensor<3x4xf32>) -> tensor<6x9xf32>
// CHECK-NEXT:  }
