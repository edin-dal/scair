// RUN: scair-opt %s | mlir-opt --allow-unregistered-dialect | sed 's/^module {/builtin.module {/' | scair-opt | filecheck %s

%unr, %t44, %tdd = "test.op"() : () -> (tensor<*xf32>, tensor<4x4xf32>, tensor<?x?xf32>)
// CHECK:       builtin.module {
// CHECK-NEXT:    %0, %1, %2 = "test.op"() : () -> (tensor<*xf32>, tensor<4x4xf32>, tensor<?x?xf32>)
%tu, %tf = "test.op"() : () -> (tensor<4xui32>, tensor<4xf32>)
// CHECK-NEXT:    %3, %4 = "test.op"() : () -> (tensor<4xui32>, tensor<4xf32>)
%t11, %t5d, %t4d = "test.op"() : () -> (tensor<1x1xf32>, tensor<?x?x?x4x?xf32>, tensor<4x5xf32>)

// CHECK-NEXT:    %5, %6, %7 = "test.op"() : () -> (tensor<1x1xf32>, tensor<?x?x?x4x?xf32>, tensor<4x5xf32>)

%b0 = tensor.bitcast %tu : tensor<4xui32> to tensor<4xi32>
// CHECK-NEXT:    %8 = tensor.bitcast %3 : tensor<4xui32> to tensor<4xi32>
%b1 = tensor.bitcast %tf : tensor<4xf32> to tensor<4xi32>

// CHECK-NEXT:    %9 = tensor.bitcast %4 : tensor<4xf32> to tensor<4xi32>

%c0 = tensor.cast %unr : tensor<*xf32> to tensor<?x?xf32>
// CHECK-NEXT:    %10 = tensor.cast %0 : tensor<*xf32> to tensor<?x?xf32>
%c1 = tensor.cast %t44 : tensor<4x4xf32> to tensor<*xf32>
// CHECK-NEXT:    %11 = tensor.cast %1 : tensor<4x4xf32> to tensor<*xf32>
%c2 = tensor.cast %tdd : tensor<?x?xf32> to tensor<4x?xf32>

// CHECK-NEXT:    %12 = tensor.cast %2 : tensor<?x?xf32> to tensor<4x?xf32>

%cs0 = tensor.collapse_shape %t11 [] : tensor<1x1xf32> into tensor<f32>
// CHECK-NEXT:    %13 = tensor.collapse_shape %5 [] : tensor<1x1xf32> into tensor<f32>
%cs1 = tensor.collapse_shape %t5d [[0], [1], [2, 3, 4]] : tensor<?x?x?x4x?xf32> into tensor<?x?x?xf32>
// CHECK-NEXT:    %14 = tensor.collapse_shape %6 [[0 : i64], [1 : i64], [2 : i64, 3 : i64, 4 : i64]] : tensor<?x?x?x4x?xf32> into tensor<?x?x?xf32>
%cs2 = tensor.collapse_shape %t4d [[0, 1]] : tensor<4x5xf32> into tensor<20xf32>
// CHECK-NEXT:    %15 = tensor.collapse_shape %7 [[0 : i64, 1 : i64]] : tensor<4x5xf32> into tensor<20xf32>
// CHECK-NEXT:  }
