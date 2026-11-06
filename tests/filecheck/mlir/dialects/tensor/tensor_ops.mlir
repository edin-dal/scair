// RUN: scair-opt %s | mlir-opt --allow-unregistered-dialect --mlir-print-op-generic | scair-opt | filecheck %s

%0:4 = "test.op"() : () -> (tensor<*xf32>, tensor<4x4xf32>, tensor<?x?xf32>, tensor<4x?xf32>)
// CHECK:       builtin.module {
// CHECK-NEXT:    %0, %1, %2, %3 = "test.op"() : () -> (tensor<*xf32>, tensor<4x4xf32>, tensor<?x?xf32>, tensor<4x?xf32>)
%1 = "test.op"() : () -> tensor<4xui32>
// CHECK-NEXT:    %4 = "test.op"() : () -> tensor<4xui32>
%2 = "test.op"() : () -> tensor<4xf32>
// CHECK-NEXT:    %5 = "test.op"() : () -> tensor<4xf32>
%3:3 = "test.op"() : () -> (index, index, index)
// CHECK-NEXT:    %6, %7, %8 = "test.op"() : () -> (index, index, index)
%4 = "test.op"() : () -> f32
// CHECK-NEXT:    %9 = "test.op"() : () -> f32
%5 = "test.op"() : () -> tensor<f32>
// CHECK-NEXT:    %10 = "test.op"() : () -> tensor<f32>
%6 = "test.op"() : () -> tensor<1x1xf32>
// CHECK-NEXT:    %11 = "test.op"() : () -> tensor<1x1xf32>
%7 = "test.op"() : () -> tensor<?x?x?xf32>
// CHECK-NEXT:    %12 = "test.op"() : () -> tensor<?x?x?xf32>
%8 = "test.op"() : () -> tensor<?x?x?x4x?xf32>
// CHECK-NEXT:    %13 = "test.op"() : () -> tensor<?x?x?x4x?xf32>
%9 = "test.op"() : () -> tensor<4x7x3xf32>
// CHECK-NEXT:    %14 = "test.op"() : () -> tensor<4x7x3xf32>
%10 = "test.op"() : () -> tensor<4x4x3xf32>
// CHECK-NEXT:    %15 = "test.op"() : () -> tensor<4x4x3xf32>
%11 = "test.op"() : () -> tensor<8x16x4xf32>
// CHECK-NEXT:    %16 = "test.op"() : () -> tensor<8x16x4xf32>
%12 = "test.op"() : () -> tensor<8x?x4xf32>
// CHECK-NEXT:    %17 = "test.op"() : () -> tensor<8x?x4xf32>
%13 = "test.op"() : () -> tensor<16x32x8xf32>
// CHECK-NEXT:    %18 = "test.op"() : () -> tensor<16x32x8xf32>
%14 = "test.op"() : () -> tensor<4x4xf32>
// CHECK-NEXT:    %19 = "test.op"() : () -> tensor<4x4xf32>
%15 = "test.op"() : () -> tensor<1x2x2x?xf32>
// CHECK-NEXT:    %20 = "test.op"() : () -> tensor<1x2x2x?xf32>
%16 = "test.op"() : () -> tensor<3x4xf32>
// CHECK-NEXT:    %21 = "test.op"() : () -> tensor<3x4xf32>
%17:3 = "test.op"() : () -> (tensor<1xi32>, tensor<2xi32>, tensor<?xi32>)
// CHECK-NEXT:    %22, %23, %24 = "test.op"() : () -> (tensor<1xi32>, tensor<2xi32>, tensor<?xi32>)
%18 = "test.op"() : () -> tensor<?xf32>
// CHECK-NEXT:    %25 = "test.op"() : () -> tensor<?xf32>
%19 = "test.op"() : () -> tensor<?x4xf32>
// CHECK-NEXT:    %26 = "test.op"() : () -> tensor<?x4xf32>

%20 = "tensor.bitcast"(%1) : (tensor<4xui32>) -> tensor<4xi32>
// CHECK-NEXT:    %27 = tensor.bitcast %4 : tensor<4xui32> to tensor<4xi32>
%21 = "tensor.bitcast"(%2) : (tensor<4xf32>) -> tensor<4xi32>
// CHECK-NEXT:    %28 = tensor.bitcast %5 : tensor<4xf32> to tensor<4xi32>

%22 = "tensor.cast"(%0#0) : (tensor<*xf32>) -> tensor<?x?xf32>
// CHECK-NEXT:    %29 = tensor.cast %0 : tensor<*xf32> to tensor<?x?xf32>
%23 = "tensor.cast"(%0#1) : (tensor<4x4xf32>) -> tensor<*xf32>
// CHECK-NEXT:    %30 = tensor.cast %1 : tensor<4x4xf32> to tensor<*xf32>
%24 = "tensor.cast"(%0#2) : (tensor<?x?xf32>) -> tensor<4x?xf32>
// CHECK-NEXT:    %31 = tensor.cast %2 : tensor<?x?xf32> to tensor<4x?xf32>

%25 = "tensor.collapse_shape"(%6) <{reassociation = []}> : (tensor<1x1xf32>) -> tensor<f32>
// CHECK-NEXT:    %32 = tensor.collapse_shape %11 [] : tensor<1x1xf32> into tensor<f32>
%26 = "tensor.collapse_shape"(%8) <{reassociation = [[0], [1], [2, 3, 4]]}> : (tensor<?x?x?x4x?xf32>) -> tensor<?x?x?xf32>
// CHECK-NEXT:    %33 = tensor.collapse_shape %13 [[0 : i64], [1 : i64], [2 : i64, 3 : i64, 4 : i64]] : tensor<?x?x?x4x?xf32> into tensor<?x?x?xf32>

%27 = "tensor.concat"(%9) <{dim = 0 : i64}> : (tensor<4x7x3xf32>) -> tensor<4x7x3xf32>
// CHECK-NEXT:    %34 = "tensor.concat"(%14) <{dim = 0 : i64}> : (tensor<4x7x3xf32>) -> tensor<4x7x3xf32>
%28 = "tensor.concat"(%9, %10) <{dim = 1 : i64}> : (tensor<4x7x3xf32>, tensor<4x4x3xf32>) -> tensor<4x11x3xf32>
// CHECK-NEXT:    %35 = "tensor.concat"(%14, %15) <{dim = 1 : i64}> : (tensor<4x7x3xf32>, tensor<4x4x3xf32>) -> tensor<4x11x3xf32>
%29 = "tensor.concat"(%9, %7) <{dim = 2 : i64}> : (tensor<4x7x3xf32>, tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
// CHECK-NEXT:    %36 = "tensor.concat"(%14, %12) <{dim = 2 : i64}> : (tensor<4x7x3xf32>, tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
%30 = "tensor.concat"(%7, %7) <{dim = 1 : i64}> : (tensor<?x?x?xf32>, tensor<?x?x?xf32>) -> tensor<?x10x?xf32>
// CHECK-NEXT:    %37 = "tensor.concat"(%12, %12) <{dim = 1 : i64}> : (tensor<?x?x?xf32>, tensor<?x?x?xf32>) -> tensor<?x10x?xf32>

%31 = "tensor.dim"(%0#3, %3#0) : (tensor<4x?xf32>, index) -> index
// CHECK-NEXT:    %38 = "tensor.dim"(%3, %6) : (tensor<4x?xf32>, index) -> index
%32 = "tensor.dim"(%0#0, %3#1) : (tensor<*xf32>, index) -> index
// CHECK-NEXT:    %39 = "tensor.dim"(%0, %7) : (tensor<*xf32>, index) -> index

%33 = "tensor.empty"() : () -> tensor<4x5xf32>
// CHECK-NEXT:    %40 = "tensor.empty"() : () -> tensor<4x5xf32>
%34 = "tensor.empty"(%3#0) : (index) -> tensor<5x?x6xf32>
// CHECK-NEXT:    %41 = "tensor.empty"(%6) : (index) -> tensor<5x?x6xf32>
%35 = "tensor.empty"(%3#0, %3#1) : (index, index) -> tensor<?x?xf32>
// CHECK-NEXT:    %42 = "tensor.empty"(%6, %7) : (index, index) -> tensor<?x?xf32>

%36 = "tensor.expand_shape"(%5) <{reassociation = [], static_output_shape = array<i64: 1, 1>}> : (tensor<f32>) -> tensor<1x1xf32>
// CHECK-NEXT:    %43 = "tensor.expand_shape"(%10) <{reassociation = [], static_output_shape = array<i64: 1, 1>}> : (tensor<f32>) -> tensor<1x1xf32>
%37 = "tensor.expand_shape"(%0#2, %3#0, %3#1, %3#2) <{reassociation = [[0, 1], [2, 3]], static_output_shape = array<i64: 5, -9223372036854775808, -9223372036854775808, -9223372036854775808>}> : (tensor<?x?xf32>, index, index, index) -> tensor<5x?x?x?xf32>
// CHECK-NEXT:    %44 = "tensor.expand_shape"(%2, %6, %7, %8) <{reassociation = [[0 : i64, 1 : i64], [2 : i64, 3 : i64]], static_output_shape = array<i64: 5, -9223372036854775808, -9223372036854775808, -9223372036854775808>}> : (tensor<?x?xf32>, index, index, index) -> tensor<5x?x?x?xf32>

%38 = "tensor.extract"(%7, %3#0, %3#1, %3#2) : (tensor<?x?x?xf32>, index, index, index) -> f32
// CHECK-NEXT:    %45 = "tensor.extract"(%12, %6, %7, %8) : (tensor<?x?x?xf32>, index, index, index) -> f32
%39 = "tensor.extract"(%5) : (tensor<f32>) -> f32
// CHECK-NEXT:    %46 = "tensor.extract"(%10) : (tensor<f32>) -> f32

%40 = "tensor.extract_slice"(%11, %3#0, %3#0, %3#0, %3#1, %3#1, %3#1, %3#2, %3#2, %3#2) <{operandSegmentSizes = array<i32: 1, 3, 3, 3>, static_offsets = array<i64: -9223372036854775808, -9223372036854775808, -9223372036854775808>, static_sizes = array<i64: -9223372036854775808, -9223372036854775808, -9223372036854775808>, static_strides = array<i64: -9223372036854775808, -9223372036854775808, -9223372036854775808>}> : (tensor<8x16x4xf32>, index, index, index, index, index, index, index, index, index) -> tensor<?x?x?xf32>
// CHECK-NEXT:    %47 = "tensor.extract_slice"(%16, %6, %6, %6, %7, %7, %7, %8, %8, %8) <{static_offsets = array<i64: -9223372036854775808, -9223372036854775808, -9223372036854775808>, static_sizes = array<i64: -9223372036854775808, -9223372036854775808, -9223372036854775808>, static_strides = array<i64: -9223372036854775808, -9223372036854775808, -9223372036854775808>, operandSegmentSizes = array<i32: 1, 3, 3, 3>}> : (tensor<8x16x4xf32>, index, index, index, index, index, index, index, index, index) -> tensor<?x?x?xf32>
%41 = "tensor.extract_slice"(%11) <{operandSegmentSizes = array<i32: 1, 0, 0, 0>, static_offsets = array<i64: 0, 2, 0>, static_sizes = array<i64: 4, 4, 4>, static_strides = array<i64: 1, 1, 1>}> : (tensor<8x16x4xf32>) -> tensor<4x4x4xf32>
// CHECK-NEXT:    %48 = "tensor.extract_slice"(%16) <{static_offsets = array<i64: 0, 2, 0>, static_sizes = array<i64: 4, 4, 4>, static_strides = array<i64: 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> : (tensor<8x16x4xf32>) -> tensor<4x4x4xf32>
%42 = "tensor.extract_slice"(%11) <{operandSegmentSizes = array<i32: 1, 0, 0, 0>, static_offsets = array<i64: 0, 2, 0>, static_sizes = array<i64: 4, 1, 4>, static_strides = array<i64: 1, 1, 1>}> : (tensor<8x16x4xf32>) -> tensor<4x4xf32>
// CHECK-NEXT:    %49 = "tensor.extract_slice"(%16) <{static_offsets = array<i64: 0, 2, 0>, static_sizes = array<i64: 4, 1, 4>, static_strides = array<i64: 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> : (tensor<8x16x4xf32>) -> tensor<4x4xf32>

%43 = "tensor.from_elements"(%3#0) : (index) -> tensor<1xindex>
// CHECK-NEXT:    %50 = "tensor.from_elements"(%6) : (index) -> tensor<1xindex>
%44 = "tensor.from_elements"(%3#0, %3#1, %3#0, %3#1, %3#0, %3#1) : (index, index, index, index, index, index) -> tensor<2x3xindex>
// CHECK-NEXT:    %51 = "tensor.from_elements"(%6, %7, %6, %7, %6, %7) : (index, index, index, index, index, index) -> tensor<2x3xindex>
%45 = "tensor.from_elements"() : () -> tensor<0xindex>
// CHECK-NEXT:    %52 = "tensor.from_elements"() : () -> tensor<0xindex>
%46 = "tensor.from_elements"(%4) : (f32) -> tensor<f32>
// CHECK-NEXT:    %53 = "tensor.from_elements"(%9) : (f32) -> tensor<f32>

%47 = "tensor.generate"(%3#0, %3#1) ({
^bb0(%arg9: index, %arg10: index, %arg11: index):
  %63 = "test.op"() : () -> f32
  "tensor.yield"(%63) : (f32) -> ()
}) : (index, index) -> tensor<?x3x?xf32>
// CHECK-NEXT:    %54 = "tensor.generate"(%6, %7) ({
// CHECK-NEXT:    ^bb0(%55: index, %56: index, %57: index):
// CHECK-NEXT:      %58 = "test.op"() : () -> f32
// CHECK-NEXT:      tensor.yield %58 : f32
// CHECK-NEXT:    }) : (index, index) -> tensor<?x3x?xf32>
%48 = "tensor.generate"() ({
^bb0(%arg8: index):
  "tensor.yield"(%4) : (f32) -> ()
}) : () -> tensor<3xf32>
// CHECK-NEXT:    %55 = "tensor.generate"() ({
// CHECK-NEXT:    ^bb0(%56: index):
// CHECK-NEXT:      tensor.yield %9 : f32
// CHECK-NEXT:    }) : () -> tensor<3xf32>

%49 = "tensor.insert"(%4, %7, %3#0, %3#1, %3#2) : (f32, tensor<?x?x?xf32>, index, index, index) -> tensor<?x?x?xf32>
// CHECK-NEXT:    %56 = "tensor.insert"(%9, %12, %6, %7, %8) : (f32, tensor<?x?x?xf32>, index, index, index) -> tensor<?x?x?xf32>

%50 = "tensor.insert_slice"(%11, %13, %3#0, %3#0, %3#0, %3#1, %3#2) <{operandSegmentSizes = array<i32: 1, 1, 3, 0, 2>, static_offsets = array<i64: -9223372036854775808, -9223372036854775808, -9223372036854775808>, static_sizes = array<i64: 8, 16, 4>, static_strides = array<i64: -9223372036854775808, 1, -9223372036854775808>}> : (tensor<8x16x4xf32>, tensor<16x32x8xf32>, index, index, index, index, index) -> tensor<16x32x8xf32>
// CHECK-NEXT:    %57 = "tensor.insert_slice"(%16, %18, %6, %6, %6, %7, %8) <{static_offsets = array<i64: -9223372036854775808, -9223372036854775808, -9223372036854775808>, static_sizes = array<i64: 8, 16, 4>, static_strides = array<i64: -9223372036854775808, 1, -9223372036854775808>, operandSegmentSizes = array<i32: 1, 1, 3, 0, 2>}> : (tensor<8x16x4xf32>, tensor<16x32x8xf32>, index, index, index, index, index) -> tensor<16x32x8xf32>
%51 = "tensor.insert_slice"(%14, %11) <{operandSegmentSizes = array<i32: 1, 1, 0, 0, 0>, static_offsets = array<i64: 0, 2, 0>, static_sizes = array<i64: 4, 1, 4>, static_strides = array<i64: 1, 1, 1>}> : (tensor<4x4xf32>, tensor<8x16x4xf32>) -> tensor<8x16x4xf32>
// CHECK-NEXT:    %58 = "tensor.insert_slice"(%19, %16) <{static_offsets = array<i64: 0, 2, 0>, static_sizes = array<i64: 4, 1, 4>, static_strides = array<i64: 1, 1, 1>, operandSegmentSizes = array<i32: 1, 1, 0, 0, 0>}> : (tensor<4x4xf32>, tensor<8x16x4xf32>) -> tensor<8x16x4xf32>
%52 = "tensor.insert_slice"(%12, %11, %3#0, %3#1) <{operandSegmentSizes = array<i32: 1, 1, 1, 1, 0>, static_offsets = array<i64: 0, -9223372036854775808, 0>, static_sizes = array<i64: 8, -9223372036854775808, 4>, static_strides = array<i64: 1, 1, 1>}> : (tensor<8x?x4xf32>, tensor<8x16x4xf32>, index, index) -> tensor<8x16x4xf32>
// CHECK-NEXT:    %59 = "tensor.insert_slice"(%17, %16, %6, %7) <{static_offsets = array<i64: 0, -9223372036854775808, 0>, static_sizes = array<i64: 8, -9223372036854775808, 4>, static_strides = array<i64: 1, 1, 1>, operandSegmentSizes = array<i32: 1, 1, 1, 1, 0>}> : (tensor<8x?x4xf32>, tensor<8x16x4xf32>, index, index) -> tensor<8x16x4xf32>

%53 = "tensor.pad"(%15, %3#0, %3#1) <{operandSegmentSizes = array<i32: 1, 1, 1>, static_high = array<i64: 3, 3, -9223372036854775808, 2>, static_low = array<i64: 2, -9223372036854775808, 3, 3>}> ({
^bb0(%arg4: index, %arg5: index, %arg6: index, %arg7: index):
  "tensor.yield"(%4) : (f32) -> ()
}) : (tensor<1x2x2x?xf32>, index, index) -> tensor<6x?x?x?xf32>
// CHECK-NEXT:    %60 = "tensor.pad"(%20, %6, %7) <{static_low = array<i64: 2, -9223372036854775808, 3, 3>, static_high = array<i64: 3, 3, -9223372036854775808, 2>, operandSegmentSizes = array<i32: 1, 1, 1>}> ({
// CHECK-NEXT:    ^bb0(%61: index, %62: index, %63: index, %64: index):
// CHECK-NEXT:      tensor.yield %9 : f32
// CHECK-NEXT:    }) : (tensor<1x2x2x?xf32>, index, index) -> tensor<6x?x?x?xf32>
%54 = "tensor.pad"(%16) <{operandSegmentSizes = array<i32: 1, 0, 0>, static_high = array<i64: 2, 3>, static_low = array<i64: 1, 2>}> ({
^bb0(%arg2: index, %arg3: index):
  "tensor.yield"(%4) : (f32) -> ()
}) : (tensor<3x4xf32>) -> tensor<6x9xf32>
// CHECK-NEXT:    %61 = "tensor.pad"(%21) <{static_low = array<i64: 1, 2>, static_high = array<i64: 2, 3>, operandSegmentSizes = array<i32: 1, 0, 0>}> ({
// CHECK-NEXT:    ^bb0(%62: index, %63: index):
// CHECK-NEXT:      tensor.yield %9 : f32
// CHECK-NEXT:    }) : (tensor<3x4xf32>) -> tensor<6x9xf32>
%55 = "tensor.pad"(%16) <{nofold, operandSegmentSizes = array<i32: 1, 0, 0>, static_high = array<i64: 2, 3>, static_low = array<i64: 1, 2>}> ({
^bb0(%arg0: index, %arg1: index):
  "tensor.yield"(%4) : (f32) -> ()
}) : (tensor<3x4xf32>) -> tensor<6x9xf32>
// CHECK-NEXT:    %62 = "tensor.pad"(%21) <{static_low = array<i64: 1, 2>, static_high = array<i64: 2, 3>, operandSegmentSizes = array<i32: 1, 0, 0>, nofold}> ({
// CHECK-NEXT:    ^bb0(%63: index, %64: index):
// CHECK-NEXT:      tensor.yield %9 : f32
// CHECK-NEXT:    }) : (tensor<3x4xf32>) -> tensor<6x9xf32>

%56 = "tensor.rank"(%7) : (tensor<?x?x?xf32>) -> index
// CHECK-NEXT:    %63 = "tensor.rank"(%12) : (tensor<?x?x?xf32>) -> index
%57 = "tensor.rank"(%0#0) : (tensor<*xf32>) -> index
// CHECK-NEXT:    %64 = "tensor.rank"(%0) : (tensor<*xf32>) -> index

%58 = "tensor.reshape"(%0#0, %17#0) : (tensor<*xf32>, tensor<1xi32>) -> tensor<?xf32>
// CHECK-NEXT:    %65 = "tensor.reshape"(%0, %22) : (tensor<*xf32>, tensor<1xi32>) -> tensor<?xf32>
%59 = "tensor.reshape"(%18, %17#1) : (tensor<?xf32>, tensor<2xi32>) -> tensor<?x?xf32>
// CHECK-NEXT:    %66 = "tensor.reshape"(%25, %23) : (tensor<?xf32>, tensor<2xi32>) -> tensor<?x?xf32>
%60 = "tensor.reshape"(%0#2, %17#2) : (tensor<?x?xf32>, tensor<?xi32>) -> tensor<*xf32>
// CHECK-NEXT:    %67 = "tensor.reshape"(%2, %24) : (tensor<?x?xf32>, tensor<?xi32>) -> tensor<*xf32>

%61 = "tensor.splat"(%4) : (f32) -> tensor<8xf32>
// CHECK-NEXT:    %68 = "tensor.splat"(%9) : (f32) -> tensor<8xf32>
%62 = "tensor.splat"(%4, %3#0, %3#1) : (f32, index, index) -> tensor<?x8x?xf32>
// CHECK-NEXT:    %69 = "tensor.splat"(%9, %6, %7) : (f32, index, index) -> tensor<?x8x?xf32>
// CHECK-NEXT:  }
