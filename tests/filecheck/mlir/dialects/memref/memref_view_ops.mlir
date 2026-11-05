// RUN: scair-opt %s | mlir-opt --allow-unregistered-dialect --mlir-print-op-generic | scair-opt | filecheck %s

func.func @views(%i: index, %j: index) {
  %m0 = "memref.alloc"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<4x5xf32>
  %m1 = "memref.alloc"(%i) <{operandSegmentSizes = array<i32: 1, 0>}> : (index) -> memref<?x5xf32>
  %m2 = "memref.alloc"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<f32>
  %m3 = "memref.alloc"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<2x3x4xf32>
  %u0 = "test.op"() : () -> memref<*xf32>

  // cast
  %cast0 = "memref.cast"(%m0) : (memref<4x5xf32>) -> memref<?x?xf32>
  %cast1 = "memref.cast"(%m0) : (memref<4x5xf32>) -> memref<*xf32>
  %cast2 = "memref.cast"(%u0) : (memref<*xf32>) -> memref<4x?xf32>

  // memory_space_cast
  %s0 = "test.op"() : () -> memref<?xf32, 5>
  %s1 = "test.op"() : () -> memref<5x4xi32>
  %s2 = "test.op"() : () -> memref<*xf32, 5>
  %msc0 = "memref.memory_space_cast"(%s0) : (memref<?xf32, 5>) -> memref<?xf32>
  %msc1 = "memref.memory_space_cast"(%s1) : (memref<5x4xi32>) -> memref<5x4xi32, 3>
  %msc2 = "memref.memory_space_cast"(%s2) : (memref<*xf32, 5>) -> memref<*xf32, 3>

  // rank
  %rank0 = "memref.rank"(%u0) : (memref<*xf32>) -> index
  %rank1 = "memref.rank"(%m0) : (memref<4x5xf32>) -> index

  // collapse_shape, expand_shape
  %col0 = "memref.collapse_shape"(%m3) <{reassociation = [[0, 1], [2]]}> : (memref<2x3x4xf32>) -> memref<6x4xf32>
  %col1 = "memref.collapse_shape"(%m3) <{reassociation = [[0, 1, 2]]}> : (memref<2x3x4xf32>) -> memref<24xf32>
  %col2 = "memref.collapse_shape"(%m2) <{reassociation = []}> : (memref<f32>) -> memref<f32>
  %exp0 = "memref.expand_shape"(%col0) <{reassociation = [[0, 1], [2]], static_output_shape = array<i64: 2, 3, 4>}> : (memref<6x4xf32>) -> memref<2x3x4xf32>
  %exp1 = "memref.expand_shape"(%m1, %j) <{reassociation = [[0, 1], [2]], static_output_shape = array<i64: -9223372036854775808, 4, 5>}> : (memref<?x5xf32>, index) -> memref<?x4x5xf32>
  %exp2 = "memref.expand_shape"(%m2) <{reassociation = [], static_output_shape = array<i64>}> : (memref<f32>) -> memref<f32>

  // reshape
  %sh0 = "test.op"() : () -> memref<1xi32>
  %sh1 = "test.op"() : () -> memref<2xindex>
  %sh2 = "test.op"() : () -> memref<?xi32>
  %resh0 = "memref.reshape"(%m0, %sh0) : (memref<4x5xf32>, memref<1xi32>) -> memref<20xf32>
  %resh1 = "memref.reshape"(%m0, %sh1) : (memref<4x5xf32>, memref<2xindex>) -> memref<5x4xf32>
  %resh2 = "memref.reshape"(%u0, %sh0) : (memref<*xf32>, memref<1xi32>) -> memref<?xf32>
  %resh3 = "memref.reshape"(%m0, %sh2) : (memref<4x5xf32>, memref<?xi32>) -> memref<*xf32>
  %resh4 = "memref.reshape"(%u0, %sh2) : (memref<*xf32>, memref<?xi32>) -> memref<*xf32>

  // subview
  %sv0 = "memref.subview"(%m3) <{operandSegmentSizes = array<i32: 1, 0, 0, 0>, static_offsets = array<i64: 0, 0, 0>, static_sizes = array<i64: 1, 3, 4>, static_strides = array<i64: 1, 1, 1>}> : (memref<2x3x4xf32>) -> memref<3x4xf32>
  %sv1 = "memref.subview"(%m3) <{operandSegmentSizes = array<i32: 1, 0, 0, 0>, static_offsets = array<i64: 0, 0, 0>, static_sizes = array<i64: 2, 3, 4>, static_strides = array<i64: 1, 1, 1>}> : (memref<2x3x4xf32>) -> memref<2x3x4xf32>
  %sv2 = "memref.subview"(%m1, %i) <{operandSegmentSizes = array<i32: 1, 0, 1, 0>, static_offsets = array<i64: 0, 0>, static_sizes = array<i64: -9223372036854775808, 5>, static_strides = array<i64: 1, 1>}> : (memref<?x5xf32>, index) -> memref<?x5xf32>
  %sv3 = "memref.subview"(%m3, %j) <{operandSegmentSizes = array<i32: 1, 0, 0, 1>, static_offsets = array<i64: 0, 0, 0>, static_sizes = array<i64: 1, 3, 4>, static_strides = array<i64: -9223372036854775808, 1, 1>}> : (memref<2x3x4xf32>, index) -> memref<3x4xf32>

  // transpose
  %tr0 = "memref.transpose"(%m0) <{permutation = affine_map<(d0, d1) -> (d0, d1)>}> : (memref<4x5xf32>) -> memref<4x5xf32>
  %tr1 = "memref.transpose"(%m2) <{permutation = affine_map<() -> ()>}> : (memref<f32>) -> memref<f32>

  // view
  %bytes = "memref.alloc"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<2048xi8>
  %v0 = "memref.view"(%bytes, %i) : (memref<2048xi8>, index) -> memref<64x4xf32>
  %v1 = "memref.view"(%bytes, %i, %i, %j) : (memref<2048xi8>, index, index, index) -> memref<?x4x?xf32>

  // extract_strided_metadata
  %b0, %o0 = "memref.extract_strided_metadata"(%m2) : (memref<f32>) -> (memref<f32>, index)
  %b1, %o1, %sz1, %st1 = "memref.extract_strided_metadata"(%bytes) : (memref<2048xi8>) -> (memref<i8>, index, index, index)
  %b2, %o2, %sz2:3, %st2:3 = "memref.extract_strided_metadata"(%m3) : (memref<2x3x4xf32>) -> (memref<f32>, index, index, index, index, index, index, index)
  func.return
}

// CHECK:       #map = affine_map<(d0, d1)[] -> (d0, d1)>
// CHECK-NEXT:  #map1 = affine_map<()[] -> ()>
// CHECK-NEXT:  builtin.module {
// CHECK-NEXT:    func.func @views(%0: index, %1: index) {
// CHECK-NEXT:      %2 = "memref.alloc"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<4x5xf32>
// CHECK-NEXT:      %3 = "memref.alloc"(%0) <{operandSegmentSizes = array<i32: 1, 0>}> : (index) -> memref<?x5xf32>
// CHECK-NEXT:      %4 = "memref.alloc"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<f32>
// CHECK-NEXT:      %5 = "memref.alloc"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<2x3x4xf32>
// CHECK-NEXT:      %6 = "test.op"() : () -> memref<*xf32>
// CHECK-NEXT:      %7 = memref.cast %2 : memref<4x5xf32> to memref<?x?xf32>
// CHECK-NEXT:      %8 = memref.cast %2 : memref<4x5xf32> to memref<*xf32>
// CHECK-NEXT:      %9 = memref.cast %6 : memref<*xf32> to memref<4x?xf32>
// CHECK-NEXT:      %10 = "test.op"() : () -> memref<?xf32, 5 : i64>
// CHECK-NEXT:      %11 = "test.op"() : () -> memref<5x4xi32>
// CHECK-NEXT:      %12 = "test.op"() : () -> memref<*xf32, 5 : i64>
// CHECK-NEXT:      %13 = memref.memory_space_cast %10 : memref<?xf32, 5 : i64> to memref<?xf32>
// CHECK-NEXT:      %14 = memref.memory_space_cast %11 : memref<5x4xi32> to memref<5x4xi32, 3 : i64>
// CHECK-NEXT:      %15 = memref.memory_space_cast %12 : memref<*xf32, 5 : i64> to memref<*xf32, 3 : i64>
// CHECK-NEXT:      %16 = "memref.rank"(%6) : (memref<*xf32>) -> index
// CHECK-NEXT:      %17 = "memref.rank"(%2) : (memref<4x5xf32>) -> index
// CHECK-NEXT:      %18 = memref.collapse_shape %5 [[0 : i64, 1 : i64], [2 : i64]] : memref<2x3x4xf32> into memref<6x4xf32>
// CHECK-NEXT:      %19 = memref.collapse_shape %5 [[0 : i64, 1 : i64, 2 : i64]] : memref<2x3x4xf32> into memref<24xf32>
// CHECK-NEXT:      %20 = memref.collapse_shape %4 [] : memref<f32> into memref<f32>
// CHECK-NEXT:      %21 = "memref.expand_shape"(%18) <{reassociation = [[0 : i64, 1 : i64], [2 : i64]], static_output_shape = array<i64: 2, 3, 4>}> : (memref<6x4xf32>) -> memref<2x3x4xf32>
// CHECK-NEXT:      %22 = "memref.expand_shape"(%3, %1) <{reassociation = [[0 : i64, 1 : i64], [2 : i64]], static_output_shape = array<i64: -9223372036854775808, 4, 5>}> : (memref<?x5xf32>, index) -> memref<?x4x5xf32>
// CHECK-NEXT:      %23 = "memref.expand_shape"(%4) <{reassociation = [], static_output_shape = array<i64>}> : (memref<f32>) -> memref<f32>
// CHECK-NEXT:      %24 = "test.op"() : () -> memref<1xi32>
// CHECK-NEXT:      %25 = "test.op"() : () -> memref<2xindex>
// CHECK-NEXT:      %26 = "test.op"() : () -> memref<?xi32>
// CHECK-NEXT:      %27 = "memref.reshape"(%2, %24) : (memref<4x5xf32>, memref<1xi32>) -> memref<20xf32>
// CHECK-NEXT:      %28 = "memref.reshape"(%2, %25) : (memref<4x5xf32>, memref<2xindex>) -> memref<5x4xf32>
// CHECK-NEXT:      %29 = "memref.reshape"(%6, %24) : (memref<*xf32>, memref<1xi32>) -> memref<?xf32>
// CHECK-NEXT:      %30 = "memref.reshape"(%2, %26) : (memref<4x5xf32>, memref<?xi32>) -> memref<*xf32>
// CHECK-NEXT:      %31 = "memref.reshape"(%6, %26) : (memref<*xf32>, memref<?xi32>) -> memref<*xf32>
// CHECK-NEXT:      %32 = "memref.subview"(%5) <{static_offsets = array<i64: 0, 0, 0>, static_sizes = array<i64: 1, 3, 4>, static_strides = array<i64: 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> : (memref<2x3x4xf32>) -> memref<3x4xf32>
// CHECK-NEXT:      %33 = "memref.subview"(%5) <{static_offsets = array<i64: 0, 0, 0>, static_sizes = array<i64: 2, 3, 4>, static_strides = array<i64: 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> : (memref<2x3x4xf32>) -> memref<2x3x4xf32>
// CHECK-NEXT:      %34 = "memref.subview"(%3, %0) <{static_offsets = array<i64: 0, 0>, static_sizes = array<i64: -9223372036854775808, 5>, static_strides = array<i64: 1, 1>, operandSegmentSizes = array<i32: 1, 0, 1, 0>}> : (memref<?x5xf32>, index) -> memref<?x5xf32>
// CHECK-NEXT:      %35 = "memref.subview"(%5, %1) <{static_offsets = array<i64: 0, 0, 0>, static_sizes = array<i64: 1, 3, 4>, static_strides = array<i64: -9223372036854775808, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 1>}> : (memref<2x3x4xf32>, index) -> memref<3x4xf32>
// CHECK-NEXT:      %36 = "memref.transpose"(%2) <{permutation = #map}> : (memref<4x5xf32>) -> memref<4x5xf32>
// CHECK-NEXT:      %37 = "memref.transpose"(%4) <{permutation = #map1}> : (memref<f32>) -> memref<f32>
// CHECK-NEXT:      %38 = "memref.alloc"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<2048xi8>
// CHECK-NEXT:      %39 = "memref.view"(%38, %0) : (memref<2048xi8>, index) -> memref<64x4xf32>
// CHECK-NEXT:      %40 = "memref.view"(%38, %0, %0, %1) : (memref<2048xi8>, index, index, index) -> memref<?x4x?xf32>
// CHECK-NEXT:      %41, %42 = "memref.extract_strided_metadata"(%4) : (memref<f32>) -> (memref<f32>, index)
// CHECK-NEXT:      %43, %44, %45, %46 = "memref.extract_strided_metadata"(%38) : (memref<2048xi8>) -> (memref<i8>, index, index, index)
// CHECK-NEXT:      %47, %48, %49, %50, %51, %52, %53, %54 = "memref.extract_strided_metadata"(%5) : (memref<2x3x4xf32>) -> (memref<f32>, index, index, index, index, index, index, index)
// CHECK-NEXT:      func.return
// CHECK-NEXT:    }
// CHECK-NEXT:  }
