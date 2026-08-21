// RUN: scair-opt %s | mlir-opt --mlir-print-op-generic | scair-opt | filecheck %s

// Round-trips the newer affine ops through real mlir-opt to check ScaIR emits
// MLIR-faithful generic syntax (op names, attribute names, operand/result shapes).
// affine.dma_start / affine.dma_wait are omitted: real mlir-opt stores their maps
// as discardable attributes ({...}) rather than properties (<{...}>), which ScaIR's
// op framework does not yet emit. They are covered by scair round-trip in
// affine_ops.mlir instead.

%memref, %zero = "test.op"() : () -> (memref<2x3xf64>, index)

%max = "affine.max"(%zero) <{"map" = affine_map<(d0) -> (d0 + 41, d0)>}> : (index) -> index

%vec = "affine.vector_load"(%memref, %zero, %zero) <{"map" = affine_map<(d0, d1) -> (d0, d1)>}> : (memref<2x3xf64>, index, index) -> vector<8xf64>

"affine.vector_store"(%vec, %memref, %zero, %zero) <{"map" = affine_map<(d0, d1) -> (d0, d1)>}> : (vector<8xf64>, memref<2x3xf64>, index, index) -> ()

"affine.prefetch"(%memref, %zero, %zero) <{"isWrite" = 0 : i1, "localityHint" = 3 : i32, "isDataCache" = 1 : i1, "map" = affine_map<(d0, d1) -> (d0, d1)>}> : (memref<2x3xf64>, index, index) -> ()

%mi:2 = "affine.delinearize_index"(%zero) <{"static_basis" = array<i64: 2, 3>}> : (index) -> (index, index)

%lin = "affine.linearize_index"(%zero, %zero) <{"static_basis" = array<i64: 2, 3>, "operandSegmentSizes" = array<i32: 2, 0>, "disjoint"}> : (index, index) -> index

// CHECK:       #map = affine_map<(d0)[] -> (d0 + 41, d0)>
// CHECK-NEXT:  #map1 = affine_map<(d0, d1)[] -> (d0, d1)>
// CHECK-NEXT:  builtin.module {
// CHECK-NEXT:    %0, %1 = "test.op"() : () -> (memref<2x3xf64>, index)
// CHECK-NEXT:    %2 = "affine.max"(%1) <{map = #map}> : (index) -> index
// CHECK-NEXT:    %3 = "affine.vector_load"(%0, %1, %1) <{map = #map1}> : (memref<2x3xf64>, index, index) -> vector<8xf64>
// CHECK-NEXT:    "affine.vector_store"(%3, %0, %1, %1) <{map = #map1}> : (vector<8xf64>, memref<2x3xf64>, index, index) -> ()
// CHECK-NEXT:    "affine.prefetch"(%0, %1, %1) <{isWrite = false, localityHint = 3 : i32, isDataCache = true, map = #map1}> : (memref<2x3xf64>, index, index) -> ()
// CHECK-NEXT:    %4, %5 = "affine.delinearize_index"(%1) <{static_basis = array<i64: 2, 3>}> : (index) -> (index, index)
// CHECK-NEXT:    %6 = "affine.linearize_index"(%1, %1) <{static_basis = array<i64: 2, 3>, operandSegmentSizes = array<i32: 2, 0>, disjoint}> : (index, index) -> index
// CHECK-NEXT:  }
