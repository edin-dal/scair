// RUN: scair-opt %s | filecheck %s

%N, %value, %memref, %zero = "test.op"() : () -> (index, f64, memref<2x3xf64>, index)

"affine.for"() <{"lowerBoundMap" = affine_map<() -> (0)>, "upperBoundMap" = affine_map<() -> (256)>, "step" = 1 : index, "operandSegmentSizes" = array<i32: 0, 0, 0>}> ({
^0(%i : index):
  affine.yield
}) : () -> ()

"affine.parallel"(%N) <{"lowerBoundsMap" = affine_map<() -> (0)>, "lowerBoundsGroups" = dense<1> : vector<1xi32>, "upperBoundsMap" = affine_map<()[s0] -> (s0)>, "upperBoundsGroups" = dense<1> : vector<1xi32>, "steps" = [1 : i64], "reductions" = []}> ({
^1(%i : index):
  affine.yield
}) : (index) -> ()

"affine.store"(%value, %memref) <{"map" = affine_map<() -> (0, 0)>}> : (f64, memref<2x3xf64>) -> ()

%2 = "affine.apply"(%zero, %zero) <{map = affine_map<(d0, d1) -> (d0 + d1, d1)>}> : (index, index) -> index

%min = "affine.min"(%zero) <{"map" = affine_map<(d0) -> (d0 + 41, d0)>}> : (index) -> index

%same_value = "affine.load"(%memref, %zero, %zero) <{"map" = affine_map<(d0, d1) -> (d0, d1)>}> : (memref<2x3xf64>, index, index) -> f64

"affine.if"() <{"condition" = affine_set<() : (0 == 0)>}> ({
  affine.yield
}, {
}) : () -> ()
"affine.if"() <{"condition" = affine_set<() : (0 == 0)>}> ({
  affine.yield
}, {
  affine.yield
}) : () -> ()

%max = "affine.max"(%zero) <{"map" = affine_map<(d0) -> (d0 + 41, d0)>}> : (index) -> index

%vec = "affine.vector_load"(%memref, %zero, %zero) <{"map" = affine_map<(d0, d1) -> (d0, d1)>}> : (memref<2x3xf64>, index, index) -> vector<8xf64>

"affine.vector_store"(%vec, %memref, %zero, %zero) <{"map" = affine_map<(d0, d1) -> (d0, d1)>}> : (vector<8xf64>, memref<2x3xf64>, index, index) -> ()

"affine.prefetch"(%memref, %zero, %zero) <{"isWrite" = 0 : i1, "localityHint" = 3 : i32, "isDataCache" = 1 : i1, "map" = affine_map<(d0, d1) -> (d0, d1)>}> : (memref<2x3xf64>, index, index) -> ()

%mi:2 = "affine.delinearize_index"(%zero) <{"static_basis" = array<i64: 2, 3>}> : (index) -> (index, index)

%lin = "affine.linearize_index"(%zero, %zero) <{"static_basis" = array<i64: 2, 3>, "operandSegmentSizes" = array<i32: 2, 0>}> : (index, index) -> index

"affine.dma_start"(%memref, %zero, %memref, %zero, %memref, %zero) {"src_map" = affine_map<(d0) -> (d0)>, "dst_map" = affine_map<(d0) -> (d0)>, "tag_map" = affine_map<(d0) -> (d0)>} : (memref<2x3xf64>, index, memref<2x3xf64>, index, memref<2x3xf64>, index) -> ()

"affine.dma_wait"(%memref, %zero) {"tag_map" = affine_map<(d0) -> (d0)>} : (memref<2x3xf64>, index) -> ()

// CHECK:       #map = affine_map<()[] -> (0)>
// CHECK-NEXT:  #map1 = affine_map<()[] -> (256)>
// CHECK-NEXT:  #map2 = affine_map<()[s0] -> (s0)>
// CHECK-NEXT:  #map3 = affine_map<()[] -> (0, 0)>
// CHECK-NEXT:  #map4 = affine_map<(d0, d1)[] -> (d0 + d1, d1)>
// CHECK-NEXT:  #map5 = affine_map<(d0)[] -> (d0 + 41, d0)>
// CHECK-NEXT:  #map6 = affine_map<(d0, d1)[] -> (d0, d1)>
// CHECK-NEXT:  #set = affine_set<()[]: (0 == 0)>
// CHECK-NEXT:  #map7 = affine_map<(d0)[] -> (d0)>
// CHECK-NEXT:  builtin.module {
// CHECK-NEXT:    %0, %1, %2, %3 = "test.op"() : () -> (index, f64, memref<2x3xf64>, index)
// CHECK-NEXT:    "affine.for"() <{lowerBoundMap = #map, upperBoundMap = #map1, step = 1 : index, operandSegmentSizes = array<i32: 0, 0, 0>}> ({
// CHECK-NEXT:    ^bb0(%4: index):
// CHECK-NEXT:      affine.yield
// CHECK-NEXT:    }) : () -> ()
// CHECK-NEXT:    "affine.parallel"(%0) <{upperBoundsGroups = dense<1> : vector<1xi32>, upperBoundsMap = #map2, lowerBoundsMap = #map, lowerBoundsGroups = dense<1> : vector<1xi32>, reductions = [], steps = [1]}> ({
// CHECK-NEXT:    ^bb0(%4: index):
// CHECK-NEXT:      affine.yield
// CHECK-NEXT:    }) : (index) -> ()
// CHECK-NEXT:    "affine.store"(%1, %2) <{map = #map3}> : (f64, memref<2x3xf64>) -> ()
// CHECK-NEXT:    %4 = "affine.apply"(%3, %3) <{map = #map4}> : (index, index) -> index
// CHECK-NEXT:    %5 = "affine.min"(%3) <{map = #map5}> : (index) -> index
// CHECK-NEXT:    %6 = "affine.load"(%2, %3, %3) <{map = #map6}> : (memref<2x3xf64>, index, index) -> f64
// CHECK-NEXT:    "affine.if"() <{condition = #set}> ({
// CHECK-NEXT:      affine.yield
// CHECK-NEXT:    }, {
// CHECK-NEXT:    }) : () -> ()
// CHECK-NEXT:    "affine.if"() <{condition = #set}> ({
// CHECK-NEXT:      affine.yield
// CHECK-NEXT:    }, {
// CHECK-NEXT:      affine.yield
// CHECK-NEXT:    }) : () -> ()
// CHECK-NEXT:    %7 = "affine.max"(%3) <{map = #map5}> : (index) -> index
// CHECK-NEXT:    %8 = "affine.vector_load"(%2, %3, %3) <{map = #map6}> : (memref<2x3xf64>, index, index) -> vector<8xf64>
// CHECK-NEXT:    "affine.vector_store"(%8, %2, %3, %3) <{map = #map6}> : (vector<8xf64>, memref<2x3xf64>, index, index) -> ()
// CHECK-NEXT:    "affine.prefetch"(%2, %3, %3) <{isWrite = false, localityHint = 3 : i32, isDataCache = true, map = #map6}> : (memref<2x3xf64>, index, index) -> ()
// CHECK-NEXT:    %9, %10 = "affine.delinearize_index"(%3) <{static_basis = array<i64: 2, 3>}> : (index) -> (index, index)
// CHECK-NEXT:    %11 = "affine.linearize_index"(%3, %3) <{static_basis = array<i64: 2, 3>, operandSegmentSizes = array<i32: 2, 0>}> : (index, index) -> index
// CHECK-NEXT:    "affine.dma_start"(%2, %3, %2, %3, %2, %3) {src_map = #map7, dst_map = #map7, tag_map = #map7} : (memref<2x3xf64>, index, memref<2x3xf64>, index, memref<2x3xf64>, index) -> ()
// CHECK-NEXT:    "affine.dma_wait"(%2, %3) {tag_map = #map7} : (memref<2x3xf64>, index) -> ()
// CHECK-NEXT:  }
