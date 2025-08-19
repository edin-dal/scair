// RUN: scair-opt %s | scair-opt | filecheck %s
#map = affine_map<()[] -> (0)>
#map1 = affine_map<()[] -> (256)>
#map2 = affine_map<()[s0] -> (s0)>
#map3 = affine_map<()[] -> (0, 0)>
#map4 = affine_map<(d0, d1)[] -> (d0 + d1)>
#map5 = affine_map<(d0)[] -> (d0 + 41, d0)>
#map6 = affine_map<(d0, d1)[] -> (d0, d1)>
#set = affine_set<()[]: (0 == 0)>

%N, %value, %memref, %zero = "test.op"() : () -> (index, f64, memref<2x3xf64>, index)

"affine.for"() <{"lowerBoundMap" = #map, "upperBoundMap" = #map1, "step" = 1 : index, "operandSegmentSizes" = array<i32: 0, 0, 0>}> ({
^0(%i : index):
  "affine.yield"() : () -> ()
}) : () -> ()

"affine.parallel"(%N) <{"lowerBoundsMap" = #map, "lowerBoundsGroups" = dense<1> : vector<1xi32>, "upperBoundsMap" = #map2, "upperBoundsGroups" = dense<1> : vector<1xi32>, "steps" = [1 : i64], "reductions" = []}> ({
^1(%i : index):
  "affine.yield"() : () -> ()
}) : (index) -> ()

"affine.store"(%value, %memref) <{"map" = #map3}> : (f64, memref<2x3xf64>) -> ()

%2 = "affine.apply"(%zero, %zero) <{map = #map4}> : (index, index) -> index

%min = "affine.min"(%zero) <{"map" = #map5}> : (index) -> index

%same_value = "affine.load"(%memref, %zero, %zero) <{"map" = #map6}> : (memref<2x3xf64>, index, index) -> f64

"affine.if"() <{"condition" = #set}> ({
  "affine.yield"() : () -> ()
}, {
}) : () -> ()
"affine.if"() <{"condition" = #set}> ({
  "affine.yield"() : () -> ()
}, {
  "affine.yield"() : () -> ()
}) : () -> ()
// CHECK:       #map = affine_map<()[] -> (0)>
// CHECK-NEXT:  #map1 = affine_map<()[] -> (256)>
// CHECK-NEXT:  #map2 = affine_map<()[s0] -> (s0)>
// CHECK-NEXT:  #map3 = affine_map<()[] -> (0, 0)>
// CHECK-NEXT:  #map4 = affine_map<(d0, d1)[] -> (d0 + d1)>
// CHECK-NEXT:  #map5 = affine_map<(d0)[] -> (d0 + 41, d0)>
// CHECK-NEXT:  #map6 = affine_map<(d0, d1)[] -> (d0, d1)>
// CHECK-NEXT:  #set = affine_set<()[]: (0 == 0)>
// CHECK-NEXT:  builtin.module {
// CHECK-NEXT:    %0, %1, %2, %3 = "test.op"() : () -> (index, f64, memref<2x3xf64>, index)
// CHECK-NEXT:    "affine.for"() <{lowerBoundMap = #map, upperBoundMap = #map1, step = 1 : index, operandSegmentSizes = array<i32: 0, 0, 0>}> ({
// CHECK-NEXT:    ^bb0(%4: index):
// CHECK-NEXT:      "affine.yield"() : () -> ()
// CHECK-NEXT:    }) : () -> ()
// CHECK-NEXT:    "affine.parallel"(%0) <{upperBoundsGroups = dense<1> : vector<1xi32>, upperBoundsMap = #map2, lowerBoundsMap = #map, lowerBoundsGroups = dense<1> : vector<1xi32>, reductions = [], steps = [1]}> ({
// CHECK-NEXT:    ^bb0(%4: index):
// CHECK-NEXT:      "affine.yield"() : () -> ()
// CHECK-NEXT:    }) : (index) -> ()
// CHECK-NEXT:    "affine.store"(%1, %2) <{map = #map3}> : (f64, memref<2x3xf64>) -> ()
// CHECK-NEXT:    %4 = "affine.apply"(%3, %3) <{map = #map4}> : (index, index) -> (index)
// CHECK-NEXT:    %5 = "affine.min"(%3) <{map = #map5}> : (index) -> (index)
// CHECK-NEXT:    %6 = "affine.load"(%2, %3, %3) <{map = #map6}> : (memref<2x3xf64>, index, index) -> (f64)
// CHECK-NEXT:    "affine.if"() <{condition = #set}> ({
// CHECK-NEXT:      "affine.yield"() : () -> ()
// CHECK-NEXT:    }, {
// CHECK-NEXT:    }) : () -> ()
// CHECK-NEXT:    "affine.if"() <{condition = #set}> ({
// CHECK-NEXT:      "affine.yield"() : () -> ()
// CHECK-NEXT:    }, {
// CHECK-NEXT:      "affine.yield"() : () -> ()
// CHECK-NEXT:    }) : () -> ()
// CHECK-NEXT:  }
