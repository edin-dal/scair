// RUN: scair-opt %s | filecheck %s

%N, %value, %memref, %zero = "test.op"() : () -> (index, f64, memref<2x3xf64>, index)

"affine.for"() <{"lowerBoundMap" = affine_map<() -> (0)>, "upperBoundMap" = affine_map<() -> (256)>, "step" = 1 : index, "operandSegmentSizes" = array<i32: 0, 0, 0>}> ({
^0(%i : index):
  "affine.yield"() : () -> ()
}) : () -> ()

"affine.parallel"(%N) <{"lowerBoundsMap" = affine_map<() -> (0)>, "lowerBoundsGroups" = dense<1> : vector<1xi32>, "upperBoundsMap" = affine_map<()[s0] -> (s0)>, "upperBoundsGroups" = dense<1> : vector<1xi32>, "steps" = [1 : i64], "reductions" = []}> ({
^1(%i : index):
  "affine.yield"() : () -> ()
}) : (index) -> ()

"affine.store"(%value, %memref) <{"map" = affine_map<() -> (0, 0)>}> : (f64, memref<2x3xf64>) -> ()

%2 = "affine.apply"(%zero, %zero) <{map = affine_map<(d0, d1) -> (d0 + d1, d1)>}> : (index, index) -> index

%min = "affine.min"(%zero) <{"map" = affine_map<(d0) -> ((d0 + 41), d0)>}> : (index) -> index

%same_value = "affine.load"(%memref, %zero, %zero) <{"map" = affine_map<(d0, d1) -> (d0, d1)>}> : (memref<2x3xf64>, index, index) -> f64

"affine.if"() <{"condition" = affine_set<() : (0 == 0)>}> ({
  "affine.yield"() : () -> ()
}, {
}) : () -> ()
"affine.if"() <{"condition" = affine_set<() : (0 == 0)>}> ({
  "affine.yield"() : () -> ()
}, {
  "affine.yield"() : () -> ()
}) : () -> ()

// CHECK:       builtin.module {
// CHECK-NEXT:    %0, %1, %2, %3 = "test.op"() : () -> (index, f64, memref<2x3xf64>, index)
// CHECK-NEXT:    "affine.for"() <{lowerBoundMap = affine_map<()[] -> (0)>, upperBoundMap = affine_map<()[] -> (256)>, step = 1 : index, operandSegmentSizes = array<i32: 0, 0, 0>}> ({
// CHECK-NEXT:    ^bb0(%4: index):
// CHECK-NEXT:      "affine.yield"() : () -> ()
// CHECK-NEXT:    }) : () -> ()
// CHECK-NEXT:    "affine.parallel"(%0) <{upperBoundsGroups = dense<1> : vector<1xi32>, upperBoundsMap = affine_map<()[s0] -> (s0)>, lowerBoundsMap = affine_map<()[] -> (0)>, lowerBoundsGroups = dense<1> : vector<1xi32>, reductions = [], steps = [1]}> ({
// CHECK-NEXT:    ^bb0(%4: index):
// CHECK-NEXT:      "affine.yield"() : () -> ()
// CHECK-NEXT:    }) : (index) -> ()
// CHECK-NEXT:    "affine.store"(%1, %2) <{map = affine_map<()[] -> (0, 0)>}> : (f64, memref<2x3xf64>) -> ()
// CHECK-NEXT:    %4 = "affine.apply"(%3, %3) <{map = affine_map<(d0, d1)[] -> (d0 + d1, d1)>}> : (index, index) -> (index)
// CHECK-NEXT:    %5 = "affine.min"(%3) <{map = affine_map<(d0)[] -> (d0 + 41, d0)>}> : (index) -> (index)
// CHECK-NEXT:    %6 = "affine.load"(%2, %3, %3) <{map = affine_map<(d0, d1)[] -> (d0, d1)>}> : (memref<2x3xf64>, index, index) -> (f64)
// CHECK-NEXT:    "affine.if"() <{condition = affine_set<()[]: (0 == 0)>}> ({
// CHECK-NEXT:      "affine.yield"() : () -> ()
// CHECK-NEXT:    }, {

// CHECK:         }) : () -> ()
// CHECK-NEXT:    "affine.if"() <{condition = affine_set<()[]: (0 == 0)>}> ({
// CHECK-NEXT:      "affine.yield"() : () -> ()
// CHECK-NEXT:    }, {
// CHECK-NEXT:      "affine.yield"() : () -> ()
// CHECK-NEXT:    }) : () -> ()
// CHECK-NEXT:  }
