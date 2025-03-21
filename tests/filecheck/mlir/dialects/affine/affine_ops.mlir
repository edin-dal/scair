// RUN: scair-opt %s | mlir-opt --mlir-print-op-generic --mlir-print-local-scope | scair-opt | filecheck %s

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

%2 = "affine.apply"(%zero, %zero) <{map = affine_map<(d0, d1) -> (d0 + d1)>}> : (index, index) -> index

%min = "affine.min"(%zero) <{"map" = affine_map<(d0) -> ((d0 + 41), d0)>}> : (index) -> index

%same_value = "affine.load"(%memref, %zero, %zero) <{"map" = affine_map<(d0, d1) -> (d0, d1)>}> : (memref<2x3xf64>, index, index) -> f64

"affine.if"() ({
  "affine.yield"() : () -> ()
}, {
}) {"condition" = affine_set<() : (0 == 0)>} : () -> ()
"affine.if"() ({
  "affine.yield"() : () -> ()
}, {
  "affine.yield"() : () -> ()
}) {"condition" = affine_set<() : (0 == 0)>} : () -> ()

// CHECK:       builtin.module {
// CHECK-NEXT:    %0, %1, %2, %3 = "test.op"() : () -> (index, f64, memref<2x3xf64>, index)
// CHECK-NEXT:    "affine.for"() <{lowerBoundMap = affine_map<()[] -> (0)>, operandSegmentSizes = array<i32: 0, 0, 0>, step = 1 : index, upperBoundMap = affine_map<()[] -> (256)>}> ({
// CHECK-NEXT:  ^bb0(%4: index):
// CHECK-NEXT:    "affine.yield"() : () -> ()
// CHECK-NEXT:  }) : () -> ()
// CHECK-NEXT:    "affine.parallel"(%0) <{lowerBoundsGroups = dense<1> : vector<1xi32>, lowerBoundsMap = affine_map<()[] -> (0)>, reductions = [], steps = [1], upperBoundsGroups = dense<1> : vector<1xi32>, upperBoundsMap = affine_map<()[s0] -> (s0)>}> ({
// CHECK-NEXT:  ^bb0(%4: index):
// CHECK-NEXT:    "affine.yield"() : () -> ()
// CHECK-NEXT:  }) : (index) -> ()
// CHECK-NEXT:    "affine.store"(%1, %2) <{map = affine_map<()[] -> (0, 0)>}> : (f64, memref<2x3xf64>) -> ()
// CHECK-NEXT:    %4 = "affine.apply"(%3, %3) <{map = affine_map<(d0, d1)[] -> (d0 + d1)>}> : (index, index) -> (index)
// CHECK-NEXT:    %5 = "affine.min"(%3) <{map = affine_map<(d0)[] -> (d0 + 41, d0)>}> : (index) -> (index)
// CHECK-NEXT:    %6 = "affine.load"(%2, %3, %3) <{map = affine_map<(d0, d1)[] -> (d0, d1)>}> : (memref<2x3xf64>, index, index) -> (f64)
// CHECK-NEXT:    "affine.if"() ({
// CHECK-NEXT:    "affine.yield"() : () -> ()
// CHECK-NEXT:  }, {

// CHECK:       }) {condition = affine_set<()[]: (0 == 0)>} : () -> ()
// CHECK-NEXT:    "affine.if"() ({
// CHECK-NEXT:    "affine.yield"() : () -> ()
// CHECK-NEXT:  }, {
// CHECK-NEXT:    "affine.yield"() : () -> ()
// CHECK-NEXT:  }) {condition = affine_set<()[]: (0 == 0)>} : () -> ()
// CHECK-NEXT:  }
