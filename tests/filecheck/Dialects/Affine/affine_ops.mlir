// RUN: scair-opt %s -s | filecheck %s

"affine.for"() <{"lowerBoundMap" = affine_map<() -> (0)>, "upperBoundMap" = affine_map<() -> (256)>, "step" = 1 : index, "operandSegmentSizes" = array<i32: 0, 0, 0>}> ({
^0(%i : index):
  "affine.yield"() : () -> ()
}) : () -> ()

"affine.parallel"(%N) <{"lowerBoundsMap" = affine_map<() -> (0)>, "lowerBoundsGroups" = dense<1> : vector<1xi32>, "upperBoundsMap" = affine_map<()[s0] -> (s0)>, "upperBoundsGroups" = dense<1> : vector<1xi32>, "steps" = [1 : i64], "reductions" = []}> ({
^1(%i : index):
  "affine.yield"() : () -> ()
}) : (index) -> ()

"affine.store"(%value, %memref) <{"map" = affine_map<() -> (0, 0)>}> : (f64, memref<2x3xf64>) -> ()

%2 = affine.apply affine_map<(d0)[s0] -> (((d0 + (s0 * 42)) + -1))> (%zero)[%zero]

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