// RUN: scair-opt %s -g | filecheck %s

"op1"() ({
^bb0(%1 : f32, %3 : f32, %4 : f32):
  "test.op"() : () -> ()
  %0 = "cmath.norm"(%1) : (f32) -> (f64)
  %2 = "cmath.mul"(%3, %4) : (f32, f32) -> (f32)

// CHECK: "builtin.module"() ({
// CHECK: ^bb0():
// CHECK:   "op1"() ({
// CHECK:   ^bb1(%0: f32, %1: f32, %2: f32):
// CHECK:     "test.op"() : () -> ()
// CHECK:     %3 = "cmath.norm"(%0) : (f32) -> (f64)
// CHECK:     %4 = "cmath.mul"(%1, %2) : (f32, f32) -> (f32)

}) : () -> ()
