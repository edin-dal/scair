// RUN: scair-opt %s -g | filecheck %s

"test.op"() ({
^bb0(%1 : !cmath.complex<f32>, %3 : !cmath.complex<f32>, %4 : !cmath.complex<f32>):
  "test.op"() : () -> ()
  %0 = "cmath.norm"(%1) : (!cmath.complex<f32>) -> (f32)
  %2 = "cmath.mul"(%3, %4) : (!cmath.complex<f32>, !cmath.complex<f32>) -> (!cmath.complex<f32>)

// CHECK: "builtin.module"() ({
// CHECK:   "test.op"() ({
// CHECK:   ^bb0(%0: !cmath.complex<f32>, %1: !cmath.complex<f32>, %2: !cmath.complex<f32>):
// CHECK:     "test.op"() : () -> ()
// CHECK:     %3 = "cmath.norm"(%0) : (!cmath.complex<f32>) -> (f32)
// CHECK:     %4 = "cmath.mul"(%1, %2) : (!cmath.complex<f32>, !cmath.complex<f32>) -> (!cmath.complex<f32>)

}) : () -> ()
