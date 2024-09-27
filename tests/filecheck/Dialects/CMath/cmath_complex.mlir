// RUN: scair-opt %s -g | filecheck %s

%0, %1, %2 = "test.op"() : () -> (!cmath.complex<f32>, !cmath.complex<index>, !cmath.complex<f32>)
%3 = "cmath.norm"(%0) : (!cmath.complex<f32>) -> (f64)
%4, %5 = "cmath.mul"(%1, %2) : (!cmath.complex<index>, !cmath.complex<f32>) -> (f32, f32)

// CHECK: "builtin.module"() ({
// CHECK: ^bb0():
// CHECK:   %0, %1, %2 = "test.op"() : () -> (!cmath.complex<f32>, !cmath.complex<index>, !cmath.complex<f32>)
// CHECK:   %3 = "cmath.norm"(%0) : (!cmath.complex<f32>) -> (f64)
// CHECK:   %4, %5 = "cmath.mul"(%1, %2) : (!cmath.complex<index>, !cmath.complex<f32>) -> (f32, f32)
