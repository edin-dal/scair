// RUN: scair-opt %s | scair-opt | filecheck %s

%0, %1, %2 = "test.op"() : () -> (!cmath.complex<f32>, !cmath.complex<f32>, !cmath.complex<f32>)
%3 = cmath.norm (%0 : <f32>) => f64
%4 = cmath.mul (%1 : <f32>, %2 : <f32>) => <f64>

// CHECK: builtin.module {
// CHECK: ^bb0():
// CHECK:   %0, %1, %2 = "test.op"() : () -> (!cmath.complex<f32>, !cmath.complex<f32>, !cmath.complex<f32>)
// CHECK:   %3 = cmath.norm (%0 : <f32>) => f64
// CHECK:   %4 = cmath.mul (%1 : <f32>, %2 : <f32>) => <f64>
// CHECK: }
