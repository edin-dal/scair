// RUN: scair-opt %s -s | filecheck %s

%0, %1, %2 = "test.op"() : () -> (!cmath.complex<f32>, !cmath.complex<index>, !cmath.complex<f32>)
%3 = "cmath.norm"(%0) : (!cmath.complex<f32>) -> (f64)
%4, %5 = "cmath.mul"(%1, %2) : (!cmath.complex<index>, !cmath.complex<f32>) -> (f32, f32)

// CHECK:       builtin.module {
// CHECK-NEXT:    %0, %1, %2 = "test.op"() : () -> (!cmath.complex<[f32]>, !cmath.complex<[index]>, !cmath.complex<[f32]>)
// CHECK-NEXT:    %3 = "cmath.norm"(%0) : (!cmath.complex<[f32]>) -> (f64)
// CHECK-NEXT:    %4, %5 = "cmath.mul"(%1, %2) : (!cmath.complex<[index]>, !cmath.complex<[f32]>) -> (f32, f32)
// CHECK-NEXT:  }
