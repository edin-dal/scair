// RUN: scair-opt %s | filecheck %s

%f = "test.op"() : () -> f32
%i = "test.op"() : () -> i32
%2 = math.absf %f : f32
%3 = math.fpowi %2, %i : f32, i32


// CHECK:      builtin.module {
// CHECK-NEXT:   %0 = "test.op"() : () -> f32
// CHECK-NEXT:   %1 = "test.op"() : () -> i32
// CHECK-NEXT:   %2 = "math.absf"(%0) <{fastmath = #arith.fastmath<none>}> : (f32) -> f32
// CHECK-NEXT:   %3 = "math.fpowi"(%2, %1) <{fastmath = #arith.fastmath<none>}> : (f32, i32) -> f32
// CHECK-NEXT: }