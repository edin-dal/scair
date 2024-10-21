// RUN: scair-opt %s -p dummy-pass -g -s | filecheck %s


%0, %1, %2 = "test.op"() ({
^bb0(%1 : f32, %3 : f32, %4 : f32):
  "test.op"() ({
    ^bb1(%444 : f32, %555 : f32, %6666 : f32):
        "test.op"()  : () -> ()
        %0 = "cmath.norm"(%1) : (f32) -> (f64)
        %2 = "cmath.mul"(%3, %4) : (f32, f32) -> (f32)
    }) {attr = "this is it"} : () -> ()
  %0 = "cmath.norm"(%1) : (f32) -> (f64)
  %2 = "cmath.mul"(%3, %4) : (f32, f32) -> (f32)
  }) : () -> (!cmath.complex<f32>, !cmath.complex<index>, !cmath.complex<f32>)
%3 = "cmath.norm"(%0) : (!cmath.complex<f32>) -> (f64)
%4, %5 = "cmath.mul"(%1, %2) : (!cmath.complex<index>, !cmath.complex<f32>) -> (f32, f32)

// CHECK:       "builtin.module"() ({
// CHECK-NEXT:  ^bb0():
// CHECK-NEXT:    %0, %1, %2 = "test.op"() ({
// CHECK-NEXT:    ^bb1(%3: f32, %4: f32, %5: f32):
// CHECK-NEXT:      "test.op"() ({
// CHECK-NEXT:      ^bb2(%6: f32, %7: f32, %8: f32):
// CHECK-NEXT:        "test.op"() {dummy = "UnregDumDum"} : () -> ()
// CHECK-NEXT:        %9 = "cmath.norm"(%3) {dummy = "NormDumDum"} : (f32) -> (f64)
// CHECK-NEXT:        %10 = "cmath.mul"(%4, %5) {dummy = "MulDumDum"} : (f32, f32) -> (f32)
// CHECK-NEXT:      }) {attr = "this is it", dummy = "UnregDumDum"} : () -> ()
// CHECK-NEXT:      %11 = "cmath.norm"(%3) {dummy = "NormDumDum"} : (f32) -> (f64)
// CHECK-NEXT:      %12 = "cmath.mul"(%4, %5) {dummy = "MulDumDum"} : (f32, f32) -> (f32)
// CHECK-NEXT:    }) {dummy = "UnregDumDum"} : () -> (!cmath.complex<f32>, !cmath.complex<index>, !cmath.complex<f32>)
// CHECK-NEXT:    %13 = "cmath.norm"(%0) {dummy = "NormDumDum"} : (!cmath.complex<f32>) -> (f64)
// CHECK-NEXT:    %14, %15 = "cmath.mul"(%1, %2) {dummy = "MulDumDum"} : (!cmath.complex<index>, !cmath.complex<f32>) -> (f32, f32)
