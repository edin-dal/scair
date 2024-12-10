// RUN: scair-opt %s | filecheck %s

"test.op"() {"1" = array<i32>, "2" = array<f64: 0.0, 2.0, 8.0>} : () -> ()
"test.op"() {"function_type" = (index, f32) -> f64} : () -> ()

// CHECK:  "test.op"() {1 = array<i32>, 2 = array<f64: 0.0, 2.0, 8.0>} : () -> ()
// CHECK:  "test.op"() {function_type = (index, f32) -> f64} : () -> ()
