// RUN: scair-opt %s | filecheck %s

%0 = "test.op1"() {hello = "world", "quoted" = i3298} : () -> (i32)
"test.op2"() {hello = dense<10.0> : tensor<2xf32>} : () -> ()
"test.op3"() {hello = dense<[10.0, 11.0]> : tensor<2xf32>} : () -> ()
"test.op4"() {hello = dense<[10, 11]> : tensor<2xf32>} : () -> ()
"test.op5"() {hello = dense<11> : tensor<2xf32>} : () -> ()
"test.op7"(%0) : (i32) -> ()

// CHECK: builtin.module {
// CHECK: ^bb0():
// CHECK:   %0 = "test.op1"() {hello = "world", quoted = i3298} : () -> (i32)
// CHECK:   "test.op2"() {hello = dense<10.0> : tensor<2xf32>} : () -> ()
// CHECK:   "test.op3"() {hello = dense<[10.0, 11.0]> : tensor<2xf32>} : () -> ()
// CHECK:   "test.op4"() {hello = dense<[10, 11]> : tensor<2xf32>} : () -> ()
// CHECK:   "test.op5"() {hello = dense<11> : tensor<2xf32>} : () -> ()
// CHECK:   "test.op7"(%0) : (i32) -> ()