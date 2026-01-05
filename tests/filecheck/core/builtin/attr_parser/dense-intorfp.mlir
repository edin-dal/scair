// RUN: scair-opt %s | filecheck %s

%0 = "test.op"() {hello = "world", "quoted" = i3298} : () -> i32
"test.op"() {hello = dense<10.0> : tensor<2xf32>} : () -> ()
"test.op"() {hello = dense<[10.0, 11.0]> : tensor<2xf32>} : () -> ()
"test.op"() {hello = dense<[10, 11]> : tensor<2xi32>} : () -> ()
"test.op"() {hello = dense<11> : tensor<2xi32>} : () -> ()
"test.op"(%0) : (i32) -> ()

// CHECK: builtin.module {
// CHECK:   %0 = "test.op"() {hello = "world", quoted = i3298} : () -> i32
// CHECK:   "test.op"() {hello = dense<10.0> : tensor<2xf32>} : () -> ()
// CHECK:   "test.op"() {hello = dense<[10.0, 11.0]> : tensor<2xf32>} : () -> ()
// CHECK:   "test.op"() {hello = dense<[10, 11]> : tensor<2xi32>} : () -> ()
// CHECK:   "test.op"() {hello = dense<11> : tensor<2xi32>} : () -> ()
// CHECK:   "test.op"(%0) : (i32) -> ()