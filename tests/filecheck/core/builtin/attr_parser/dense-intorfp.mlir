// RUN: scair-opt %s | filecheck %s

%0 = "test.op"() {hello = "world", "quoted" = i3298} : () -> i32
"test.op"() {hello = dense<10.0> : tensor<2xf32>} : () -> ()
"test.op"() {hello = dense<[10.0, 11.0]> : tensor<2xf32>} : () -> ()
"test.op"() {hello = dense<[10, 11]> : tensor<2xi32>} : () -> ()
"test.op"() {hello = dense<11> : tensor<2xi32>} : () -> ()
"test.op"() {hello = dense<[10, 11]> : tensor<2xi64>} : () -> ()
"test.op"() {hello = dense<[10.0, 11.0]> : tensor<2xf64>} : () -> ()
"test.op"() {hello = dense<[true, false]> : tensor<2xi1>} : () -> ()
"test.op"() {hello = dense<[10, 11]> : tensor<2xindex>} : () -> ()
"test.op"() {hello = dense<[[1, 2], [3, 4]]> : tensor<2x2xi16>} : () -> ()
"test.op"() {hello = dense<> : tensor<0xi32>} : () -> ()
"test.op"() {hello = dense<42> : tensor<i32>} : () -> ()
"test.op"(%0) : (i32) -> ()

// CHECK: builtin.module {
// CHECK:   %0 = "test.op"() {hello = "world", quoted = i3298} : () -> i32
// CHECK:   "test.op"() {hello = dense<10.0> : tensor<2xf32>} : () -> ()
// CHECK:   "test.op"() {hello = dense<[10.0, 11.0]> : tensor<2xf32>} : () -> ()
// CHECK:   "test.op"() {hello = dense<[10, 11]> : tensor<2xi32>} : () -> ()
// CHECK:   "test.op"() {hello = dense<11> : tensor<2xi32>} : () -> ()
// CHECK:   "test.op"() {hello = dense<[10, 11]> : tensor<2xi64>} : () -> ()
// CHECK:   "test.op"() {hello = dense<[10.0, 11.0]> : tensor<2xf64>} : () -> ()
// CHECK:   "test.op"() {hello = dense<[true, false]> : tensor<2xi1>} : () -> ()
// CHECK:   "test.op"() {hello = dense<[10, 11]> : tensor<2xindex>} : () -> ()
// CHECK:   "test.op"() {hello = dense<[[1, 2], [3, 4]]> : tensor<2x2xi16>} : () -> ()
// CHECK:   "test.op"() {hello = dense<> : tensor<0xi32>} : () -> ()
// CHECK:   "test.op"() {hello = dense<42> : tensor<i32>} : () -> ()
// CHECK:   "test.op"(%0) : (i32) -> ()
