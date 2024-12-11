// RUN: scair-opt %s | filecheck %s

%0 = "test.op"() {hello = "world", "quoted" = i3298} : () -> (i32)
"test.op"() {hello = tensor<f32>} : () -> ()
"test.op"() {hello = tensor<1xf32>} : () -> ()
"test.op"() {hello = tensor<?xf32>} : () -> ()
"test.op"() {hello = tensor<3x?x5xf32>} : () -> ()
"test.op"() {hello = tensor<?x5x?xf32>} : () -> ()
"test.op"(%0) : (i32) -> ()

// CHECK: builtin.module {
// CHECK: ^bb0():
// CHECK:   %0 = "test.op"() {hello = "world", quoted = i3298} : () -> (i32)
// CHECK:   "test.op"() {hello = tensor<f32>} : () -> ()
// CHECK:   "test.op"() {hello = tensor<1xf32>} : () -> ()
// CHECK:   "test.op"() {hello = tensor<?xf32>} : () -> ()
// CHECK:   "test.op"() {hello = tensor<3x?x5xf32>} : () -> ()
// CHECK:   "test.op"() {hello = tensor<?x5x?xf32>} : () -> ()
// CHECK:   "test.op"(%0) : (i32) -> ()