// RUN: scair-opt %s | filecheck %s

%0 = "test.op1"() {hello = "world", "quoted" = i3298} : () -> (i32)
"test.op2"() {hello = tensor<f32>} : () -> ()
"test.op3"() {hello = tensor<1xf32>} : () -> ()
"test.op4"() {hello = tensor<?xf32>} : () -> ()
"test.op5"() {hello = tensor<3x?x5xf32>} : () -> ()
"test.op6"() {hello = tensor<?x5x?xf32>} : () -> ()
"test.op7"(%0) : (i32) -> ()

// CHECK: builtin.module {
// CHECK: ^bb0():
// CHECK:   %0 = "test.op1"() {hello = "world", quoted = i3298} : () -> (i32)
// CHECK:   "test.op2"() {hello = tensor<f32>} : () -> ()
// CHECK:   "test.op3"() {hello = tensor<1xf32>} : () -> ()
// CHECK:   "test.op4"() {hello = tensor<?xf32>} : () -> ()
// CHECK:   "test.op5"() {hello = tensor<3x?x5xf32>} : () -> ()
// CHECK:   "test.op6"() {hello = tensor<?x5x?xf32>} : () -> ()
// CHECK:   "test.op7"(%0) : (i32) -> ()