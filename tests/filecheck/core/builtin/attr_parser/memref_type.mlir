// RUN: scair-opt %s | filecheck %s

%0 = "test.op"() {hello = "world", "quoted" = i3298} : () -> i32
"test.op"() {hello = memref<f32>} : () -> ()
"test.op"() {hello = memref<1xf32>} : () -> ()
"test.op"() {hello = memref<?xf32>} : () -> ()
"test.op"() {hello = memref<3x?x5xf32>} : () -> ()
"test.op"() {hello = memref<?x5x?xf32>} : () -> ()
"test.op"(%0) : (i32) -> ()

// CHECK: builtin.module {
// CHECK:   %0 = "test.op"() {hello = "world", quoted = i3298} : () -> i32
// CHECK:   "test.op"() {hello = memref<f32>} : () -> ()
// CHECK:   "test.op"() {hello = memref<1xf32>} : () -> ()
// CHECK:   "test.op"() {hello = memref<?xf32>} : () -> ()
// CHECK:   "test.op"() {hello = memref<3x?x5xf32>} : () -> ()
// CHECK:   "test.op"() {hello = memref<?x5x?xf32>} : () -> ()
// CHECK:   "test.op"(%0) : (i32) -> ()