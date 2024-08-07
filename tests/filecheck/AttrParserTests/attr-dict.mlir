// RUN: scair-opt %s | filecheck %s

%0 = "test.op"() {hello = "world", "quoted" = i3298} : () -> (i32)
"test.op"(%0) : (i32) -> ()

// CHECK: builtin.module {
// CHECK: ^bb0():
// CHECK:   %0 = "test.op"() {hello = "world", quoted = i3298} : () -> (i32)
// CHECK:   "test.op"(%0) : (i32) -> ()