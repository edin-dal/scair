// RUN: scair-opt %s | filecheck %s

%0 = "test.op"() : () -> (i32)
"test.op"(%0) : (i32) -> ()

// CHECK: builtin.module {
// CHECK:     %0 = "test.op"() : () -> (i32)
// CHECK:     "test.op"(%0) : (i32) -> ()
// CHECK: }