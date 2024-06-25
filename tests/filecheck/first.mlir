// RUN: scair-opt %s | filecheck %s

"builtin.module"() ({
^bb0():
    %0 = "test.op"() : () -> (i32)
    "test.op"(%0) : (i32) -> ()
}) : () -> ()

// CHECK: "builtin.module"() ({
// CHECK: ^bb0():
// CHECK:     %0 = "test.op"() : () -> (i32)
// CHECK:     "test.op"(%0) : (i32) -> ()
// CHECK: }) : () -> ()