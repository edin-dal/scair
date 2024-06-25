// RUN: scair-opt %s | filecheck %s

"builtin.module"() ({
^bb0():
    %0, %1, %2 = "test.op"() : () -> (i32, i16, si23)
    "test.op"(%0) : (i32) -> ()
}) : () -> ()

// CHECK: "builtin.module"()({
// CHECK: ^bb0():
// CHECK:     %0 = "test.op"() : () -> (i32)
// CHECK:     "test.op"(%0) : (i32) -> ()
// CHECK: }) : () -> ()
