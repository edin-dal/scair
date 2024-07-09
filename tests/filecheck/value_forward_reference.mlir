// RUN: scair-opt %s | filecheck %s

"builtin.module"() ({
  "op1"() ({
  ^bb0():
    "test.op"(%0) : (i32) -> ()
    "test.op"()({
      ^bb1():
        "test.op"(%0) : (f32) -> ()
        %0 = "test.op"() : () -> (f32)
    }) : () -> ()
    %0 = "test.op"() : () -> (f32)
  }) : () -> ()

// CHECK: ^bb0():
// CHECK:   "op1"() ({
// CHECK:   ^bb1():
// CHECK:     "test.op"(%0) : (i32) -> ()
// CHECK:     "test.op"() ({
// CHECK:     ^bb2():
// CHECK:       "test.op"(%1) : (f32) -> ()
// CHECK:       %1 = "test.op"() : () -> (f32)
// CHECK:     }) : () -> ()
// CHECK:     %0 = "test.op"() : () -> (i32)
// CHECK:   }) : () -> ()
// CHECK: 

}) : () -> ()