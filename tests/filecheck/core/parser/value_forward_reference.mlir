// RUN: scair-opt %s --skip-verify | filecheck %s

"test.op"() ({
  "test.op"(%0) : (i32) -> ()
  "test.op"()({
      "test.op"(%1) : (f32) -> ()
      %1 = "test.op"() : () -> f32
  }) : () -> ()
  %0 = "test.op"() : () -> i32
}) : () -> ()

// CHECK: builtin.module {
// CHECK:   "test.op"() ({
// CHECK:     "test.op"(%0) : (i32) -> ()
// CHECK:     "test.op"() ({
// CHECK:       "test.op"(%1) : (f32) -> ()
// CHECK:       %1 = "test.op"() : () -> f32
// CHECK:     }) : () -> ()
// CHECK:     %0 = "test.op"() : () -> i32
// CHECK:   }) : () -> ()
// CHECK: }
