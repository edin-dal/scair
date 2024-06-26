// RUN: scair-opt %s | filecheck %s

"op1"() ({
^bb0(%0: i32): // Some comment at end of line
  %1, %2, %3 = "test.op"() : () -> (i32, i64, i32)
  // Some comment-only line, plus some extra whitspace there
  "test.op"(%2, %1) : (i64, i32) -> ()
^bb1(%4: i32):
  %5, %6, %7 = "test.op"()[^bb0] : () -> (i32, i64, i32)
  "test.op"(%6, %5) : (i64, i32) -> ()
}) : () -> ()

// CHECK: "op1"() ({
// CHECK: ^bb0(%0: i32):
// CHECK:   %1, %2, %3 = "test.op"() : () -> (i32, i64, i32)
// CHECK:   "test.op"(%2, %1) : (i64, i32) -> ()
// CHECK: ^bb1(%4: i32):
// CHECK:   %5, %6, %7 = "test.op"()[^bb0] : () -> (i32, i64, i32)
// CHECK:   "test.op"(%6, %5) : (i64, i32) -> ()
// CHECK: }) : () -> ()