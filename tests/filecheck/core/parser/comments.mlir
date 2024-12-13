// RUN: scair-opt %s | filecheck %s

"test.op"() ({
^bb0(%0: i32): // Some comment at end of line
  %1, %2, %3 = "test.op"() : () -> (i32, i64, i32)
  // Some comment-only line, plus some extra whitspace there
  "test.op"(%2, %1) : (i64, i32) -> ()
^bb1(%4: i32):
  %5, %6, %7 = "test.op"() : () -> (i32, i64, i32)
  "test.op"(%6, %5) : (i64, i32) -> ()
}) : () -> ()

// CHECK:      builtin.module {
// CHECK:        "test.op"() ({
// CHECK-NEXT:   ^bb0(%0: i32):
// CHECK-NEXT:     %1, %2, %3 = "test.op"() : () -> (i32, i64, i32)
// CHECK-NEXT:     "test.op"(%2, %1) : (i64, i32) -> ()
// CHECK-NEXT:   ^bb1(%4: i32):
// CHECK-NEXT:     %5, %6, %7 = "test.op"() : () -> (i32, i64, i32)
// CHECK-NEXT:     "test.op"(%6, %5) : (i64, i32) -> ()
// CHECK-NEXT:   }) : () -> ()