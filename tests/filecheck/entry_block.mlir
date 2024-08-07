// RUN: scair-opt %s | filecheck %s

"test.op"() ({
  // No block header ;)
  "test.op"() : () -> ()
}) : () -> ()

// CHECK:      builtin.module {
// CHECK:      ^bb0():
// CHECK:        "test.op"() ({
// CHECK-NEXT:   ^bb1():
// CHECK-NEXT:     "test.op"() : () -> ()
// CHECK-NEXT:   }) : () -> ()