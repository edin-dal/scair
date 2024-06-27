// RUN: scair-opt %s | filecheck %s

"test.op"() ({
  // No block header ;)
  "test.op"() : () -> ()
}) : () -> ()

// CHECK:      "test.op"() ({
// CHECK-NEXT: ^bb0():
// CHECK-NEXT:   "test.op"() : () -> ()
// CHECK-NEXT: }) : () -> ()