// RUN: scair-opt %s | filecheck %s

"test.op"() ({
  // No block header ;)
  "test.op"() : () -> ()
}) : () -> ()

// CHECK:      builtin.module {
// CHECK:        "test.op"() ({
// CHECK-NEXT:     "test.op"() : () -> ()
// CHECK-NEXT:   }) : () -> ()