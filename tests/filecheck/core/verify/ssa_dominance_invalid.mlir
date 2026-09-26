// RUN: scair-opt %s --verify-diagnostics | filecheck %s

"test.op"(%0) : (i32) -> ()
%0 = "test.op"() : () -> i32

// CHECK:         "test.op"(%0) : (i32) -> ()
// CHECK-NEXT:    ^^^^^^^^^^^^^^^^^^^^^^^^^^^
// CHECK-NEXT:    > ssa-dominance: operand #0 does not dominate this use
// CHECK-NEXT:    %0 = "test.op"() : () -> i32
// CHECK-NEXT:  }) : () -> ()
// CHECK-NOT:   >
