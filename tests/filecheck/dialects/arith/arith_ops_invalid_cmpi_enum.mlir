// RUN: scair-opt %s --verify-diagnostics | filecheck %s

%lhsi32, %rhsi32 = "test.op"() : () -> (i32, i32)
%cmpi = "arith.cmpi"(%lhsi32, %rhsi32) <{"predicate" = 14 : i64}> : (i32, i32) -> i1

// CHECK:       "builtin.module"() ({
// CHECK-NEXT:    %0, %1 = "test.op"() : () -> (i32, i32)
// CHECK-NEXT:    %2 = "arith.cmpi"(%0, %1) <{predicate = 14}> : (i32, i32) -> i1
// CHECK-NEXT:    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
// CHECK-NEXT:    > enum scair.dialects.arith.CmpIPredicate has no case with ordinal: 14
// CHECK-NEXT:  }) : () -> ()