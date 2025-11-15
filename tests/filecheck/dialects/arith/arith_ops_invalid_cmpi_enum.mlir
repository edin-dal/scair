// RUN: scair-opt %s --verify-diagnostics | filecheck %s

%lhsi32, %rhsi32 = "test.op"() : () -> (i32, i32)
%cmpi = "arith.cmpi"(%lhsi32, %rhsi32) <{"predicate" = 14 : i64}> : (i32, i32) -> i1
// CHECK: java.util.NoSuchElementException: enum scair.dialects.arith.CmpIPredicate has no case with ordinal: 14