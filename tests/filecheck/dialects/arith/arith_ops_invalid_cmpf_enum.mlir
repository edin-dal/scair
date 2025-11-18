// RUN: scair-opt %s --verify-diagnostics | filecheck %s

%lhsf32, %rhsf32 = "test.op"() : () -> (f32, f32)
%cmpf = "arith.cmpf"(%lhsf32, %rhsf32) <{"fastmath" = #arith.fastmath<none>, "predicate" = 19 : i64}> : (f32, f32) -> i1
// CHECK: java.util.NoSuchElementException: enum scair.dialects.arith.CmpFPredicate has no case with ordinal: 19