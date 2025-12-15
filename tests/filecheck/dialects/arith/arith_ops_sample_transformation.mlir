// RUN: scair-opt %s -p sample-constant-folding | scair-opt -p canonicalize | filecheck %s

%0 = "arith.constant"() <{value = 1}> : () -> i64
%1 = "arith.constant"() <{value = 1}> : () -> i64
%2 = "arith.addi"(%0, %1) : (i64, i64) -> i64
%3 = "test.op"(%2) : (i64) -> i64

// CHECK:      builtin.module {
// CHECK-NEXT:   %0 = "arith.constant"() <{value = 2}> : () -> i64
// CHECK-NEXT:   %1 = "test.op"(%0) : (i64) -> i64
// CHECK-NEXT: }