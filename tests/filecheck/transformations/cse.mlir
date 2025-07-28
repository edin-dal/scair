// RUN: scair-opt %s -p cse | filecheck %s

%0 = "arith.constant"() <{value = 0 : i32}> : () -> i32
%1 = "arith.constant"() <{value = 1 : i32}> : () -> i32
%2 = "arith.constant"() <{value = 0 : i32}> : () -> i32

// CHECK:       builtin.module {
// CHECK-NEXT:    %0 = "arith.constant"() <{value = 0 : i32}> : () -> (i32)
// CHECK-NEXT:    %1 = "arith.constant"() <{value = 1 : i32}> : () -> (i32)
// CHECK-NEXT:  }