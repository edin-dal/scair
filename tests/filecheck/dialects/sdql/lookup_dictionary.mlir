// RUN: scair-opt %s | filecheck %s

%dict = sdql.empty_dictionary : dictionary<i32, f16>

%key = "arith.constant"() <{value = 1 : i32}> : () -> i32

%val = sdql.lookup_dictionary %dict [%key : i32] : dictionary<i32, f16> -> f16

// CHECK: builtin.module {
// CHECK:     %0 = sdql.empty_dictionary : dictionary<i32, f16>
// CHECK:   %1 = "arith.constant"() <{value = 1 : i32}> : () -> i32
// CHECK:   %2 = sdql.lookup_dictionary %0[%1 : i32] : dictionary<i32, f16> -> f16
// CHECK: }