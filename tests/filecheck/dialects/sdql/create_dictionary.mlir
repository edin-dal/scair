// RUN: scair-opt %s | filecheck %s
%k1 = "arith.constant"() <{value = 1 : i32}> : () -> i32

%v1 = "arith.constant"() <{value = 0.05 : f32}> : () -> f32
%k2 = "arith.constant"() <{value = 6 : i32}> : () -> i32
%v2 = "arith.constant"() <{value = 0.1 : f32}> : () -> f32

%res = sdql.create_dictionary %k1, %v1, %k2, %v2 : i32, f32, i32, f32 -> dictionary<i32, f32>

// CHECK: builtin.module {
// CHECK:   %0 = "arith.constant"() <{value = 1 : i32}> : () -> i32
// CHECK:   %1 = "arith.constant"() <{value = 0.05 : f32}> : () -> f32
// CHECK:   %2 = "arith.constant"() <{value = 6 : i32}> : () -> i32
// CHECK:   %3 = "arith.constant"() <{value = 0.1 : f32}> : () -> f32
// CHECK:   %4 = sdql.create_dictionary %0, %1, %2, %3 : i32, f32, i32, f32 -> dictionary<i32, f32>
// CHECK: }
