// RUN: scair-opt %s | filecheck %s

%0 = "arith.constant"() <{value = 1 : i32}> : () -> i32

%1 = "arith.constant"() <{value = 0.05 : f32}> : () -> f32

%rec = sdql.create_record {fields = ["a", "b"]} %0, %1 : i32, f32 -> record<"a": i32, "b": f32>

%val = sdql.access_record %rec "a" : record<"a": i32, "b": f32> -> i32

// CHECK: builtin.module {
// CHECK:   %0 = "arith.constant"() <{value = 1 : i32}> : () -> i32
// CHECK:   %1 = "arith.constant"() <{value = 0.05 : f32}> : () -> f32
// CHECK:   %2 = sdql.create_record {fields = ["a", "b"]} %0, %1 : i32, f32 -> record<"a": i32, "b": f32>
// CHECK:   %3 = sdql.access_record %2 "a" : record<"a": i32, "b": f32> -> i32
// CHECK: }
