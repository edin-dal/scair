// RUN: scair-opt %s | filecheck %s

%0 = sdql.empty_dictionary : dictionary<i32, f16>

%letin = sdql.let_in %0 -> record<"field": dictionary<i32, f16>> {
^bb0(%x: dictionary<i32, f16>):
  %out = sdql.create_record {fields = ["field"]} %x : dictionary<i32, f16> -> record<"field": dictionary<i32, f16>>
  sdql.yield %out : record<"field": dictionary<i32, f16>>
}
// CHECK: builtin.module {
// CHECK:     %0 = sdql.empty_dictionary : dict<i32, f16>
// CHECK:     %1 = "sdql.let_in"(%0) ({
// CHECK:     ^bb0(%2: dict<i32, f16>):
// CHECK:       %3 = sdql.create_record {fields = ["field"]} %2 : dict<i32, f16> -> record<"field": dict<i32, f16>>
// CHECK:       sdql.yield %3 : record<"field": dict<i32, f16>>
// CHECK:     }) : (dict<i32, f16>) -> record<"field": dict<i32, f16>>
// CHECK: }