// RUN: scair-opt %s | filecheck %s

// sum (x in d) x.val

%0 = sdql.empty_dictionary : dictionary<i32, f16>

%1 = sdql.sum %0 : dictionary<i32, f16> -> f16 {
^bb0(%x: record<"key": i32, "value": f16>):
  %tmp = sdql.access_record %x "value" : record<"key": i32, "value": f16> -> f16
  sdql.yield %tmp : f16
}

// CHECK: builtin.module {
// CHECK:     %0 = sdql.empty_dictionary : dict<i32, f16>
// CHECK:     %1 = "sdql.let_in"(%0) ({
// CHECK:     ^bb0(%2: dict<i32, f16>):
// CHECK:       %3 = sdql.create_record {fields = ["field"]} %2 : dict<i32, f16> -> record<"field": dict<i32, f16>>
// CHECK:       sdql.yield %3 : record<"field": dict<i32, f16>>
// CHECK:     }) : (dict<i32, f16>) -> record<"field": dict<i32, f16>>
// CHECK: }
