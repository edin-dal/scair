// RUN: scair-opt %s | filecheck %s

// sum (x in d) x.val

%0 = sdql.empty_dictionary : dictionary<i32, f16>

%1 = sdql.sum %0 : dictionary<i32, f16> -> f16 {
^bb0(%k: i32, %v: f16):
  sdql.yield %v : f16
}

// CHECK: builtin.module {
// CHECK:   %0 = sdql.empty_dictionary : dictionary<i32, f16>
// CHECK:   %1 = "sdql.sum"(%0) <{inType = dictionary<i32, f16>}> ({
// CHECK:   ^bb0(%2: i32, %3: f16):
// CHECK:     sdql.yield %3 : f16
// CHECK:   }) : (dictionary<i32, f16>) -> f16
// CHECK: }
