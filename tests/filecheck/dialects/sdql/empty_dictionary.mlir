// RUN: scair-opt %s | filecheck %s

%0 = sdql.empty_dictionary : dictionary<i32, f16>

// CHECK: builtin.module {
// CHECK:   %0 = sdql.empty_dictionary : dictionary<i32, f16>
// CHECK: }