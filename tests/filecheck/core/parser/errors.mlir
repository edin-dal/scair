// RUN: scair-opt %s --parsing-diagnostics | filecheck %s -DFILE=%s

%0 = "test.op"() : () -> i32
%2 = "test.op"() : () -> ()
%3 = "test.op"() : () -> i64

// CHECK:       Parse error at [[FILE]]:5:1:

// CHECK:       %3 = "test.op"() : () -> i64
// CHECK-NEXT:  ^
// CHECK-NEXT:  Number of results (1) does not match the number of the corresponding result types (0).
