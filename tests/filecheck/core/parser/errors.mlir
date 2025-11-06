// RUN: scair-opt %s --parsing-diagnostics | filecheck %s -DFILE=%s

%0 = "test.op"() : () -> i32
%2 = "test.op"() : () -> ()
%3 = "test.op"() : () -> i64

// CHECK:       Parse error at [[FILE]]:4:28:

// CHECK:       %2 = "test.op"() : () -> ()
// CHECK-NEXT:                             ^
// CHECK-NEXT:  Number of results (1) does not match the number of the corresponding result types (0) in "test.op".
