// RUN: scair-opt %s | filecheck %s

%0:2 = "test.op"() : () -> (i1, i1)

// CHECK: %0, %1 = "test.op"() : () -> (i1, i1)