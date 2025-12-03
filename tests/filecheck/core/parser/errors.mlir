// RUN: scair-opt %s --parsing-diagnostics --split-input-file | filecheck %s -DSOURCE_FILE=%s

%1, %2, %3 = "test.op"() : () -> (i8, i16)

// CHECK:       Parse error at [[SOURCE_FILE]]:3:42:

// CHECK:       %1, %2, %3 = "test.op"() : () -> (i8, i16)
// CHECK-NEXT:                                           ^
// CHECK-NEXT:  Number of results (3) does not match the number of the corresponding result types (2).
