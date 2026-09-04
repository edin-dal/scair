// RUN: scair-opt %s --parsing-diagnostics --split-input-file | filecheck %s -DSOURCE_FILE=%s

"test.op"() {"mode" = #test.rounding_mode<mauve>} : () -> ()

// CHECK:       Parse error at [[SOURCE_FILE]]:3:48:

// CHECK:       "test.op"() {"mode" = #test.rounding_mode<mauve>} : () -> ()
// CHECK-NEXT:                                                 ^
// CHECK-NEXT:  expected one of to_nearest_even, downward, upward, toward_zero, to_nearest_away for enum test.rounding_mode, but got 'mauve'

// -----

%0 = "test.rounding_mode_op"() <{"mode" = #test.rounding_mode<>}> : () -> i32

// CHECK:       Parse error at [[SOURCE_FILE]]:13:63:

// CHECK:       %0 = "test.rounding_mode_op"() <{"mode" = #test.rounding_mode<>}> : () -> i32
// CHECK-NEXT:                                                                ^
// CHECK-NEXT:  [a-zA-Z_]
