// RUN: scair-opt %s | filecheck %s

// A standalone enum attribute, i.e., one that is an attribute in its own right
// rather than being backed by an integer one, spelled #dialect.enum<case>.

"test.op"() {"mode" = #test.rounding_mode<to_nearest_even>} : () -> ()
// CHECK:       "test.op"() {mode = #test.rounding_mode<to_nearest_even>} : () -> ()
"test.op"() {"mode" = #test.rounding_mode<downward>} : () -> ()
// CHECK-NEXT:  "test.op"() {mode = #test.rounding_mode<downward>} : () -> ()
"test.op"() {"mode" = #test.rounding_mode<upward>} : () -> ()
// CHECK-NEXT:  "test.op"() {mode = #test.rounding_mode<upward>} : () -> ()
"test.op"() {"mode" = #test.rounding_mode<toward_zero>} : () -> ()
// CHECK-NEXT:  "test.op"() {mode = #test.rounding_mode<toward_zero>} : () -> ()
"test.op"() {"mode" = #test.rounding_mode<to_nearest_away>} : () -> ()
// CHECK-NEXT:  "test.op"() {mode = #test.rounding_mode<to_nearest_away>} : () -> ()

// As a required operation property.
%0 = "test.rounding_mode_op"() <{"mode" = #test.rounding_mode<downward>}> : () -> i32
// CHECK-NEXT:  %0 = "test.rounding_mode_op"() <{mode = #test.rounding_mode<downward>}> : () -> i32

// With the optional one supplied as well.
%1 = "test.rounding_mode_op"() <{"mode" = #test.rounding_mode<upward>, "fallbackMode" = #test.rounding_mode<toward_zero>}> : () -> i32
// CHECK-NEXT:  %1 = "test.rounding_mode_op"() <{mode = #test.rounding_mode<upward>, fallbackMode = #test.rounding_mode<toward_zero>}> : () -> i32
