// RUN: scair-opt %s --split-input-file | filecheck %s

// Definition before use, in the same block and in a successor block.
"test.op"() ({
^bb0:
  %0 = "test.op"() : () -> i32
  "test.op"(%0) : (i32) -> ()
  "test.op"()[^bb1] : () -> ()
^bb1:
  "test.op"(%0) : (i32) -> ()
}) : () -> ()

// CHECK:      builtin.module {
// CHECK-NEXT:   "test.op"() ({
// CHECK-NEXT:     %0 = "test.op"() : () -> i32
// CHECK-NEXT:     "test.op"(%0) : (i32) -> ()
// CHECK-NEXT:     "test.op"()[^bb0] : () -> ()
// CHECK-NEXT:   ^bb0:
// CHECK-NEXT:     "test.op"(%0) : (i32) -> ()
// CHECK-NEXT:   }) : () -> ()
// CHECK-NEXT: }

// -----

// Definition in the entry block, used after a join.
"test.op"() ({
^bb0:
  %0 = "test.op"() : () -> i32
  "test.op"()[^bb1, ^bb2] : () -> ()
^bb1:
  "test.op"()[^bb3] : () -> ()
^bb2:
  "test.op"()[^bb3] : () -> ()
^bb3:
  "test.op"(%0) : (i32) -> ()
}) : () -> ()

// CHECK:      builtin.module {
// CHECK-NEXT:   "test.op"() ({
// CHECK-NEXT:     %0 = "test.op"() : () -> i32
// CHECK-NEXT:     "test.op"()[^bb0, ^bb1] : () -> ()
// CHECK-NEXT:   ^bb0:
// CHECK-NEXT:     "test.op"()[^bb2] : () -> ()
// CHECK-NEXT:   ^bb1:
// CHECK-NEXT:     "test.op"()[^bb2] : () -> ()
// CHECK-NEXT:   ^bb2:
// CHECK-NEXT:     "test.op"(%0) : (i32) -> ()
// CHECK-NEXT:   }) : () -> ()
// CHECK-NEXT: }

// -----

// Definition in a loop header, used in the loop body.
"test.op"() ({
^bb0:
  "test.op"()[^bb1] : () -> ()
^bb1:
  %0 = "test.op"() : () -> i32
  "test.op"()[^bb2] : () -> ()
^bb2:
  "test.op"(%0) : (i32) -> ()
  "test.op"()[^bb1] : () -> ()
}) : () -> ()

// CHECK:      builtin.module {
// CHECK-NEXT:   "test.op"() ({
// CHECK-NEXT:     "test.op"()[^bb0] : () -> ()
// CHECK-NEXT:   ^bb0:
// CHECK-NEXT:     %0 = "test.op"() : () -> i32
// CHECK-NEXT:     "test.op"()[^bb1] : () -> ()
// CHECK-NEXT:   ^bb1:
// CHECK-NEXT:     "test.op"(%0) : (i32) -> ()
// CHECK-NEXT:     "test.op"()[^bb0] : () -> ()
// CHECK-NEXT:   }) : () -> ()
// CHECK-NEXT: }

// -----

// Definition in a parent scope, used two regions down.
%0 = "test.op"() : () -> i32
"test.op"() ({
  "test.op"() ({
    "test.op"(%0) : (i32) -> ()
  }) : () -> ()
}) : () -> ()

// CHECK:      builtin.module {
// CHECK-NEXT:   %0 = "test.op"() : () -> i32
// CHECK-NEXT:   "test.op"() ({
// CHECK-NEXT:     "test.op"() ({
// CHECK-NEXT:       "test.op"(%0) : (i32) -> ()
// CHECK-NEXT:     }) : () -> ()
// CHECK-NEXT:   }) : () -> ()
// CHECK-NEXT: }

// -----

// Definition in a parent region, used in a non-entry block of a nested region.
"test.op"() ({
^bb0:
  %0 = "test.op"() : () -> i32
  "test.op"()[^bb1] : () -> ()
^bb1:
  "test.op"() ({
  ^bb2:
    "test.op"()[^bb3] : () -> ()
  ^bb3:
    "test.op"(%0) : (i32) -> ()
  }) : () -> ()
}) : () -> ()

// CHECK:      builtin.module {
// CHECK-NEXT:   "test.op"() ({
// CHECK-NEXT:     %0 = "test.op"() : () -> i32
// CHECK-NEXT:     "test.op"()[^bb0] : () -> ()
// CHECK-NEXT:   ^bb0:
// CHECK-NEXT:     "test.op"() ({
// CHECK-NEXT:       "test.op"()[^bb1] : () -> ()
// CHECK-NEXT:     ^bb1:
// CHECK-NEXT:       "test.op"(%0) : (i32) -> ()
// CHECK-NEXT:     }) : () -> ()
// CHECK-NEXT:   }) : () -> ()
// CHECK-NEXT: }

// -----

// Block argument, used in its own block and in a nested region.
"test.op"() ({
^bb0(%0: i32):
  "test.op"(%0) : (i32) -> ()
  "test.op"() ({
    "test.op"(%0) : (i32) -> ()
  }) : () -> ()
}) : () -> ()

// CHECK:      builtin.module {
// CHECK-NEXT:   "test.op"() ({
// CHECK-NEXT:   ^bb0(%0: i32):
// CHECK-NEXT:     "test.op"(%0) : (i32) -> ()
// CHECK-NEXT:     "test.op"() ({
// CHECK-NEXT:       "test.op"(%0) : (i32) -> ()
// CHECK-NEXT:     }) : () -> ()
// CHECK-NEXT:   }) : () -> ()
// CHECK-NEXT: }
