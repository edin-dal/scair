// RUN: { ! scair-opt %s; } 2>&1 | filecheck %s

// Definition on one branch only, used after the join.
"test.op"() ({
^bb0:
  %0 = "test.op"() : () -> i32
  "test.op"()[^bb1, ^bb2] : () -> ()
^bb1:
  %1 = "test.op"() : () -> f32
  "test.op"(%0, %1) : (i32, f32) -> ()
  "test.op"()[^bb3] : () -> ()
^bb2:
  "test.op"(%0) : (i32) -> ()
  "test.op"()[^bb3] : () -> ()
^bb3:
  "test.op"(%0) : (i32) -> ()
  %2 = "builtin.unrealized_conversion_cast"(%0, %1) : (i32, f32) -> i32
}) : () -> ()

// CHECK: ssa-dominance: value Value(f32) does not dominate its use in op `builtin.unrealized_conversion_cast`
