// RUN: { ! scair-opt %s; } 2>&1 | filecheck %s

// Definition in an unreachable block, used in a reachable one.
"test.op"() ({
^bb0:
  %0 = "test.op"() : () -> i32
  "test.op"(%0) : (i32) -> ()
  "test.op"()[^bb2] : () -> ()
^bb1:
  %1 = "test.op"() : () -> f32
  "test.op"()[^bb2] : () -> ()
^bb2:
  %2 = "builtin.unrealized_conversion_cast"(%1) : (f32) -> i32
}) : () -> ()

// CHECK: ssa-dominance: value Value(f32) does not dominate its use in op `builtin.unrealized_conversion_cast`
