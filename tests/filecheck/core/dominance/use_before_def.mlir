// RUN: { ! scair-opt %s; } 2>&1 | filecheck %s

// Use before definition in the same block.
%0 = "test.op"() : () -> i32
"test.op"(%0) : (i32) -> ()
%1 = "builtin.unrealized_conversion_cast"(%0, %2) : (i32, f32) -> i32
%2 = "test.op"() : () -> f32

// CHECK: ssa-dominance: value Value(f32) does not dominate its use in op `builtin.unrealized_conversion_cast`
