// RUN: scair-opt %s --split-input-file --verify-diagnostics | filecheck %s

// Use before definition in the same block.
func.func @use_before_def() {
  "test.op"(%0) : (i32) -> ()
  %0 = "test.op"() : () -> i32
  "func.return"() : () -> ()
}

// CHECK:      "builtin.module"() ({
// CHECK-NEXT:   "func.func"() <{sym_name = "use_before_def", function_type = () -> ()}> ({
// CHECK-NEXT:     "test.op"(%0) : (i32) -> ()
// CHECK-NEXT:     ^^^^^^^^^^^^^^^^^^^^^^^^^^^
// CHECK-NEXT:     > ssa-dominance: operand #0 does not dominate this use
// CHECK-NEXT:     %0 = "test.op"() : () -> i32
// CHECK-NEXT:     "func.return"() : () -> ()
// CHECK-NEXT:   }) : () -> ()
// CHECK-NEXT: }) : () -> ()

// -----

// Only the operand defined too late is reported, not the one before it.
func.func @late_second_operand() {
  %0 = "test.op"() : () -> i32
  "test.op"(%0, %1) : (i32, i32) -> ()
  %1 = "test.op"() : () -> i32
  "func.return"() : () -> ()
}

// CHECK:      "builtin.module"() ({
// CHECK-NEXT:   "func.func"() <{sym_name = "late_second_operand", function_type = () -> ()}> ({
// CHECK-NEXT:     %0 = "test.op"() : () -> i32
// CHECK-NEXT:     "test.op"(%0, %1) : (i32, i32) -> ()
// CHECK-NEXT:     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
// CHECK-NEXT:     > ssa-dominance: operand #1 does not dominate this use
// CHECK-NEXT:     %1 = "test.op"() : () -> i32
// CHECK-NEXT:     "func.return"() : () -> ()
// CHECK-NEXT:   }) : () -> ()
// CHECK-NEXT: }) : () -> ()

// -----

// Use in the entry block, definition in its successor.
func.func @use_in_predecessor() {
  %0 = "test.op"() : () -> i32
  "test.op"(%0, %1) : (i32, i32) -> ()
  "test.op"()[^bb0] : () -> ()
^bb0:
  %1 = "test.op"() : () -> i32
  "func.return"() : () -> ()
}

// CHECK:      "builtin.module"() ({
// CHECK-NEXT:   "func.func"() <{sym_name = "use_in_predecessor", function_type = () -> ()}> ({
// CHECK-NEXT:     %0 = "test.op"() : () -> i32
// CHECK-NEXT:     "test.op"(%0, %1) : (i32, i32) -> ()
// CHECK-NEXT:     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
// CHECK-NEXT:     > ssa-dominance: operand #1 does not dominate this use
// CHECK-NEXT:     "test.op"()[^bb0] : () -> ()
// CHECK-NEXT:   ^bb0:
// CHECK-NEXT:     %1 = "test.op"() : () -> i32
// CHECK-NEXT:     "func.return"() : () -> ()
// CHECK-NEXT:   }) : () -> ()
// CHECK-NEXT: }) : () -> ()

// -----

// Definition on one branch only, used after the join.
func.func @join_bypass() {
  %0 = "test.op"() : () -> i32
  "test.op"()[^bb0, ^bb1] : () -> ()
^bb0:
  %1 = "test.op"() : () -> i32
  "test.op"(%0, %1) : (i32, i32) -> ()
  "test.op"()[^bb2] : () -> ()
^bb1:
  "test.op"(%0) : (i32) -> ()
  "test.op"()[^bb2] : () -> ()
^bb2:
  "test.op"(%0) : (i32) -> ()
  "test.op"(%0, %1) : (i32, i32) -> ()
  "func.return"() : () -> ()
}

// CHECK:      "builtin.module"() ({
// CHECK-NEXT:   "func.func"() <{sym_name = "join_bypass", function_type = () -> ()}> ({
// CHECK-NEXT:     %0 = "test.op"() : () -> i32
// CHECK-NEXT:     "test.op"()[^bb0, ^bb1] : () -> ()
// CHECK-NEXT:   ^bb0:
// CHECK-NEXT:     %1 = "test.op"() : () -> i32
// CHECK-NEXT:     "test.op"(%0, %1) : (i32, i32) -> ()
// CHECK-NEXT:     "test.op"()[^bb2] : () -> ()
// CHECK-NEXT:   ^bb1:
// CHECK-NEXT:     "test.op"(%0) : (i32) -> ()
// CHECK-NEXT:     "test.op"()[^bb2] : () -> ()
// CHECK-NEXT:   ^bb2:
// CHECK-NEXT:     "test.op"(%0) : (i32) -> ()
// CHECK-NEXT:     "test.op"(%0, %1) : (i32, i32) -> ()
// CHECK-NEXT:     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
// CHECK-NEXT:     > ssa-dominance: operand #1 does not dominate this use
// CHECK-NEXT:     "func.return"() : () -> ()
// CHECK-NEXT:   }) : () -> ()
// CHECK-NEXT: }) : () -> ()

// -----

// Definition in an unreachable block, used in a reachable one.
func.func @unreachable_def() {
  %0 = "test.op"() : () -> i32
  "test.op"(%0) : (i32) -> ()
  "test.op"()[^bb1] : () -> ()
^bb0:
  %1 = "test.op"() : () -> i32
  "test.op"()[^bb1] : () -> ()
^bb1:
  "test.op"(%1) : (i32) -> ()
  "func.return"() : () -> ()
}

// CHECK:      "builtin.module"() ({
// CHECK-NEXT:   "func.func"() <{sym_name = "unreachable_def", function_type = () -> ()}> ({
// CHECK-NEXT:     %0 = "test.op"() : () -> i32
// CHECK-NEXT:     "test.op"(%0) : (i32) -> ()
// CHECK-NEXT:     "test.op"()[^bb0] : () -> ()
// CHECK-NEXT:   ^bb1:
// CHECK-NEXT:     %1 = "test.op"() : () -> i32
// CHECK-NEXT:     "test.op"()[^bb0] : () -> ()
// CHECK-NEXT:   ^bb0:
// CHECK-NEXT:     "test.op"(%1) : (i32) -> ()
// CHECK-NEXT:     ^^^^^^^^^^^^^^^^^^^^^^^^^^^
// CHECK-NEXT:     > ssa-dominance: operand #0 does not dominate this use
// CHECK-NEXT:     "func.return"() : () -> ()
// CHECK-NEXT:   }) : () -> ()
// CHECK-NEXT: }) : () -> ()

// -----

// Block argument on one branch only, used after the join.
func.func @block_arg_join_bypass() {
  "test.op"()[^bb1, ^bb2] : () -> ()
^bb1(%0: i32):
  "test.op"()[^bb3] : () -> ()
^bb2:
  "test.op"()[^bb3] : () -> ()
^bb3:
  "test.op"(%0) : (i32) -> ()
  "func.return"() : () -> ()
}

// CHECK:      "builtin.module"() ({
// CHECK-NEXT:   "func.func"() <{sym_name = "block_arg_join_bypass", function_type = () -> ()}> ({
// CHECK-NEXT:     "test.op"()[^bb0, ^bb1] : () -> ()
// CHECK-NEXT:   ^bb0(%0: i32):
// CHECK-NEXT:     "test.op"()[^bb2] : () -> ()
// CHECK-NEXT:   ^bb1:
// CHECK-NEXT:     "test.op"()[^bb2] : () -> ()
// CHECK-NEXT:   ^bb2:
// CHECK-NEXT:     "test.op"(%0) : (i32) -> ()
// CHECK-NEXT:     ^^^^^^^^^^^^^^^^^^^^^^^^^^^
// CHECK-NEXT:     > ssa-dominance: operand #0 does not dominate this use
// CHECK-NEXT:     "func.return"() : () -> ()
// CHECK-NEXT:   }) : () -> ()
// CHECK-NEXT: }) : () -> ()

// -----

// Block argument used in a predecessor of its block.
func.func @block_arg_use_in_predecessor() {
  "test.op"(%0) : (i32) -> ()
  "test.op"()[^bb1] : () -> ()
^bb1(%0: i32):
  "func.return"() : () -> ()
}

// CHECK:      "builtin.module"() ({
// CHECK-NEXT:   "func.func"() <{sym_name = "block_arg_use_in_predecessor", function_type = () -> ()}> ({
// CHECK-NEXT:     "test.op"(%0) : (i32) -> ()
// CHECK-NEXT:     ^^^^^^^^^^^^^^^^^^^^^^^^^^^
// CHECK-NEXT:     > ssa-dominance: operand #0 does not dominate this use
// CHECK-NEXT:     "test.op"()[^bb0] : () -> ()
// CHECK-NEXT:   ^bb0(%0: i32):
// CHECK-NEXT:     "func.return"() : () -> ()
// CHECK-NEXT:   }) : () -> ()
// CHECK-NEXT: }) : () -> ()

// -----

// Block argument used in a sibling branch.
func.func @block_arg_sibling() {
  "test.op"()[^bb1, ^bb2] : () -> ()
^bb1(%0: i32):
  "func.return"() : () -> ()
^bb2:
  "test.op"(%0) : (i32) -> ()
  "func.return"() : () -> ()
}

// CHECK:      "builtin.module"() ({
// CHECK-NEXT:   "func.func"() <{sym_name = "block_arg_sibling", function_type = () -> ()}> ({
// CHECK-NEXT:     "test.op"()[^bb0, ^bb1] : () -> ()
// CHECK-NEXT:   ^bb0(%0: i32):
// CHECK-NEXT:     "func.return"() : () -> ()
// CHECK-NEXT:   ^bb1:
// CHECK-NEXT:     "test.op"(%0) : (i32) -> ()
// CHECK-NEXT:     ^^^^^^^^^^^^^^^^^^^^^^^^^^^
// CHECK-NEXT:     > ssa-dominance: operand #0 does not dominate this use
// CHECK-NEXT:     "func.return"() : () -> ()
// CHECK-NEXT:   }) : () -> ()
// CHECK-NEXT: }) : () -> ()
