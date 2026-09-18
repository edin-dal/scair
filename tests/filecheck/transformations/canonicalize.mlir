// RUN: scair-opt %s -p canonicalize | filecheck %s

// Dialect-independent canonicalization patterns. Dialect-specific folds live
// beside their dialect, in tests/filecheck/dialects/<dialect>/canonicalize.mlir.

// CHECK:       builtin.module {

// An unused operation deriving its memory effects from its regions is dead when
// every operation nested in it is effect free.
func.func @dead_effect_free_loop(%lb: index, %ub: index, %step: index) {
  "scf.for"(%lb, %ub, %step) ({
  ^bb0(%iv: index):
    %c = "arith.constant"() <{value = 1 : i32}> : () -> i32
    "scf.yield"() : () -> ()
  }) : (index, index, index) -> ()
  func.return
}
// CHECK-NEXT:    func.func @dead_effect_free_loop(%0: index, %1: index, %2: index) {
// CHECK-NEXT:      func.return
// CHECK-NEXT:    }

// The same loop is kept once a nested operation has unknown effects, since the
// effects of the loop are then unknown too.
func.func @dead_loop_with_unknown_effects(%lb: index, %ub: index, %step: index) {
  "scf.for"(%lb, %ub, %step) ({
  ^bb0(%iv: index):
    "test.op"() : () -> ()
    "scf.yield"() : () -> ()
  }) : (index, index, index) -> ()
  func.return
}
// CHECK-NEXT:    func.func @dead_loop_with_unknown_effects(%0: index, %1: index, %2: index) {
// CHECK-NEXT:      "scf.for"(%0, %1, %2) ({
// CHECK-NEXT:      ^bb0(%3: index):
// CHECK-NEXT:        "test.op"() : () -> ()
// CHECK-NEXT:        scf.yield
// CHECK-NEXT:      }) : (index, index, index) -> ()
// CHECK-NEXT:      func.return
// CHECK-NEXT:    }

// Nesting: the outer loop is dead only because the inner one is.
func.func @dead_nested_effect_free_loops(%lb: index, %ub: index, %step: index) {
  "scf.for"(%lb, %ub, %step) ({
  ^bb0(%i: index):
    "scf.for"(%lb, %ub, %step) ({
    ^bb0(%j: index):
      "scf.yield"() : () -> ()
    }) : (index, index, index) -> ()
    "scf.yield"() : () -> ()
  }) : (index, index, index) -> ()
  func.return
}
// CHECK-NEXT:    func.func @dead_nested_effect_free_loops(%0: index, %1: index, %2: index) {
// CHECK-NEXT:      func.return
// CHECK-NEXT:    }
