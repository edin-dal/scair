// RUN: scair-opt %s --split-input-file -p licm | filecheck %s

// Ported from mlir/test/Transforms/loop-invariant-code-motion.mlir, keeping
// upstream's case names, ordering and CHECK style so the two can be diffed.
// The IR is spelled in generic form, as ScaIR has no custom syntax for most of
// scf and affine yet, and CHECK lines carrying operand syntax are adjusted to
// match; CHECK lines that only name an operation are used verbatim.
//
// Not ported, for want of the dialects or analyses they exercise:
//   - tensor, linalg and vector cases,
//   - `test.graph_loop` cases (graph regions),
//   - `tensor.dim`/`memref.dim` and integer division speculation, which need
//     `ConditionallySpeculatable` implementations backed by integer range
//     analysis. See the TODOs in dialects/memref and dialects/arith.

func.func @nested_loops_both_having_invariant_code() {
  %m = "memref.alloc"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<10xf32>
  %cf7 = "arith.constant"() <{value = 7.0 : f32}> : () -> f32
  %cf8 = "arith.constant"() <{value = 8.0 : f32}> : () -> f32
  %ci0 = "arith.constant"() <{value = 0 : index}> : () -> index
  %ci10 = "arith.constant"() <{value = 10 : index}> : () -> index
  %ci1 = "arith.constant"() <{value = 1 : index}> : () -> index

  "scf.for"(%ci0, %ci10, %ci1) ({
  ^bb0(%arg0: index):
    %v0 = "arith.addf"(%cf7, %cf8) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
    "scf.for"(%ci0, %ci10, %ci1) ({
    ^bb0(%arg1: index):
      %v1 = "arith.addf"(%cf7, %cf7) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      "scf.yield"() : () -> ()
    }) : (index, index, index) -> ()
    "scf.yield"() : () -> ()
  }) : (index, index, index) -> ()

  // CHECK: memref.alloc
  // CHECK-NEXT: arith.constant
  // CHECK-NEXT: arith.constant
  // CHECK-NEXT: arith.constant
  // CHECK-NEXT: arith.constant
  // CHECK-NEXT: arith.constant
  // CHECK-NEXT: arith.addf
  // CHECK-NEXT: arith.addf
  // CHECK-NEXT: scf.for
  // CHECK-NEXT: ^
  // CHECK-NEXT: scf.yield
  // CHECK-NEXT: }
  // CHECK-NEXT: scf.for

  func.return
}

// -----

func.func @nested_loops_code_invariant_to_both() {
  %m = "memref.alloc"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<10xf32>
  %cf7 = "arith.constant"() <{value = 7.0 : f32}> : () -> f32
  %cf8 = "arith.constant"() <{value = 8.0 : f32}> : () -> f32

  "affine.for"() <{lowerBoundMap = affine_map<() -> (0)>, upperBoundMap = affine_map<() -> (10)>, step = 1 : index, operandSegmentSizes = array<i32: 0, 0, 0>}> ({
  ^bb0(%arg0: index):
    "affine.for"() <{lowerBoundMap = affine_map<() -> (0)>, upperBoundMap = affine_map<() -> (10)>, step = 1 : index, operandSegmentSizes = array<i32: 0, 0, 0>}> ({
    ^bb0(%arg1: index):
      %v0 = "arith.addf"(%cf7, %cf8) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      "affine.yield"() : () -> ()
    }) : () -> ()
    "affine.yield"() : () -> ()
  }) : () -> ()

  // CHECK: memref.alloc
  // CHECK-NEXT: arith.constant
  // CHECK-NEXT: arith.constant
  // CHECK-NEXT: arith.addf

  func.return
}

// -----

func.func @single_loop_nothing_invariant() {
  %m1 = "memref.alloc"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<10xf32>
  %m2 = "memref.alloc"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<10xf32>

  "affine.for"() <{lowerBoundMap = affine_map<() -> (0)>, upperBoundMap = affine_map<() -> (10)>, step = 1 : index, operandSegmentSizes = array<i32: 0, 0, 0>}> ({
  ^bb0(%arg0: index):
    %v0 = "affine.load"(%m1, %arg0) <{map = affine_map<(d0) -> (d0)>}> : (memref<10xf32>, index) -> f32
    %v1 = "affine.load"(%m2, %arg0) <{map = affine_map<(d0) -> (d0)>}> : (memref<10xf32>, index) -> f32
    %v2 = "arith.addf"(%v0, %v1) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
    "affine.store"(%v2, %m1, %arg0) <{map = affine_map<(d0) -> (d0)>}> : (f32, memref<10xf32>, index) -> ()
    "affine.yield"() : () -> ()
  }) : () -> ()

  // CHECK: memref.alloc
  // CHECK-NEXT: memref.alloc
  // CHECK-NEXT: affine.for
  // CHECK-NEXT: ^
  // CHECK-NEXT: affine.load
  // CHECK-NEXT: affine.load
  // CHECK-NEXT: arith.addf
  // CHECK-NEXT: affine.store

  func.return
}

// -----

func.func @invariant_loop_dialect() {
  %ci0 = "arith.constant"() <{value = 0 : index}> : () -> index
  %ci10 = "arith.constant"() <{value = 10 : index}> : () -> index
  %ci1 = "arith.constant"() <{value = 1 : index}> : () -> index
  %m = "memref.alloc"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<10xf32>
  %cf7 = "arith.constant"() <{value = 7.0 : f32}> : () -> f32
  %cf8 = "arith.constant"() <{value = 8.0 : f32}> : () -> f32

  "scf.for"(%ci0, %ci10, %ci1) ({
  ^bb0(%arg0: index):
    "scf.for"(%ci0, %ci10, %ci1) ({
    ^bb0(%arg1: index):
      %v0 = "arith.addf"(%cf7, %cf8) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      "scf.yield"() : () -> ()
    }) : (index, index, index) -> ()
    "scf.yield"() : () -> ()
  }) : (index, index, index) -> ()

  // CHECK: memref.alloc
  // CHECK-NEXT: arith.constant
  // CHECK-NEXT: arith.constant
  // CHECK-NEXT: arith.addf

  func.return
}

// -----

func.func @variant_loop_dialect() {
  %ci0 = "arith.constant"() <{value = 0 : index}> : () -> index
  %ci10 = "arith.constant"() <{value = 10 : index}> : () -> index
  %ci1 = "arith.constant"() <{value = 1 : index}> : () -> index
  %m = "memref.alloc"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<10xf32>

  "scf.for"(%ci0, %ci10, %ci1) ({
  ^bb0(%arg0: index):
    "scf.for"(%ci0, %ci10, %ci1) ({
    ^bb0(%arg1: index):
      %v0 = "arith.addi"(%arg0, %arg1) <{overflowFlags = #arith.overflow<none>}> : (index, index) -> index
      "scf.yield"() : () -> ()
    }) : (index, index, index) -> ()
    "scf.yield"() : () -> ()
  }) : (index, index, index) -> ()

  // CHECK: memref.alloc
  // CHECK-NEXT: scf.for
  // CHECK-NEXT: ^
  // CHECK-NEXT: scf.for
  // CHECK-NEXT: ^
  // CHECK-NEXT: arith.addi

  func.return
}

// -----

func.func @parallel_loop_with_invariant() {
  %c0 = "arith.constant"() <{value = 0 : index}> : () -> index
  %c10 = "arith.constant"() <{value = 10 : index}> : () -> index
  %c1 = "arith.constant"() <{value = 1 : index}> : () -> index
  %c7 = "arith.constant"() <{value = 7 : i32}> : () -> i32
  %c8 = "arith.constant"() <{value = 8 : i32}> : () -> i32

  "scf.parallel"(%c0, %c10, %c1) <{operandSegmentSizes = array<i32: 1, 1, 1, 0>}> ({
  ^bb0(%arg0: index):
    %v0 = "arith.addi"(%c7, %c8) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
    %v3 = "arith.addi"(%arg0, %arg0) <{overflowFlags = #arith.overflow<none>}> : (index, index) -> index
    "scf.reduce"() : () -> ()
  }) : (index, index, index) -> ()

  // CHECK: arith.constant
  // CHECK-NEXT: arith.constant
  // CHECK-NEXT: arith.constant
  // CHECK-NEXT: arith.constant
  // CHECK-NEXT: arith.constant
  // CHECK-NEXT: arith.addi
  // CHECK-NEXT: scf.parallel
  // CHECK-NEXT: ^
  // CHECK-NEXT: arith.addi
  // CHECK-NEXT: scf.reduce

  func.return
}

// -----

// Test that two ops that feed into each other are moved without violating
// dominance in non-graph regions.
// CHECK-LABEL: func.func @invariant_subgraph
func.func @invariant_subgraph(%lb: index, %ub: index, %step: index, %arg: i32) {
  // CHECK:      arith.addi
  // CHECK-NEXT: arith.addi
  // CHECK-NEXT: scf.for
  "scf.for"(%lb, %ub, %step) ({
  ^bb0(%i: index):
    // CHECK-NEXT: ^
    // CHECK-NEXT: test.op
    %v0 = "arith.addi"(%arg, %arg) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
    %v1 = "arith.addi"(%arg, %v0) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
    "test.op"(%v1) : (i32) -> ()
    "scf.yield"() : () -> ()
  }) : (index, index, index) -> ()
  func.return
}

// -----

// CHECK-LABEL: test_always_speculatable_op
func.func @test_always_speculatable_op(%lb: index, %ub: index, %step: index) {
  // CHECK: test.always_speculatable_op
  // CHECK-NEXT: scf.for
  "scf.for"(%lb, %ub, %step) ({
  ^bb0(%i: index):
    %val = "test.always_speculatable_op"() : () -> i32
    "scf.yield"() : () -> ()
  }) : (index, index, index) -> ()

  func.return
}

// -----

// CHECK-LABEL: test_never_speculatable_op
func.func @test_never_speculatable_op(%lb: index, %ub: index, %step: index) {
  // CHECK: scf.for
  // CHECK-NEXT: ^
  // CHECK-NEXT: test.never_speculatable_op
  "scf.for"(%lb, %ub, %step) ({
  ^bb0(%i: index):
    %val = "test.never_speculatable_op"() : () -> i32
    "scf.yield"() : () -> ()
  }) : (index, index, index) -> ()

  func.return
}

// -----

// CHECK-LABEL: test_conditionally_speculatable_op_success
func.func @test_conditionally_speculatable_op_success(%lb: index, %ub: index, %step: index) {
  // CHECK: arith.constant
  // CHECK-NEXT: test.conditionally_speculatable_op
  // CHECK-NEXT: scf.for
  "scf.for"(%lb, %ub, %step) ({
  ^bb0(%i: index):
    %const_val = "arith.constant"() <{value = 5 : i32}> : () -> i32
    %val = "test.conditionally_speculatable_op"(%const_val) : (i32) -> i32
    "scf.yield"() : () -> ()
  }) : (index, index, index) -> ()

  func.return
}

// -----

// CHECK-LABEL: test_conditionally_speculatable_op_failure
func.func @test_conditionally_speculatable_op_failure(%lb: index, %ub: index, %step: index, %arg: i32) {
  // CHECK: scf.for
  // CHECK-NEXT: ^
  // CHECK-NEXT: test.conditionally_speculatable_op
  %const_5 = "arith.constant"() <{value = 5 : i32}> : () -> i32
  %non_const = "arith.addi"(%arg, %const_5) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
  "scf.for"(%lb, %ub, %step) ({
  ^bb0(%i: index):
    %val = "test.conditionally_speculatable_op"(%non_const) : (i32) -> i32
    "scf.yield"() : () -> ()
  }) : (index, index, index) -> ()

  func.return
}

// -----

// CHECK-LABEL: test_recursively_speculatable_op_success
func.func @test_recursively_speculatable_op_success(%lb: index, %ub: index, %step: index, %arg: i32) {
  // CHECK: test.recursively_speculatable_op
  // CHECK: scf.for
  "scf.for"(%lb, %ub, %step) ({
  ^bb0(%i: index):
    %val = "test.recursively_speculatable_op"() ({
      %result = "arith.addi"(%arg, %arg) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      "test.region_yield"(%result) : (i32) -> ()
    }) : () -> i32
    "scf.yield"() : () -> ()
  }) : (index, index, index) -> ()

  func.return
}

// -----

// CHECK-LABEL: test_recursively_speculatable_op_failure
func.func @test_recursively_speculatable_op_failure(%lb: index, %ub: index, %step: index, %arg: i32) {
  // CHECK: scf.for
  // CHECK-NEXT: ^
  // CHECK-NEXT: test.recursively_speculatable_op
  "scf.for"(%lb, %ub, %step) ({
  ^bb0(%i: index):
    %val = "test.recursively_speculatable_op"() ({
      %result = "test.never_speculatable_op"() : () -> i32
      "test.region_yield"(%result) : (i32) -> ()
    }) : () -> i32
    "scf.yield"() : () -> ()
  }) : (index, index, index) -> ()

  func.return
}
