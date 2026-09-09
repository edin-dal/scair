// RUN: scair-opt %s -p canonicalize | filecheck %s

// Coverage for `EraseIdentityGeneric`, the port of MLIR's
// `EraseIdentityLinalgOp`, registered on `linalg.generic`.
//
// Note on the memref cases at the bottom: ScaIR's method-free port of Linalg has
// no equivalent of MLIR's `LinalgOp::getEffects`, which declares a write on each
// output operand. A `linalg.generic` with no results therefore looks memory
// effect free, and the pass' own `RemoveUnusedOperations` deletes it before the
// identity pattern ever gets a say. The buffer branch of the pattern is ported
// faithfully all the same, and those tests pin the current behaviour so that
// modelling the effects shows up here.

// CHECK:       #map = affine_map<(d0, d1)[] -> (d1, d0)>
// CHECK-NEXT:  #map1 = affine_map<(d0, d1)[] -> (d0, d1)>
// CHECK-NEXT:  builtin.module {

// An identity generic on tensors folds away, forwarding its input.
func.func @identity_tensor(%0: tensor<4x8xf32>, %1: tensor<4x8xf32>) -> tensor<4x8xf32> {
  %2 = "linalg.generic"(%0, %1) <{
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>],
    operandSegmentSizes = array<i32: 1, 1>
  }> ({
  ^bb0(%a: f32, %b: f32):
    "linalg.yield"(%a) : (f32) -> ()
  }) : (tensor<4x8xf32>, tensor<4x8xf32>) -> tensor<4x8xf32>
  func.return %2 : tensor<4x8xf32>
}
// CHECK-NEXT:    func.func @identity_tensor(%0: tensor<4x8xf32>, %1: tensor<4x8xf32>) -> tensor<4x8xf32> {
// CHECK-NEXT:      func.return %0 : tensor<4x8xf32>
// CHECK-NEXT:    }

// The yielded block argument may just as well be the one of the output operand.
func.func @identity_tensor_yields_init(%0: tensor<4x8xf32>, %1: tensor<4x8xf32>) -> tensor<4x8xf32> {
  %2 = "linalg.generic"(%0, %1) <{
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>],
    operandSegmentSizes = array<i32: 1, 1>
  }> ({
  ^bb0(%a: f32, %b: f32):
    "linalg.yield"(%b) : (f32) -> ()
  }) : (tensor<4x8xf32>, tensor<4x8xf32>) -> tensor<4x8xf32>
  func.return %2 : tensor<4x8xf32>
}
// CHECK-NEXT:    func.func @identity_tensor_yields_init(%0: tensor<4x8xf32>, %1: tensor<4x8xf32>) -> tensor<4x8xf32> {
// CHECK-NEXT:      func.return %1 : tensor<4x8xf32>
// CHECK-NEXT:    }

// Differing indexing maps make this a transpose rather than an identity.
func.func @transpose_tensor(%0: tensor<4x4xf32>, %1: tensor<4x4xf32>) -> tensor<4x4xf32> {
  %2 = "linalg.generic"(%0, %1) <{
    indexing_maps = [affine_map<(d0, d1) -> (d1, d0)>, affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>],
    operandSegmentSizes = array<i32: 1, 1>
  }> ({
  ^bb0(%a: f32, %b: f32):
    "linalg.yield"(%a) : (f32) -> ()
  }) : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
  func.return %2 : tensor<4x4xf32>
}
// CHECK-NEXT:    func.func @transpose_tensor(%0: tensor<4x4xf32>, %1: tensor<4x4xf32>) -> tensor<4x4xf32> {
// CHECK-NEXT:      %2 = "linalg.generic"(%0, %1) <{indexing_maps = [#map, #map1], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operandSegmentSizes = array<i32: 1, 1>}> ({
// CHECK-NEXT:      ^bb0(%3: f32, %4: f32):
// CHECK-NEXT:        "linalg.yield"(%3) : (f32) -> ()
// CHECK-NEXT:      }) : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
// CHECK-NEXT:      func.return %2 : tensor<4x4xf32>
// CHECK-NEXT:    }

// A body that computes something is not an identity.
func.func @add_tensor(%0: tensor<4x8xf32>, %1: tensor<4x8xf32>) -> tensor<4x8xf32> {
  %2 = "linalg.generic"(%0, %1) <{
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>],
    operandSegmentSizes = array<i32: 1, 1>
  }> ({
  ^bb0(%a: f32, %b: f32):
    %c = "arith.addf"(%a, %b) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
    "linalg.yield"(%c) : (f32) -> ()
  }) : (tensor<4x8xf32>, tensor<4x8xf32>) -> tensor<4x8xf32>
  func.return %2 : tensor<4x8xf32>
}
// CHECK-NEXT:    func.func @add_tensor(%0: tensor<4x8xf32>, %1: tensor<4x8xf32>) -> tensor<4x8xf32> {
// CHECK-NEXT:      %2 = "linalg.generic"(%0, %1) <{indexing_maps = [#map1, #map1], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operandSegmentSizes = array<i32: 1, 1>}> ({
// CHECK-NEXT:      ^bb0(%3: f32, %4: f32):
// CHECK-NEXT:        %5 = "arith.addf"(%3, %4) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
// CHECK-NEXT:        "linalg.yield"(%5) : (f32) -> ()
// CHECK-NEXT:      }) : (tensor<4x8xf32>, tensor<4x8xf32>) -> tensor<4x8xf32>
// CHECK-NEXT:      func.return %2 : tensor<4x8xf32>
// CHECK-NEXT:    }

// The forwarded operand and the result have different types, which MLIR would
// bridge with a `tensor.cast`. ScaIR has no such op, so the pattern bails out.
func.func @identity_tensor_cast(%0: tensor<?x?xf32>, %1: tensor<4x8xf32>) -> tensor<4x8xf32> {
  %2 = "linalg.generic"(%0, %1) <{
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>],
    operandSegmentSizes = array<i32: 1, 1>
  }> ({
  ^bb0(%a: f32, %b: f32):
    "linalg.yield"(%a) : (f32) -> ()
  }) : (tensor<?x?xf32>, tensor<4x8xf32>) -> tensor<4x8xf32>
  func.return %2 : tensor<4x8xf32>
}
// CHECK-NEXT:    func.func @identity_tensor_cast(%0: tensor<?x?xf32>, %1: tensor<4x8xf32>) -> tensor<4x8xf32> {
// CHECK-NEXT:      %2 = "linalg.generic"(%0, %1) <{indexing_maps = [#map1, #map1], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operandSegmentSizes = array<i32: 1, 1>}> ({
// CHECK-NEXT:      ^bb0(%3: f32, %4: f32):
// CHECK-NEXT:        "linalg.yield"(%3) : (f32) -> ()
// CHECK-NEXT:      }) : (tensor<?x?xf32>, tensor<4x8xf32>) -> tensor<4x8xf32>
// CHECK-NEXT:      func.return %2 : tensor<4x8xf32>
// CHECK-NEXT:    }

// An identity generic on memrefs copying a buffer onto itself is erased — as the
// buffer branch of the pattern says it should be, though as noted at the top of
// this file `RemoveUnusedOperations` gets there first.
func.func @identity_memref(%0: memref<4x8xf32>) {
  "linalg.generic"(%0, %0) <{
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>],
    operandSegmentSizes = array<i32: 1, 1>
  }> ({
  ^bb0(%a: f32, %b: f32):
    "linalg.yield"(%a) : (f32) -> ()
  }) : (memref<4x8xf32>, memref<4x8xf32>) -> ()
  func.return
}
// CHECK-NEXT:    func.func @identity_memref(%0: memref<4x8xf32>) {
// CHECK-NEXT:      func.return
// CHECK-NEXT:    }

// TODO: this one copies between two *different* buffers and so is not an
// identity — the pattern rightly declines it, but it disappears all the same,
// for the missing-write-effect reason described at the top of this file.
func.func @copy_memref(%0: memref<4x8xf32>, %1: memref<4x8xf32>) {
  "linalg.generic"(%0, %1) <{
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>],
    operandSegmentSizes = array<i32: 1, 1>
  }> ({
  ^bb0(%a: f32, %b: f32):
    "linalg.yield"(%a) : (f32) -> ()
  }) : (memref<4x8xf32>, memref<4x8xf32>) -> ()
  func.return
}
// CHECK-NEXT:    func.func @copy_memref(%0: memref<4x8xf32>, %1: memref<4x8xf32>) {
// CHECK-NEXT:      func.return
// CHECK-NEXT:    }

// TODO: likewise, a fill-like op yields a value that is not a block argument, so
// the pattern declines it, and it disappears for the same unrelated reason.
func.func @fill_memref(%0: memref<4x8xf32>, %1: f32) {
  "linalg.generic"(%0, %0) <{
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>],
    operandSegmentSizes = array<i32: 1, 1>
  }> ({
  ^bb0(%a: f32, %b: f32):
    "linalg.yield"(%1) : (f32) -> ()
  }) : (memref<4x8xf32>, memref<4x8xf32>) -> ()
  func.return
}
// CHECK-NEXT:    func.func @fill_memref(%0: memref<4x8xf32>, %1: f32) {
// CHECK-NEXT:      func.return
// CHECK-NEXT:    }

// TODO: and mixed tensor/memref semantics, which upstream declines outright.
func.func @mixed_semantics(%0: tensor<4x8xf32>, %1: memref<4x8xf32>) {
  "linalg.generic"(%0, %1) <{
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>],
    operandSegmentSizes = array<i32: 1, 1>
  }> ({
  ^bb0(%a: f32, %b: f32):
    "linalg.yield"(%a) : (f32) -> ()
  }) : (tensor<4x8xf32>, memref<4x8xf32>) -> ()
  func.return
}
// CHECK-NEXT:    func.func @mixed_semantics(%0: tensor<4x8xf32>, %1: memref<4x8xf32>) {
// CHECK-NEXT:      func.return
// CHECK-NEXT:    }

// CHECK-NEXT:  }
