// RUN: scair-opt %s --allow-unregistered-dialect -p cse | filecheck %s

// CHECK:       builtin.module {

func.func @simple_constant() -> (i32, i32) {
  %0 = "arith.constant"() <{value = 1 : i32}> : () -> (i32)
  %1 = "arith.constant"() <{value = 1 : i32}> : () -> (i32)
  func.return %0, %1 : i32, i32
}
// CHECK-NEXT:    func.func @simple_constant() -> (i32, i32) {
// CHECK-NEXT:      %0 = "arith.constant"() <{value = 1 : i32}> : () -> (i32)
// CHECK-NEXT:      func.return %0, %0 : i32, i32
// CHECK-NEXT:    }

func.func @simple_float_constant() -> (f32, f32) {
  %2 = "arith.constant"() <{value = 1.0 : f32}> : () -> (f32)
  %3 = "arith.constant"() <{value = 1.0 : f32}> : () -> (f32)
  func.return %2, %3 : f32, f32
}
// CHECK-NEXT:    func.func @simple_float_constant() -> (f32, f32) {
// CHECK-NEXT:      %0 = "arith.constant"() <{value = 1.0 : f32}> : () -> (f32)
// CHECK-NEXT:      func.return %0, %0 : f32, f32
// CHECK-NEXT:    

func.func @basic() -> (index, index) {
  %4 = "arith.constant"() <{value = 0 : index}> : () -> (index)
  %5 = "arith.constant"() <{value = 0 : index}> : () -> (index)
  %6 = "affine.apply"(%4) <{map = affine_map<(d0)[] -> (d0 mod 2)>}> : (index) -> (index)
  %7 = "affine.apply"(%5) <{map = affine_map<(d0)[] -> (d0 mod 2)>}> : (index) -> (index)
  func.return %6, %7 : index, index
}
// CHECK-NEXT:    func.func @basic() -> (index, index) {
// CHECK-NEXT:      %0 = "arith.constant"() <{value = 0 : index}> : () -> (index)
// CHECK-NEXT:      %1 = "affine.apply"(%0) <{map = affine_map<(d0)[] -> (d0 mod 2)>}> : (index) -> (index)
// CHECK-NEXT:      func.return %1, %1 : index, index
// CHECK-NEXT:    }

func.func @many(%8: f32, %9: f32) -> f32 {
  %10 = "arith.addf"(%8, %9) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> (f32)
  %11 = "arith.addf"(%8, %9) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> (f32)
  %12 = "arith.addf"(%8, %9) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> (f32)
  %13 = "arith.addf"(%8, %9) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> (f32)
  %14 = "arith.addf"(%10, %11) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> (f32)
  %15 = "arith.addf"(%12, %13) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> (f32)
  %16 = "arith.addf"(%10, %12) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> (f32)
  %17 = "arith.addf"(%14, %15) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> (f32)
  %18 = "arith.addf"(%15, %16) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> (f32)
  %19 = "arith.addf"(%17, %18) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> (f32)
  func.return %19 : f32
}
// CHECK-NEXT:    func.func @many(%4: f32, %5: f32) -> f32 {
// CHECK-NEXT:      %6 = "arith.addf"(%4, %5) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> (f32)
// CHECK-NEXT:      %7 = "arith.addf"(%6, %6) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> (f32)
// CHECK-NEXT:      %8 = "arith.addf"(%7, %7) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> (f32)
// CHECK-NEXT:      %9 = "arith.addf"(%8, %8) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> (f32)
// CHECK-NEXT:      func.return %9 : f32
// CHECK-NEXT:    }

func.func @different_ops() -> (i32, i32) {
  %20 = "arith.constant"() <{value = 0 : i32}> : () -> (i32)
  %21 = "arith.constant"() <{value = 1 : i32}> : () -> (i32)
  func.return %20, %21 : i32, i32
}
// CHECK-NEXT:    func.func @different_ops() -> (i32, i32) {
// CHECK-NEXT:      %10 = "arith.constant"() <{value = 0 : i32}> : () -> (i32)
// CHECK-NEXT:      %11 = "arith.constant"() <{value = 1 : i32}> : () -> (i32)
// CHECK-NEXT:      func.return %10, %11 : i32, i32
// CHECK-NEXT:    }

// TODO: Requires memref.cast 
// "func.func"() <{sym_name = "different_results", function_type = (memref<*xf32>) -> (memref<?x?xf32>, memref<4x?xf32>)}> ({
// ^0(%arg0 : memref<*xf32>):
//   %0 = "memref.cast"(%arg0) : (memref<*xf32>) -> memref<?x?xf32>
//   %1 = "memref.cast"(%arg0) : (memref<*xf32>) -> memref<4x?xf32>
//   "func.return"(%0, %1) : (memref<?x?xf32>, memref<4x?xf32>) -> ()
// }) : () -> ()

func.func @different_attributes(%22: index, %23: index) -> (i1, i1, i1) {
  %24 = "arith.cmpi"(%22, %23) <{predicate = 2}> : (index, index) -> (i1)
  %25 = "arith.cmpi"(%22, %23) <{predicate = 1}> : (index, index) -> (i1)
  %26 = "arith.cmpi"(%22, %23) <{predicate = 1}> : (index, index) -> (i1)
  func.return %24, %25, %26 : i1, i1, i1
}
// CHECK-NEXT:    func.func @different_attributes(%12: index, %13: index) -> (i1, i1, i1) {
// CHECK-NEXT:      %14 = "arith.cmpi"(%12, %13) <{predicate = 2}> : (index, index) -> (i1)
// CHECK-NEXT:      %15 = "arith.cmpi"(%12, %13) <{predicate = 1}> : (index, index) -> (i1)
// CHECK-NEXT:      func.return %14, %15, %15 : i1, i1, i1
// CHECK-NEXT:    }

func.func @side_effect() -> (memref<2x1xf32>, memref<2x1xf32>) {
  %27 = "memref.alloc"() <{alignment = 0, operandSegmentSizes = array<i32: 0, 0>}> : () -> (memref<2x1xf32>)
  %28 = "memref.alloc"() <{alignment = 0, operandSegmentSizes = array<i32: 0, 0>}> : () -> (memref<2x1xf32>)
  func.return %27, %28 : memref<2x1xf32>, memref<2x1xf32>
}
// CHECK-NEXT:    func.func @side_effect() -> (memref<2x1xf32>, memref<2x1xf32>) {
// CHECK-NEXT:      %16 = "memref.alloc"() <{alignment = 0, operandSegmentSizes = array<i32: 0, 0>}> : () -> (memref<2x1xf32>)
// CHECK-NEXT:      %17 = "memref.alloc"() <{alignment = 0, operandSegmentSizes = array<i32: 0, 0>}> : () -> (memref<2x1xf32>)
// CHECK-NEXT:      func.return %16, %17 : memref<2x1xf32>, memref<2x1xf32>
// CHECK-NEXT:    }

func.func @down_propagate_for() {
  %29 = "arith.constant"() <{value = 1 : i32}> : () -> (i32)
  "affine.for"() <{lowerBoundMap = affine_map<()[] -> (0)>, upperBoundMap = affine_map<()[] -> (4)>, step = 1 : index, operandSegmentSizes = array<i32: 0, 0, 0>}> ({
  ^bb0(%30: index):
    %31 = "arith.constant"() <{value = 1 : i32}> : () -> (i32)
    "foo"(%29, %31) : (i32, i32) -> ()
    "affine.yield"() : () -> ()
  }) : () -> ()
  func.return
}
// CHECK-NEXT:    func.func @down_propagate_for() {
// CHECK-NEXT:      %18 = "arith.constant"() <{value = 1 : i32}> : () -> (i32)
// CHECK-NEXT:      "affine.for"() <{lowerBoundMap = affine_map<()[] -> (0)>, upperBoundMap = affine_map<()[] -> (4)>, step = 1 : index, operandSegmentSizes = array<i32: 0, 0, 0>}> ({
// CHECK-NEXT:      ^bb0(%19: index):
// CHECK-NEXT:        "foo"(%18, %18) : (i32, i32) -> ()
// CHECK-NEXT:        "affine.yield"() : () -> ()
// CHECK-NEXT:      }) : () -> ()
// CHECK-NEXT:      func.return
// CHECK-NEXT:    }

// TODO: This requires the notions of SSACFG regions and block dominance
// "func.func"() <{sym_name = "down_propagate", function_type = () -> i32}> ({
//   %0 = "arith.constant"() <{value = 1 : i32}> : () -> i32
//   %1 = "arith.constant"() <{value = true}> : () -> i1
//   "cf.cond_br"(%1, %0) [^0, ^1] <{operandSegmentSizes = array<i32: 1, 0, 1>}> : (i1, i32) -> ()
// ^0:
//   %2 = "arith.constant"() <{value = 1 : i32}> : () -> i32
//   "cf.br"(%2) [^1] : (i32) -> ()
// ^1(%3 : i32):
//   "func.return"(%3) : (i32) -> ()
// }) : () -> ()
func.func @up_propagate_for() -> i32 {
  "affine.for"() <{lowerBoundMap = affine_map<()[] -> (0)>, upperBoundMap = affine_map<()[] -> (4)>, step = 1 : index, operandSegmentSizes = array<i32: 0, 0, 0>}> ({
  ^bb0(%30: index):
    %31 = "arith.constant"() <{value = 1 : i32}> : () -> (i32)
    "foo"(%31) : (i32) -> ()
    "affine.yield"() : () -> ()
  }) : () -> ()
  %30 = "arith.constant"() <{value = 1 : i32}> : () -> (i32)
  func.return %30 : i32
}
// CHECK-NEXT:    func.func @up_propagate_for() -> i32 {
// CHECK-NEXT:      "affine.for"() <{lowerBoundMap = affine_map<()[] -> (0)>, upperBoundMap = affine_map<()[] -> (4)>, step = 1 : index, operandSegmentSizes = array<i32: 0, 0, 0>}> ({
// CHECK-NEXT:      ^bb0(%19: index):
// CHECK-NEXT:        %20 = "arith.constant"() <{value = 1 : i32}> : () -> (i32)
// CHECK-NEXT:        "foo"(%20) : (i32) -> ()
// CHECK-NEXT:        "affine.yield"() : () -> ()
// CHECK-NEXT:      }) : () -> ()
// CHECK-NEXT:      %19 = "arith.constant"() <{value = 1 : i32}> : () -> (i32)
// CHECK-NEXT:      func.return %19 : i32
// CHECK-NEXT:    }

// TODO: Requires the notions of SSACFG regions and block dominance
// "func.func"() <{sym_name = "up_propagate", function_type = () -> i32}> ({
//   %0 = "arith.constant"() <{value = 0 : i32}> : () -> i32
//   %1 = "arith.constant"() <{value = true}> : () -> i1
//   "cf.cond_br"(%1, %0) [^0, ^1] <{operandSegmentSizes = array<i32: 1, 0, 1>}> : (i1, i32) -> ()
// ^0:
//   %2 = "arith.constant"() <{value = 1 : i32}> : () -> i32
//   "cf.br"(%2) [^1] : (i32) -> ()
// ^1(%3 : i32):
//   %4 = "arith.constant"() <{value = 1 : i32}> : () -> i32
//   %5 = "arith.addi"(%3, %4) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
//   "func.return"(%5) : (i32) -> ()
// }) : () -> ()
// "func.func"() <{sym_name = "up_propagate_region", function_type = () -> i32}> ({
//   %0 = "foo.region"() ({
//     %1 = "arith.constant"() <{value = 0 : i32}> : () -> i32
//     %2 = "arith.constant"() <{value = true}> : () -> i1
//     "cf.cond_br"(%2, %1) [^0, ^1] <{operandSegmentSizes = array<i32: 1, 0, 1>}> : (i1, i32) -> ()
//   ^0:
//     %3 = "arith.constant"() <{value = 1 : i32}> : () -> i32
//     "cf.br"(%3) [^1] : (i32) -> ()
//   ^1(%4 : i32):
//     %5 = "arith.constant"() <{value = 1 : i32}> : () -> i32
//     %6 = "arith.addi"(%4, %5) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
//     "foo.yield"(%6) : (i32) -> ()
//   }) : () -> i32
//   "func.return"(%0) : (i32) -> ()
// }) : () -> ()

/// This test checks that nested regions that are isolated from above are
/// properly handled.
func.func @nested_isolated() -> i32 {
  %31 = "arith.constant"() <{value = 1 : i32}> : () -> (i32)
  func.func @nested_func() {
    %32 = "arith.constant"() <{value = 1 : i32}> : () -> (i32)
    "foo.yield"(%32) : (i32) -> ()
  }
  "foo.region"() ({
    %33 = "arith.constant"() <{value = 1 : i32}> : () -> (i32)
    "foo.yield"(%33) : (i32) -> ()
  }) : () -> ()
  func.return %31 : i32
}
// CHECK-NEXT:    func.func @nested_isolated() -> i32 {
// CHECK-NEXT:      %20 = "arith.constant"() <{value = 1 : i32}> : () -> (i32)
// CHECK-NEXT:      func.func @nested_func() {
// CHECK-NEXT:        %21 = "arith.constant"() <{value = 1 : i32}> : () -> (i32)
// CHECK-NEXT:        "foo.yield"(%21) : (i32) -> ()
// CHECK-NEXT:      }
// CHECK-NEXT:      "foo.region"() ({
// CHECK-NEXT:        %22 = "arith.constant"() <{value = 1 : i32}> : () -> (i32)
// CHECK-NEXT:        "foo.yield"(%22) : (i32) -> ()
// CHECK-NEXT:      }) : () -> ()
// CHECK-NEXT:      func.return %20 : i32
// CHECK-NEXT:    }

/// This test is checking that CSE gracefully handles values in graph regions
/// where the use occurs before the def, and one of the defs could be CSE'd with
/// the other.

/// NB: Above is the verbatim comment from the original test in MLIR.
/// Though, MLIR itself does not seem to CSE those constants...
/// Mentionning in case there is a good reason, but I guess this might just work better
/// here!
func.func @use_before_def() {
  "test.graph_region"() ({
    %33 = "arith.addi"(%34, %35) : (i32, i32) -> (i32)
    %34 = "arith.constant"() <{value = 1 : i32}> : () -> (i32)
    %35 = "arith.constant"() <{value = 1 : i32}> : () -> (i32)
    "foo.yield"(%33) : (i32) -> ()
  }) : () -> ()
  func.return
}
// CHECK-NEXT:    func.func @use_before_def() {
// CHECK-NEXT:      "test.graph_region"() ({
// CHECK-NEXT:        %22 = "arith.addi"(%23, %23) : (i32, i32) -> (i32)
// CHECK-NEXT:        %23 = "arith.constant"() <{value = 1 : i32}> : () -> (i32)
// CHECK-NEXT:        "foo.yield"(%22) : (i32) -> ()
// CHECK-NEXT:      }) : () -> ()
// CHECK-NEXT:      func.return
// CHECK-NEXT:    }

// TODO: The following tests require more nuanced notions of side effects yet to be implemented in ScaIR
// "func.func"() <{sym_name = "remove_direct_duplicated_read_op", function_type = () -> i32}> ({
//   %0 = "test.op_with_memread"() : () -> i32
//   %1 = "test.op_with_memread"() : () -> i32
//   %2 = "arith.addi"(%0, %1) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
//   "func.return"(%2) : (i32) -> ()
// }) : () -> ()
// "func.func"() <{sym_name = "remove_multiple_duplicated_read_op", function_type = () -> i64}> ({
//   %0 = "test.op_with_memread"() : () -> i64
//   %1 = "test.op_with_memread"() : () -> i64
//   %2 = "arith.addi"(%0, %1) <{overflowFlags = #arith.overflow<none>}> : (i64, i64) -> i64
//   %3 = "test.op_with_memread"() : () -> i64
//   %4 = "arith.addi"(%2, %3) <{overflowFlags = #arith.overflow<none>}> : (i64, i64) -> i64
//   %5 = "test.op_with_memread"() : () -> i64
//   %6 = "arith.addi"(%4, %5) <{overflowFlags = #arith.overflow<none>}> : (i64, i64) -> i64
//   "func.return"(%6) : (i64) -> ()
// }) : () -> ()
// "func.func"() <{sym_name = "dont_remove_duplicated_read_op_with_sideeffecting", function_type = () -> i32}> ({
//   %0 = "test.op_with_memread"() : () -> i32
//   "test.op_with_memwrite"() : () -> ()
//   %1 = "test.op_with_memread"() : () -> i32
//   %2 = "arith.addi"(%0, %1) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
//   "func.return"(%2) : (i32) -> ()
// }) : () -> ()
// "func.func"() <{sym_name = "cse_single_block_ops", function_type = (tensor<?x?xf32>, tensor<?x?xf32>) -> (tensor<?x?xf32>, tensor<?x?xf32>)}> ({
// ^0(%arg0 : tensor<?x?xf32>, %arg1 : tensor<?x?xf32>):
//   %0 = "test.pureop"(%arg0, %arg1) ({
//   ^1(%arg2 : f32):
//     "test.region_yield"(%arg2) : (f32) -> ()
//   }) : (tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xf32>
//   %1 = "test.pureop"(%arg0, %arg1) ({
//   ^2(%arg2_1 : f32):
//     "test.region_yield"(%arg2_1) : (f32) -> ()
//   }) : (tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xf32>
//   "func.return"(%0, %1) : (tensor<?x?xf32>, tensor<?x?xf32>) -> ()
// }) : () -> ()
// "func.func"() <{sym_name = "no_cse_varied_bbargs", function_type = (tensor<?x?xf32>, tensor<?x?xf32>) -> (tensor<?x?xf32>, tensor<?x?xf32>)}> ({
// ^0(%arg0 : tensor<?x?xf32>, %arg1 : tensor<?x?xf32>):
//   %0 = "test.pureop"(%arg0, %arg1) ({
//   ^1(%arg2 : f32, %arg3 : f32):
//     "test.region_yield"(%arg2) : (f32) -> ()
//   }) : (tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xf32>
//   %1 = "test.pureop"(%arg0, %arg1) ({
//   ^2(%arg2_1 : f32):
//     "test.region_yield"(%arg2_1) : (f32) -> ()
//   }) : (tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xf32>
//   "func.return"(%0, %1) : (tensor<?x?xf32>, tensor<?x?xf32>) -> ()
// }) : () -> ()
// "func.func"() <{sym_name = "no_cse_region_difference_simple", function_type = (tensor<?x?xf32>, tensor<?x?xf32>) -> (tensor<?x?xf32>, tensor<?x?xf32>)}> ({
// ^0(%arg0 : tensor<?x?xf32>, %arg1 : tensor<?x?xf32>):
//   %0 = "test.pureop"(%arg0, %arg1) ({
//   ^1(%arg2 : f32, %arg3 : f32):
//     "test.region_yield"(%arg2) : (f32) -> ()
//   }) : (tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xf32>
//   %1 = "test.pureop"(%arg0, %arg1) ({
//   ^2(%arg2_1 : f32, %arg3_1 : f32):
//     "test.region_yield"(%arg3_1) : (f32) -> ()
//   }) : (tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xf32>
//   "func.return"(%0, %1) : (tensor<?x?xf32>, tensor<?x?xf32>) -> ()
// }) : () -> ()
// "func.func"() <{sym_name = "cse_single_block_ops_identical_bodies", function_type = (tensor<?x?xf32>, tensor<?x?xf32>, f32, i1) -> (tensor<?x?xf32>, tensor<?x?xf32>)}> ({
// ^0(%arg0 : tensor<?x?xf32>, %arg1 : tensor<?x?xf32>, %arg2 : f32, %arg3 : i1):
//   %0 = "test.pureop"(%arg0, %arg1) ({
//   ^1(%arg4 : f32, %arg5 : f32):
//     %1 = "arith.divf"(%arg4, %arg5) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
//     %2 = "arith.remf"(%arg4, %arg2) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
//     %3 = "arith.select"(%arg3, %1, %2) : (i1, f32, f32) -> f32
//     "test.region_yield"(%3) : (f32) -> ()
//   }) : (tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xf32>
//   %4 = "test.pureop"(%arg0, %arg1) ({
//   ^2(%arg4_1 : f32, %arg5_1 : f32):
//     %5 = "arith.divf"(%arg4_1, %arg5_1) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
//     %6 = "arith.remf"(%arg4_1, %arg2) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
//     %7 = "arith.select"(%arg3, %5, %6) : (i1, f32, f32) -> f32
//     "test.region_yield"(%7) : (f32) -> ()
//   }) : (tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xf32>
//   "func.return"(%0, %4) : (tensor<?x?xf32>, tensor<?x?xf32>) -> ()
// }) : () -> ()
// "func.func"() <{sym_name = "no_cse_single_block_ops_different_bodies", function_type = (tensor<?x?xf32>, tensor<?x?xf32>, f32, i1) -> (tensor<?x?xf32>, tensor<?x?xf32>)}> ({
// ^0(%arg0 : tensor<?x?xf32>, %arg1 : tensor<?x?xf32>, %arg2 : f32, %arg3 : i1):
//   %0 = "test.pureop"(%arg0, %arg1) ({
//   ^1(%arg4 : f32, %arg5 : f32):
//     %1 = "arith.divf"(%arg4, %arg5) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
//     %2 = "arith.remf"(%arg4, %arg2) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
//     %3 = "arith.select"(%arg3, %1, %2) : (i1, f32, f32) -> f32
//     "test.region_yield"(%3) : (f32) -> ()
//   }) : (tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xf32>
//   %4 = "test.pureop"(%arg0, %arg1) ({
//   ^2(%arg4_1 : f32, %arg5_1 : f32):
//     %5 = "arith.divf"(%arg4_1, %arg5_1) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
//     %6 = "arith.remf"(%arg4_1, %arg2) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
//     %7 = "arith.select"(%arg3, %6, %5) : (i1, f32, f32) -> f32
//     "test.region_yield"(%7) : (f32) -> ()
//   }) : (tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xf32>
//   "func.return"(%0, %4) : (tensor<?x?xf32>, tensor<?x?xf32>) -> ()
// }) : () -> ()
// "func.func"() <{sym_name = "failing_issue_59135", function_type = (tensor<2x2xi1>, f32, tensor<2xi1>) -> (tensor<2xi1>, tensor<2xi1>)}> ({
// ^0(%arg0 : tensor<2x2xi1>, %arg1 : f32, %arg2 : tensor<2xi1>):
//   %0 = "arith.constant"() <{value = false}> : () -> i1
//   %1 = "arith.constant"() <{value = true}> : () -> i1
//   %2 = "test.pureop"(%arg2) ({
//   ^1(%arg3 : i1):
//     %3 = "arith.constant"() <{value = true}> : () -> i1
//     "test.region_yield"(%3) : (i1) -> ()
//   }) : (tensor<2xi1>) -> tensor<2xi1>
//   %4 = "test.pureop"(%arg2) ({
//   ^2(%arg3_1 : i1):
//     %5 = "arith.constant"() <{value = true}> : () -> i1
//     "test.region_yield"(%5) : (i1) -> ()
//   }) : (tensor<2xi1>) -> tensor<2xi1>
//   %6 = "arith.maxsi"(%0, %1) : (i1, i1) -> i1
//   "func.return"(%2, %4) : (tensor<2xi1>, tensor<2xi1>) -> ()
// }) : () -> ()
// "func.func"() <{sym_name = "cse_multiple_regions", function_type = (i1, tensor<5xf32>) -> (tensor<5xf32>, tensor<5xf32>)}> ({
// ^0(%arg0 : i1, %arg1 : tensor<5xf32>):
//   %0 = "scf.if"(%arg0) ({
//     %1 = "tensor.empty"() : () -> tensor<5xf32>
//     "scf.yield"(%1) : (tensor<5xf32>) -> ()
//   }, {
//     "scf.yield"(%arg1) : (tensor<5xf32>) -> ()
//   }) : (i1) -> tensor<5xf32>
//   %2 = "scf.if"(%arg0) ({
//     %3 = "tensor.empty"() : () -> tensor<5xf32>
//     "scf.yield"(%3) : (tensor<5xf32>) -> ()
//   }, {
//     "scf.yield"(%arg1) : (tensor<5xf32>) -> ()
//   }) : (i1) -> tensor<5xf32>
//   "func.return"(%0, %2) : (tensor<5xf32>, tensor<5xf32>) -> ()
// }) : () -> ()
// "func.func"() <{sym_name = "no_cse_multiple_regions_side_effect", function_type = (i1, memref<5xf32>) -> (memref<5xf32>, memref<5xf32>)}> ({
// ^0(%arg0 : i1, %arg1 : memref<5xf32>):
//   %0 = "scf.if"(%arg0) ({
//     %1 = "memref.alloc"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<5xf32>
//     "scf.yield"(%1) : (memref<5xf32>) -> ()
//   }, {
//     "scf.yield"(%arg1) : (memref<5xf32>) -> ()
//   }) : (i1) -> memref<5xf32>
//   %2 = "scf.if"(%arg0) ({
//     %3 = "memref.alloc"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<5xf32>
//     "scf.yield"(%3) : (memref<5xf32>) -> ()
//   }, {
//     "scf.yield"(%arg1) : (memref<5xf32>) -> ()
//   }) : (i1) -> memref<5xf32>
//   "func.return"(%0, %2) : (memref<5xf32>, memref<5xf32>) -> ()
// }) : () -> ()
// "func.func"() <{sym_name = "cse_recursive_effects_success", function_type = () -> (i32, i32, i32)}> ({
//   %0 = "test.op_with_memread"() : () -> i32
//   %1 = "arith.constant"() <{value = true}> : () -> i1
//   %2 = "scf.if"(%1) ({
//     %3 = "arith.constant"() <{value = 42 : i32}> : () -> i32
//     "scf.yield"(%3) : (i32) -> ()
//   }, {
//     %4 = "arith.constant"() <{value = 24 : i32}> : () -> i32
//     "scf.yield"(%4) : (i32) -> ()
//   }) : (i1) -> i32
//   %5 = "test.op_with_memread"() : () -> i32
//   "func.return"(%0, %5, %2) : (i32, i32, i32) -> ()
// }) : () -> ()
// "func.func"() <{sym_name = "cse_recursive_effects_failure", function_type = () -> (i32, i32, i32)}> ({
//   %0 = "test.op_with_memread"() : () -> i32
//   %1 = "arith.constant"() <{value = true}> : () -> i1
//   %2 = "scf.if"(%1) ({
//     "test.op_with_memwrite"() : () -> ()
//     %3 = "arith.constant"() <{value = 42 : i32}> : () -> i32
//     "scf.yield"(%3) : (i32) -> ()
//   }, {
//     %4 = "arith.constant"() <{value = 24 : i32}> : () -> i32
//     "scf.yield"(%4) : (i32) -> ()
//   }) : (i1) -> i32
//   %5 = "test.op_with_memread"() : () -> i32
//   "func.return"(%0, %5, %2) : (i32, i32, i32) -> ()
// }) : () -> ()

// CHECK-NEXT:  }