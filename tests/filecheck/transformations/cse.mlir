// RUN: scair-opt %s -p cse | filecheck %s

"func.func"() <{sym_name = "simple_constant", function_type = () -> (i32, i32)}> ({
  %0 = "arith.constant"() <{value = 1 : i32}> : () -> (i32)
  %1 = "arith.constant"() <{value = 1 : i32}> : () -> (i32)
  func.return %0, %1 : i32, i32
}) : () -> ()

"func.func"() <{sym_name = "simple_float_constant", function_type = () -> (f32, f32)}> ({
  %0 = "arith.constant"() <{value = 1.0 : f32}> : () -> (f32)
  %1 = "arith.constant"() <{value = 1.0 : f32}> : () -> (f32)
  func.return %0, %1 : f32, f32
}) : () -> ()

"func.func"() <{sym_name = "basic", function_type = () -> (index, index)}> ({
  %0 = "arith.constant"() <{value = 0 : index}> : () -> (index)
  %1 = "arith.constant"() <{value = 0 : index}> : () -> (index)
  %2 = "affine.apply"(%0) <{map = affine_map<(d0)[] -> (d0 mod 2)>}> : (index) -> (index)
  %3 = "affine.apply"(%1) <{map = affine_map<(d0)[] -> (d0 mod 2)>}> : (index) -> (index)
  func.return %2, %3 : index, index
}) : () -> ()

"func.func"() <{sym_name = "many", function_type = (f32, f32) -> f32}> ({
^bb0(%0: f32, %1: f32):
  %2 = "arith.addf"(%0, %1) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> (f32)
  %3 = "arith.addf"(%0, %1) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> (f32)
  %4 = "arith.addf"(%0, %1) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> (f32)
  %5 = "arith.addf"(%0, %1) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> (f32)
  %6 = "arith.addf"(%2, %3) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> (f32)
  %7 = "arith.addf"(%4, %5) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> (f32)
  %8 = "arith.addf"(%2, %4) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> (f32)
  %9 = "arith.addf"(%6, %7) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> (f32)
  %10 = "arith.addf"(%7, %8) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> (f32)
  %11 = "arith.addf"(%9, %10) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> (f32)
  func.return %11 : f32
}) : () -> ()

"func.func"() <{sym_name = "different_ops", function_type = () -> (i32, i32)}> ({
  %0 = "arith.constant"() <{value = 0 : i32}> : () -> (i32)
  %1 = "arith.constant"() <{value = 1 : i32}> : () -> (i32)
  func.return %0, %1 : i32, i32
}) : () -> ()

// TODO: Requires memref.cast 
// "func.func"() <{sym_name = "different_results", function_type = (memref<*xf32>) -> (memref<?x?xf32>, memref<4x?xf32>)}> ({
// ^0(%arg0 : memref<*xf32>):
//   %0 = "memref.cast"(%arg0) : (memref<*xf32>) -> memref<?x?xf32>
//   %1 = "memref.cast"(%arg0) : (memref<*xf32>) -> memref<4x?xf32>
//   "func.return"(%0, %1) : (memref<?x?xf32>, memref<4x?xf32>) -> ()
// }) : () -> ()

"func.func"() <{sym_name = "different_attributes", function_type = (index, index) -> (i1, i1, i1)}> ({
^0(%arg0 : index, %arg1 : index):
  %0 = "arith.cmpi"(%arg0, %arg1) <{predicate = 2 : i64}> : (index, index) -> i1
  %1 = "arith.cmpi"(%arg0, %arg1) <{predicate = 1 : i64}> : (index, index) -> i1
  %2 = "arith.cmpi"(%arg0, %arg1) <{predicate = 1 : i64}> : (index, index) -> i1
  "func.return"(%0, %1, %2) : (i1, i1, i1) -> ()
}) : () -> ()

// TODO: Requires more effects infrastructure
// "func.func"() <{sym_name = "side_effect", function_type = () -> (memref<2x1xf32>, memref<2x1xf32>)}> ({
//   %0 = "memref.alloc"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<2x1xf32>
//   %1 = "memref.alloc"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<2x1xf32>
//   "func.return"(%0, %1) : (memref<2x1xf32>, memref<2x1xf32>) -> ()
// }) : () -> ()
// "func.func"() <{sym_name = "down_propagate_for", function_type = () -> ()}> ({
//   %0 = "arith.constant"() <{value = 1 : i32}> : () -> i32
//   "affine.for"() <{lowerBoundMap = affine_map<() -> (0)>, operandSegmentSizes = array<i32: 0, 0, 0>, step = 1 : index, upperBoundMap = affine_map<() -> (4)>}> ({
//   ^0(%arg0 : index):
//     %1 = "arith.constant"() <{value = 1 : i32}> : () -> i32
//     "foo"(%0, %1) : (i32, i32) -> ()
//     "affine.yield"() : () -> ()
//   }) : () -> ()
//   "func.return"() : () -> ()
// }) : () -> ()
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
// "func.func"() <{sym_name = "up_propagate_for", function_type = () -> i32}> ({
//   "affine.for"() <{lowerBoundMap = affine_map<() -> (0)>, operandSegmentSizes = array<i32: 0, 0, 0>, step = 1 : index, upperBoundMap = affine_map<() -> (4)>}> ({
//   ^0(%arg0 : index):
//     %0 = "arith.constant"() <{value = 1 : i32}> : () -> i32
//     "foo"(%0) : (i32) -> ()
//     "affine.yield"() : () -> ()
//   }) : () -> ()
//   %1 = "arith.constant"() <{value = 1 : i32}> : () -> i32
//   "func.return"(%1) : (i32) -> ()
// }) : () -> ()
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
// "func.func"() <{sym_name = "nested_isolated", function_type = () -> i32}> ({
//   %0 = "arith.constant"() <{value = 1 : i32}> : () -> i32
//   "func.func"() <{sym_name = "nested_func", function_type = () -> ()}> ({
//     %1 = "arith.constant"() <{value = 1 : i32}> : () -> i32
//     "foo.yield"(%1) : (i32) -> ()
//   }) : () -> ()
//   "foo.region"() ({
//     %1 = "arith.constant"() <{value = 1 : i32}> : () -> i32
//     "foo.yield"(%1) : (i32) -> ()
//   }) : () -> ()
//   "func.return"(%0) : (i32) -> ()
// }) : () -> ()
// "func.func"() <{sym_name = "use_before_def", function_type = () -> ()}> ({
//   "test.graph_region"() ({
//     %0 = "arith.addi"(%1, %2) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
//     %1 = "arith.constant"() <{value = 1 : i32}> : () -> i32
//     %2 = "arith.constant"() <{value = 1 : i32}> : () -> i32
//     "foo.yield"(%0) : (i32) -> ()
//   }) : () -> ()
//   "func.return"() : () -> ()
// }) : () -> ()
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

// CHECK:       builtin.module {
// CHECK-NEXT:    "func.func"() <{sym_name = "simple_constant", function_type = () -> (i32, i32)}> ({
// CHECK-NEXT:      %0 = "arith.constant"() <{value = 1 : i32}> : () -> (i32)
// CHECK-NEXT:      func.return %0, %0 : i32, i32
// CHECK-NEXT:    }) : () -> ()
// CHECK-NEXT:    "func.func"() <{sym_name = "simple_float_constant", function_type = () -> (f32, f32)}> ({
// CHECK-NEXT:      %0 = "arith.constant"() <{value = 1.0 : f32}> : () -> (f32)
// CHECK-NEXT:      func.return %0, %0 : f32, f32
// CHECK-NEXT:    }) : () -> ()
// CHECK-NEXT:    "func.func"() <{sym_name = "basic", function_type = () -> (index, index)}> ({
// CHECK-NEXT:      %0 = "arith.constant"() <{value = 0 : index}> : () -> (index)
// CHECK-NEXT:      %1 = "affine.apply"(%0) <{map = affine_map<(d0)[] -> (d0 mod 2)>}> : (index) -> (index)
// CHECK-NEXT:      func.return %1, %1 : index, index
// CHECK-NEXT:    }) : () -> ()
// CHECK-NEXT:    "func.func"() <{sym_name = "many", function_type = (f32, f32) -> f32}> ({
// CHECK-NEXT:    ^bb0(%0: f32, %1: f32):
// CHECK-NEXT:      %2 = "arith.addf"(%0, %1) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> (f32)
// CHECK-NEXT:      %3 = "arith.addf"(%2, %2) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> (f32)
// CHECK-NEXT:      %4 = "arith.addf"(%3, %3) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> (f32)
// CHECK-NEXT:      %5 = "arith.addf"(%4, %4) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> (f32)
// CHECK-NEXT:      func.return %5 : f32
// CHECK-NEXT:    }) : () -> ()
// CHECK-NEXT:    "func.func"() <{sym_name = "different_ops", function_type = () -> (i32, i32)}> ({
// CHECK-NEXT:      %0 = "arith.constant"() <{value = 0 : i32}> : () -> (i32)
// CHECK-NEXT:      %1 = "arith.constant"() <{value = 1 : i32}> : () -> (i32)
// CHECK-NEXT:      func.return %0, %1 : i32, i32
// CHECK-NEXT:    }) : () -> ()
// CHECK-NEXT:    "func.func"() <{sym_name = "different_attributes", function_type = (index, index) -> (i1, i1, i1)}> ({
// CHECK-NEXT:    ^bb0(%0: index, %1: index):
// CHECK-NEXT:      %2 = "arith.cmpi"(%0, %1) <{predicate = 2}> : (index, index) -> (i1)
// CHECK-NEXT:      %3 = "arith.cmpi"(%0, %1) <{predicate = 1}> : (index, index) -> (i1)
// CHECK-NEXT:      func.return %2, %3, %3 : i1, i1, i1
// CHECK-NEXT:    }) : () -> ()
// CHECK-NEXT:  }