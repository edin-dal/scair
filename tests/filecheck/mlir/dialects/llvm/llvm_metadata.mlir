// RUN: mlir-opt %s --mlir-print-op-generic | scair-opt | scair-opt | mlir-opt --mlir-print-op-generic | filecheck %s
// RUN: scair-opt %s | mlir-opt --mlir-print-op-generic | filecheck %s

"builtin.module"() ({
  "llvm.func"() <{CConv = #llvm.cconv<fastcc>, arg_attrs = [{llvm.signext}], frame_pointer = #llvm.framePointerKind<"non-leaf">, function_type = !llvm.func<void (i32)>, linkage = #llvm.linkage<internal>, memory_effects = #llvm.memory_effects<other = none, argMem = read, inaccessibleMem = write, errnoMem = none, targetMem0 = readwrite, targetMem1 = none>, no_infs_fp_math = true, no_inline, no_nans_fp_math = false, no_signed_zeros_fp_math = true, optimize_none, sym_name = "metadata", target_cpu = "generic", target_features = #llvm.target_features<["+sse2"]>, unnamed_addr = 2 : i64, uwtable_kind = #llvm.uwtableKind<async>, vec_type_hint = #llvm.vec_type_hint<hint = vector<4xi32>, is_signed = true>, visibility_ = 1 : i64, vscale_range = #llvm.vscale_range<minRange = 1 : i32, maxRange = 8 : i32>}> ({
  ^bb0(%arg0: i32):
    "llvm.call"(%arg0, %arg0) <{CConv = #llvm.cconv<ccc>, TailCallKind = #llvm.tailcallkind<none>, callee = @bundled, fastmathFlags = #llvm.fastmath<none>, op_bundle_sizes = array<i32: 1>, op_bundle_tags = ["deopt"], operandSegmentSizes = array<i32: 1, 1>}> : (i32, i32) -> ()
    "llvm.call"() <{CConv = #llvm.cconv<ccc>, TailCallKind = #llvm.tailcallkind<none>, callee = @empty, fastmathFlags = #llvm.fastmath<none>, op_bundle_sizes = array<i32: 0>, op_bundle_tags = ["deopt"], operandSegmentSizes = array<i32: 0, 0>}> : () -> ()
    "llvm.return"() : () -> ()
  }) : () -> ()
  "llvm.func"() <{CConv = #llvm.cconv<ccc>, function_type = !llvm.func<void (i32)>, linkage = #llvm.linkage<external>, sym_name = "bundled", unnamed_addr = 0 : i64, visibility_ = 0 : i64}> ({
  }) : () -> ()
  "llvm.func"() <{CConv = #llvm.cconv<ccc>, function_type = !llvm.func<void ()>, linkage = #llvm.linkage<external>, sym_name = "empty", unnamed_addr = 0 : i64, visibility_ = 0 : i64}> ({
  }) : () -> ()
}) : () -> ()


// CHECK: CConv = #llvm.cconv<fastcc>
// CHECK-SAME: arg_attrs = [{llvm.signext}]
// CHECK-SAME: frame_pointer = #llvm.framePointerKind<"non-leaf">
// CHECK-SAME: memory_effects = #llvm.memory_effects<other = none, argMem = read, inaccessibleMem = write, errnoMem = none, targetMem0 = readwrite, targetMem1 = none>
// CHECK-SAME: target_features = #llvm.target_features<["+sse2"]>
// CHECK-SAME: uwtable_kind = #llvm.uwtableKind<async>
// CHECK-SAME: vec_type_hint = #llvm.vec_type_hint<hint = vector<4xi32>, is_signed = true>
// CHECK-SAME: vscale_range = #llvm.vscale_range<minRange = 1 : i32, maxRange = 8 : i32>
// CHECK: op_bundle_sizes = array<i32: 1>, op_bundle_tags = ["deopt"]
// CHECK: op_bundle_sizes = array<i32: 0>, op_bundle_tags = ["deopt"]
