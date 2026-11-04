// RUN: scair-opt %s | filecheck %s

"builtin.module"() ({
  %0, %1, %2, %3, %4 = "test.op"() : () -> (!llvm.ptr, i32, i8, i64, f64)
  %5 = "llvm.load"(%0) <{ordering = 0 : i64}> : (!llvm.ptr) -> i32
  %6 = "llvm.load"(%0) <{ordering = 0 : i64}> : (!llvm.ptr) -> i64
  %7 = "llvm.getelementptr"(%0, %5) <{rawConstantIndices = array<i32: -2147483648>, elem_type = i32, noWrapFlags = 0 : i32}> : (!llvm.ptr, i32) -> !llvm.ptr
  "llvm.store"(%1, %7) <{ordering = 0 : i64}> : (i32, !llvm.ptr) -> ()
  %8 = "llvm.mlir.constant"() <{value = 7 : i32}> : () -> i32
  %9 = "llvm.mlir.constant"() <{value = 1 : i64}> : () -> i64
  %10 = "llvm.mlir.zero"() : () -> !llvm.ptr
  %11 = "llvm.add"(%1, %8) <{overflowFlags = 0 : i32}> : (i32, i32) -> i32
  %12 = "llvm.mul"(%11, %8) <{overflowFlags = 0 : i32}> : (i32, i32) -> i32
  %13 = "llvm.ptrtoint"(%10) : (!llvm.ptr) -> i64
  %14 = "llvm.inttoptr"(%13) : (i64) -> !llvm.ptr
  %15 = "llvm.mlir.constant"() <{value = 1.25 : f32}> : () -> f32
  %16 = "llvm.mlir.constant"() <{value = 2.5 : f32}> : () -> f32
  %17 = "llvm.fadd"(%15, %16) <{fastmathFlags = #llvm.fastmath<none>}> : (f32, f32) -> f32
  %18 = "llvm.fmul"(%17, %15) <{fastmathFlags = #llvm.fastmath<none>}> : (f32, f32) -> f32
  %19 = "llvm.mlir.poison"() : () -> !llvm.struct<(i32, !llvm.array<2 x i32>)>
  %20 = "llvm.insertvalue"(%19, %8) <{position = array<i64: 0>}> : (!llvm.struct<(i32, !llvm.array<2 x i32>)>, i32) -> !llvm.struct<(i32, !llvm.array<2 x i32>)>
  %21 = "llvm.extractvalue"(%20) <{position = array<i64: 0>}> : (!llvm.struct<(i32, !llvm.array<2 x i32>)>) -> i32
  %22 = "llvm.sub"(%1, %8) <{overflowFlags = 0 : i32}> : (i32, i32) -> i32
  %23 = "llvm.sdiv"(%1, %8) : (i32, i32) -> i32
  %24 = "llvm.srem"(%1, %8) : (i32, i32) -> i32
  %25 = "llvm.fsub"(%15, %16) <{fastmathFlags = #llvm.fastmath<none>}> : (f32, f32) -> f32
  %26 = "llvm.fdiv"(%15, %16) <{fastmathFlags = #llvm.fastmath<none>}> : (f32, f32) -> f32
  %27 = "llvm.frem"(%15, %16) <{fastmathFlags = #llvm.fastmath<none>}> : (f32, f32) -> f32
  %28 = "llvm.fneg"(%15) <{fastmathFlags = #llvm.fastmath<none>}> : (f32) -> f32
  %29 = "llvm.fcmp"(%15, %16) <{predicate = 1 : i64, fastmathFlags = #llvm.fastmath<none>}> : (f32, f32) -> i1
  %30 = "llvm.and"(%1, %8) : (i32, i32) -> i32
  %31 = "llvm.or"(%1, %8) : (i32, i32) -> i32
  %32 = "llvm.xor"(%1, %8) : (i32, i32) -> i32
  %33 = "llvm.shl"(%1, %8) <{overflowFlags = 0 : i32}> : (i32, i32) -> i32
  %34 = "llvm.lshr"(%1, %8) : (i32, i32) -> i32
  %35 = "llvm.ashr"(%1, %8) : (i32, i32) -> i32
  %36 = "llvm.select"(%29, %1, %8) <{fastmathFlags = #llvm.fastmath<none>}> : (i1, i32, i32) -> i32
  %37 = "llvm.trunc"(%1) <{overflowFlags = 0 : i32}> : (i32) -> i8
  %38 = "llvm.zext"(%2) : (i8) -> i64
  %39 = "llvm.sext"(%2) : (i8) -> i64
  %40 = "llvm.sitofp"(%1) : (i32) -> f32
  %41 = "llvm.fptosi"(%15) : (f32) -> i32
  %42 = "llvm.fptrunc"(%4) : (f64) -> f32
  %43 = "llvm.fpext"(%15) : (f32) -> f64
  %44 = "llvm.alloca"(%8) <{elem_type = i32}> : (i32) -> !llvm.ptr
  "llvm.call"(%0) <{CConv = #llvm.cconv<ccc>, op_bundle_sizes = array<i32>, operandSegmentSizes = array<i32: 1, 0>, fastmathFlags = #llvm.fastmath<none>, TailCallKind = #llvm.tailcallkind<none>}> : (!llvm.ptr) -> ()
  "builtin.module"() ({
    "llvm.func"() <{CConv = #llvm.cconv<ccc>, linkage = #llvm.linkage<external>, sym_name = "callee", visibility_ = 0 : i64, function_type = !llvm.func<i32 (i32)>}> ({
    ^bb0(%45: i32):
      %46 = "llvm.mlir.constant"() <{value = 0 : i32}> : () -> i32
      %47 = "llvm.icmp"(%45, %46) <{predicate = 2 : i64}> : (i32, i32) -> i1
      "llvm.cond_br"(%47)[^bb1, ^bb2] <{operandSegmentSizes = array<i32: 1, 0, 0>}> : (i1) -> ()
    ^bb1:
      "llvm.br"()[^bb3] : () -> ()
    ^bb2:
      "llvm.br"()[^bb3] : () -> ()
    ^bb3:
      %48 = "llvm.mlir.constant"() <{value = 1 : i32}> : () -> i32
      "llvm.return"(%48) : (i32) -> ()
    }) : () -> ()
    "llvm.func"() <{CConv = #llvm.cconv<ccc>, linkage = #llvm.linkage<external>, sym_name = "caller", visibility_ = 0 : i64, function_type = !llvm.func<i32 ()>}> ({
      %45 = "llvm.mlir.constant"() <{value = 9 : i32}> : () -> i32
      %46 = "llvm.call"(%45) <{CConv = #llvm.cconv<ccc>, op_bundle_sizes = array<i32>, fastmathFlags = #llvm.fastmath<none>, TailCallKind = #llvm.tailcallkind<none>, operandSegmentSizes = array<i32: 1, 0>, callee = @callee}> : (i32) -> i32
      "llvm.return"(%46) : (i32) -> ()
    }) : () -> ()
  }) : () -> ()
  "llvm.func"() <{CConv = #llvm.cconv<ccc>, linkage = #llvm.linkage<external>, sym_name = "flagged", dso_local, visibility_ = 0 : i64, no_inline, arm_streaming, function_type = !llvm.func<!llvm.void ()>}> ({
    "llvm.return"() : () -> ()
  }) : () -> ()
  "func.func"() <{sym_name = "ops", function_type = (i32, !llvm.ptr) -> ()}> ({
  ^bb0(%45: i32, %46: !llvm.ptr):
    %47 = "llvm.load"(%46) <{invariantGroup, volatile_, nontemporal, ordering = 0 : i64, invariant}> : (!llvm.ptr) -> i32
    %48 = "llvm.load"(%46) <{ordering = 0 : i64}> : (!llvm.ptr) -> i32
    "llvm.store"(%45, %46) <{ordering = 0 : i64, volatile_, nontemporal, invariantGroup}> : (i32, !llvm.ptr) -> ()
    %49 = "test.op"() : () -> i32
    %50 = "llvm.alloca"(%49) <{elem_type = i32, inalloca}> : (i32) -> !llvm.ptr
    %51 = "llvm.sdiv"(%45, %45) <{isExact}> : (i32, i32) -> i32
    %52 = "llvm.lshr"(%45, %45) <{isExact}> : (i32, i32) -> i32
    %53 = "llvm.ashr"(%45, %45) <{isExact}> : (i32, i32) -> i32
    %54 = "llvm.or"(%45, %45) <{isDisjoint}> : (i32, i32) -> i32
    %55 = "llvm.zext"(%45) <{nonNeg}> : (i32) -> i64
    "llvm.call"() <{CConv = #llvm.cconv<ccc>, op_bundle_sizes = array<i32>, always_inline, will_return, fastmathFlags = #llvm.fastmath<none>, no_unwind, TailCallKind = #llvm.tailcallkind<none>, no_inline, inline_hint, convergent, operandSegmentSizes = array<i32: 0, 0>, callee = @flagged}> : () -> ()
    %56 = "llvm.call"(%45) <{CConv = #llvm.cconv<ccc>, op_bundle_sizes = array<i32>, fastmathFlags = #llvm.fastmath<none>, TailCallKind = #llvm.tailcallkind<none>, operandSegmentSizes = array<i32: 1, 0>, callee = @flagged}> : (i32) -> i32
    "llvm.call"() <{CConv = #llvm.cconv<ccc>, op_bundle_sizes = array<i32>, fastmathFlags = #llvm.fastmath<none>, TailCallKind = #llvm.tailcallkind<none>, no_inline, operandSegmentSizes = array<i32: 0, 0>, callee = @flagged}> {hello = "world"} : () -> ()
    "func.return"() : () -> ()
  }) : () -> ()
}) : () -> ()

// CHECK: builtin.module {
// CHECK-NEXT:   %0, %1, %2, %3, %4 = "test.op"() : () -> (!llvm.ptr, i32, i8, i64, f64)
// CHECK-NEXT:   %5 = "llvm.load"(%0) <{ordering = 0 : i64}> : (!llvm.ptr) -> i32
// CHECK-NEXT:   %6 = "llvm.load"(%0) <{ordering = 0 : i64}> : (!llvm.ptr) -> i64
// CHECK-NEXT:   %7 = "llvm.getelementptr"(%0, %5) <{rawConstantIndices = array<i32: -2147483648>, elem_type = i32, noWrapFlags = 0 : i32}> : (!llvm.ptr, i32) -> !llvm.ptr
// CHECK-NEXT:   "llvm.store"(%1, %7) <{ordering = 0 : i64}> : (i32, !llvm.ptr) -> ()
// CHECK-NEXT:   %8 = "llvm.mlir.constant"() <{value = 7 : i32}> : () -> i32
// CHECK-NEXT:   %9 = "llvm.mlir.constant"() <{value = 1 : i64}> : () -> i64
// CHECK-NEXT:   %10 = "llvm.mlir.zero"() : () -> !llvm.ptr
// CHECK-NEXT:   %11 = "llvm.add"(%1, %8) <{overflowFlags = 0 : i32}> : (i32, i32) -> i32
// CHECK-NEXT:   %12 = "llvm.mul"(%11, %8) <{overflowFlags = 0 : i32}> : (i32, i32) -> i32
// CHECK-NEXT:   %13 = "llvm.ptrtoint"(%10) : (!llvm.ptr) -> i64
// CHECK-NEXT:   %14 = "llvm.inttoptr"(%13) : (i64) -> !llvm.ptr
// CHECK-NEXT:   %15 = "llvm.mlir.constant"() <{value = 1.25 : f32}> : () -> f32
// CHECK-NEXT:   %16 = "llvm.mlir.constant"() <{value = 2.5 : f32}> : () -> f32
// CHECK-NEXT:   %17 = "llvm.fadd"(%15, %16) <{fastmathFlags = #llvm.fastmath<none>}> : (f32, f32) -> f32
// CHECK-NEXT:   %18 = "llvm.fmul"(%17, %15) <{fastmathFlags = #llvm.fastmath<none>}> : (f32, f32) -> f32
// CHECK-NEXT:   %19 = llvm.mlir.poison : !llvm.struct<(i32, !llvm.array<2 x i32>)>
// CHECK-NEXT:   %20 = "llvm.insertvalue"(%19, %8) <{position = array<i64: 0>}> : (!llvm.struct<(i32, !llvm.array<2 x i32>)>, i32) -> !llvm.struct<(i32, !llvm.array<2 x i32>)>
// CHECK-NEXT:   %21 = "llvm.extractvalue"(%20) <{position = array<i64: 0>}> : (!llvm.struct<(i32, !llvm.array<2 x i32>)>) -> i32
// CHECK-NEXT:   %22 = "llvm.sub"(%1, %8) <{overflowFlags = 0 : i32}> : (i32, i32) -> i32
// CHECK-NEXT:   %23 = "llvm.sdiv"(%1, %8) : (i32, i32) -> i32
// CHECK-NEXT:   %24 = "llvm.srem"(%1, %8) : (i32, i32) -> i32
// CHECK-NEXT:   %25 = "llvm.fsub"(%15, %16) <{fastmathFlags = #llvm.fastmath<none>}> : (f32, f32) -> f32
// CHECK-NEXT:   %26 = "llvm.fdiv"(%15, %16) <{fastmathFlags = #llvm.fastmath<none>}> : (f32, f32) -> f32
// CHECK-NEXT:   %27 = "llvm.frem"(%15, %16) <{fastmathFlags = #llvm.fastmath<none>}> : (f32, f32) -> f32
// CHECK-NEXT:   %28 = "llvm.fneg"(%15) <{fastmathFlags = #llvm.fastmath<none>}> : (f32) -> f32
// CHECK-NEXT:   %29 = "llvm.fcmp"(%15, %16) <{predicate = 1 : i64, fastmathFlags = #llvm.fastmath<none>}> : (f32, f32) -> i1
// CHECK-NEXT:   %30 = "llvm.and"(%1, %8) : (i32, i32) -> i32
// CHECK-NEXT:   %31 = "llvm.or"(%1, %8) : (i32, i32) -> i32
// CHECK-NEXT:   %32 = "llvm.xor"(%1, %8) : (i32, i32) -> i32
// CHECK-NEXT:   %33 = "llvm.shl"(%1, %8) <{overflowFlags = 0 : i32}> : (i32, i32) -> i32
// CHECK-NEXT:   %34 = "llvm.lshr"(%1, %8) : (i32, i32) -> i32
// CHECK-NEXT:   %35 = "llvm.ashr"(%1, %8) : (i32, i32) -> i32
// CHECK-NEXT:   %36 = "llvm.select"(%29, %1, %8) <{fastmathFlags = #llvm.fastmath<none>}> : (i1, i32, i32) -> i32
// CHECK-NEXT:   %37 = "llvm.trunc"(%1) <{overflowFlags = 0 : i32}> : (i32) -> i8
// CHECK-NEXT:   %38 = "llvm.zext"(%2) : (i8) -> i64
// CHECK-NEXT:   %39 = "llvm.sext"(%2) : (i8) -> i64
// CHECK-NEXT:   %40 = "llvm.sitofp"(%1) : (i32) -> f32
// CHECK-NEXT:   %41 = "llvm.fptosi"(%15) : (f32) -> i32
// CHECK-NEXT:   %42 = "llvm.fptrunc"(%4) : (f64) -> f32
// CHECK-NEXT:   %43 = "llvm.fpext"(%15) : (f32) -> f64
// CHECK-NEXT:   %44 = "llvm.alloca"(%8) <{elem_type = i32}> : (i32) -> !llvm.ptr
// CHECK-NEXT:   "llvm.call"(%0) <{CConv = #llvm.cconv<ccc>, op_bundle_sizes = array<i32>, operandSegmentSizes = array<i32: 1, 0>, fastmathFlags = #llvm.fastmath<none>, TailCallKind = #llvm.tailcallkind<none>}> : (!llvm.ptr) -> ()
// CHECK-NEXT:   builtin.module {
// CHECK-NEXT:     "llvm.func"() <{CConv = #llvm.cconv<ccc>, linkage = #llvm.linkage<external>, sym_name = "callee", visibility_ = 0 : i64, function_type = !llvm.func<i32 (i32)>}> ({
// CHECK-NEXT:     ^bb0(%45: i32):
// CHECK-NEXT:       %46 = "llvm.mlir.constant"() <{value = 0 : i32}> : () -> i32
// CHECK-NEXT:       %47 = "llvm.icmp"(%45, %46) <{predicate = 2 : i64}> : (i32, i32) -> i1
// CHECK-NEXT:       "llvm.cond_br"(%47)[^bb1, ^bb2] <{operandSegmentSizes = array<i32: 1, 0, 0>}> : (i1) -> ()
// CHECK-NEXT:     ^bb1:
// CHECK-NEXT:       "llvm.br"()[^bb3] : () -> ()
// CHECK-NEXT:     ^bb2:
// CHECK-NEXT:       "llvm.br"()[^bb3] : () -> ()
// CHECK-NEXT:     ^bb3:
// CHECK-NEXT:       %48 = "llvm.mlir.constant"() <{value = 1 : i32}> : () -> i32
// CHECK-NEXT:       "llvm.return"(%48) : (i32) -> ()
// CHECK-NEXT:     }) : () -> ()
// CHECK-NEXT:     "llvm.func"() <{CConv = #llvm.cconv<ccc>, linkage = #llvm.linkage<external>, sym_name = "caller", visibility_ = 0 : i64, function_type = !llvm.func<i32 ()>}> ({
// CHECK-NEXT:       %45 = "llvm.mlir.constant"() <{value = 9 : i32}> : () -> i32
// CHECK-NEXT:       %46 = "llvm.call"(%45) <{CConv = #llvm.cconv<ccc>, op_bundle_sizes = array<i32>, fastmathFlags = #llvm.fastmath<none>, TailCallKind = #llvm.tailcallkind<none>, operandSegmentSizes = array<i32: 1, 0>, callee = @callee}> : (i32) -> i32
// CHECK-NEXT:       "llvm.return"(%46) : (i32) -> ()
// CHECK-NEXT:     }) : () -> ()
// CHECK-NEXT:   }
// CHECK-NEXT:   "llvm.func"() <{CConv = #llvm.cconv<ccc>, linkage = #llvm.linkage<external>, sym_name = "flagged", dso_local, visibility_ = 0 : i64, no_inline, arm_streaming, function_type = !llvm.func<!llvm.void ()>}> ({
// CHECK-NEXT:     "llvm.return"() : () -> ()
// CHECK-NEXT:   }) : () -> ()
// CHECK-NEXT:   func.func @ops(%45: i32, %46: !llvm.ptr) {
// CHECK-NEXT:     %47 = "llvm.load"(%46) <{invariantGroup, volatile_, nontemporal, ordering = 0 : i64, invariant}> : (!llvm.ptr) -> i32
// CHECK-NEXT:     %48 = "llvm.load"(%46) <{ordering = 0 : i64}> : (!llvm.ptr) -> i32
// CHECK-NEXT:     "llvm.store"(%45, %46) <{ordering = 0 : i64, volatile_, nontemporal, invariantGroup}> : (i32, !llvm.ptr) -> ()
// CHECK-NEXT:     %49 = "test.op"() : () -> i32
// CHECK-NEXT:     %50 = "llvm.alloca"(%49) <{elem_type = i32, inalloca}> : (i32) -> !llvm.ptr
// CHECK-NEXT:     %51 = "llvm.sdiv"(%45, %45) <{isExact}> : (i32, i32) -> i32
// CHECK-NEXT:     %52 = "llvm.lshr"(%45, %45) <{isExact}> : (i32, i32) -> i32
// CHECK-NEXT:     %53 = "llvm.ashr"(%45, %45) <{isExact}> : (i32, i32) -> i32
// CHECK-NEXT:     %54 = "llvm.or"(%45, %45) <{isDisjoint}> : (i32, i32) -> i32
// CHECK-NEXT:     %55 = "llvm.zext"(%45) <{nonNeg}> : (i32) -> i64
// CHECK-NEXT:     "llvm.call"() <{CConv = #llvm.cconv<ccc>, op_bundle_sizes = array<i32>, always_inline, will_return, fastmathFlags = #llvm.fastmath<none>, no_unwind, TailCallKind = #llvm.tailcallkind<none>, no_inline, inline_hint, convergent, operandSegmentSizes = array<i32: 0, 0>, callee = @flagged}> : () -> ()
// CHECK-NEXT:     %56 = "llvm.call"(%45) <{CConv = #llvm.cconv<ccc>, op_bundle_sizes = array<i32>, fastmathFlags = #llvm.fastmath<none>, TailCallKind = #llvm.tailcallkind<none>, operandSegmentSizes = array<i32: 1, 0>, callee = @flagged}> : (i32) -> i32
// CHECK-NEXT:     "llvm.call"() <{CConv = #llvm.cconv<ccc>, op_bundle_sizes = array<i32>, fastmathFlags = #llvm.fastmath<none>, TailCallKind = #llvm.tailcallkind<none>, no_inline, operandSegmentSizes = array<i32: 0, 0>, callee = @flagged}> {hello = "world"} : () -> ()
// CHECK-NEXT:     func.return
// CHECK-NEXT:   }
// CHECK-NEXT: }
