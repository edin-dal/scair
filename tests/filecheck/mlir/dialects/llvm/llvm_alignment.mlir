// RUN: scair-opt %s | mlir-opt --mlir-print-op-generic | scair-opt | scair-opt | mlir-opt --mlir-print-op-generic | filecheck %s
// RUN: mlir-opt %s --mlir-print-op-generic | scair-opt | mlir-opt --mlir-print-op-generic | filecheck %s

"builtin.module"() ({
  "llvm.func"() <{CConv = #llvm.cconv<ccc>, function_type = !llvm.func<i32 (i32)>, linkage = #llvm.linkage<external>, sym_name = "callee", unnamed_addr = 0 : i64, visibility_ = 0 : i64}> ({
  }) : () -> ()
  "llvm.func"() <{CConv = #llvm.cconv<ccc>, function_type = !llvm.func<void ()>, linkage = #llvm.linkage<external>, sym_name = "void_callee", unnamed_addr = 0 : i64, visibility_ = 0 : i64}> ({
  }) : () -> ()
  "llvm.func"() <{CConv = #llvm.cconv<ccc>, function_type = !llvm.func<void (i32, ...)>, linkage = #llvm.linkage<external>, sym_name = "vararg", unnamed_addr = 0 : i64, visibility_ = 0 : i64}> ({
  }) : () -> ()
  "llvm.func"() <{CConv = #llvm.cconv<ccc>, function_type = !llvm.func<void (i32, i32, f32, f32, f64, ptr, ptr<3>, i1, vector<4xi32>, vector<4xf32>, vector<4x!llvm.ptr>, vector<4xi1>, vector<[4]xi32>, vector<[4]x!llvm.ptr>, vector<[4]xf32>, vector<[4]xi1>)>, linkage = #llvm.linkage<external>, sym_name = "operations", unnamed_addr = 0 : i64, visibility_ = 0 : i64}> ({
  ^bb0(%arg0: i32, %arg1: i32, %arg2: f32, %arg3: f32, %arg4: f64, %arg5: !llvm.ptr, %arg6: !llvm.ptr<3>, %arg7: i1, %arg8: vector<4xi32>, %arg9: vector<4xf32>, %arg10: vector<4x!llvm.ptr>, %arg11: vector<4xi1>, %arg12: vector<[4]xi32>, %arg13: vector<[4]x!llvm.ptr>, %arg14: vector<[4]xf32>, %arg15: vector<[4]xi1>):
    %0 = "llvm.mlir.constant"() <{value = 42 : i32}> : () -> i32
    %1 = "llvm.mlir.zero"() : () -> !llvm.ptr
    %2 = "llvm.mlir.poison"() : () -> !llvm.struct<(i32, array<2 x i32>)>
    %3 = "llvm.add"(%arg0, %arg0) <{overflowFlags = 3 : i32}> : (i32, i32) -> i32
    %4 = "llvm.add"(%arg8, %arg8) <{overflowFlags = 3 : i32}> : (vector<4xi32>, vector<4xi32>) -> vector<4xi32>
    %5 = "llvm.add"(%arg12, %arg12) <{overflowFlags = 3 : i32}> : (vector<[4]xi32>, vector<[4]xi32>) -> vector<[4]xi32>
    %6 = "llvm.sub"(%arg0, %arg0) <{overflowFlags = 3 : i32}> : (i32, i32) -> i32
    %7 = "llvm.sub"(%arg8, %arg8) <{overflowFlags = 3 : i32}> : (vector<4xi32>, vector<4xi32>) -> vector<4xi32>
    %8 = "llvm.sub"(%arg12, %arg12) <{overflowFlags = 3 : i32}> : (vector<[4]xi32>, vector<[4]xi32>) -> vector<[4]xi32>
    %9 = "llvm.mul"(%arg0, %arg0) <{overflowFlags = 3 : i32}> : (i32, i32) -> i32
    %10 = "llvm.mul"(%arg8, %arg8) <{overflowFlags = 3 : i32}> : (vector<4xi32>, vector<4xi32>) -> vector<4xi32>
    %11 = "llvm.mul"(%arg12, %arg12) <{overflowFlags = 3 : i32}> : (vector<[4]xi32>, vector<[4]xi32>) -> vector<[4]xi32>
    %12 = "llvm.sdiv"(%arg0, %arg0) <{isExact}> : (i32, i32) -> i32
    %13 = "llvm.sdiv"(%arg8, %arg8) <{isExact}> : (vector<4xi32>, vector<4xi32>) -> vector<4xi32>
    %14 = "llvm.sdiv"(%arg12, %arg12) <{isExact}> : (vector<[4]xi32>, vector<[4]xi32>) -> vector<[4]xi32>
    %15 = "llvm.srem"(%arg0, %arg0) : (i32, i32) -> i32
    %16 = "llvm.srem"(%arg8, %arg8) : (vector<4xi32>, vector<4xi32>) -> vector<4xi32>
    %17 = "llvm.srem"(%arg12, %arg12) : (vector<[4]xi32>, vector<[4]xi32>) -> vector<[4]xi32>
    %18 = "llvm.and"(%arg0, %arg0) : (i32, i32) -> i32
    %19 = "llvm.and"(%arg8, %arg8) : (vector<4xi32>, vector<4xi32>) -> vector<4xi32>
    %20 = "llvm.and"(%arg12, %arg12) : (vector<[4]xi32>, vector<[4]xi32>) -> vector<[4]xi32>
    %21 = "llvm.or"(%arg0, %arg0) <{isDisjoint}> : (i32, i32) -> i32
    %22 = "llvm.or"(%arg8, %arg8) <{isDisjoint}> : (vector<4xi32>, vector<4xi32>) -> vector<4xi32>
    %23 = "llvm.or"(%arg12, %arg12) <{isDisjoint}> : (vector<[4]xi32>, vector<[4]xi32>) -> vector<[4]xi32>
    %24 = "llvm.xor"(%arg0, %arg0) : (i32, i32) -> i32
    %25 = "llvm.xor"(%arg8, %arg8) : (vector<4xi32>, vector<4xi32>) -> vector<4xi32>
    %26 = "llvm.xor"(%arg12, %arg12) : (vector<[4]xi32>, vector<[4]xi32>) -> vector<[4]xi32>
    %27 = "llvm.shl"(%arg0, %arg0) <{overflowFlags = 3 : i32}> : (i32, i32) -> i32
    %28 = "llvm.shl"(%arg8, %arg8) <{overflowFlags = 3 : i32}> : (vector<4xi32>, vector<4xi32>) -> vector<4xi32>
    %29 = "llvm.shl"(%arg12, %arg12) <{overflowFlags = 3 : i32}> : (vector<[4]xi32>, vector<[4]xi32>) -> vector<[4]xi32>
    %30 = "llvm.lshr"(%arg0, %arg0) <{isExact}> : (i32, i32) -> i32
    %31 = "llvm.lshr"(%arg8, %arg8) <{isExact}> : (vector<4xi32>, vector<4xi32>) -> vector<4xi32>
    %32 = "llvm.lshr"(%arg12, %arg12) <{isExact}> : (vector<[4]xi32>, vector<[4]xi32>) -> vector<[4]xi32>
    %33 = "llvm.ashr"(%arg0, %arg0) <{isExact}> : (i32, i32) -> i32
    %34 = "llvm.ashr"(%arg8, %arg8) <{isExact}> : (vector<4xi32>, vector<4xi32>) -> vector<4xi32>
    %35 = "llvm.ashr"(%arg12, %arg12) <{isExact}> : (vector<[4]xi32>, vector<[4]xi32>) -> vector<[4]xi32>
    %36 = "llvm.fadd"(%arg2, %arg2) <{fastmathFlags = #llvm.fastmath<nnan, contract>}> : (f32, f32) -> f32
    %37 = "llvm.fadd"(%arg9, %arg9) <{fastmathFlags = #llvm.fastmath<nnan, contract>}> : (vector<4xf32>, vector<4xf32>) -> vector<4xf32>
    %38 = "llvm.fadd"(%arg14, %arg14) <{fastmathFlags = #llvm.fastmath<nnan, contract>}> : (vector<[4]xf32>, vector<[4]xf32>) -> vector<[4]xf32>
    %39 = "llvm.fsub"(%arg2, %arg2) <{fastmathFlags = #llvm.fastmath<nnan, contract>}> : (f32, f32) -> f32
    %40 = "llvm.fsub"(%arg9, %arg9) <{fastmathFlags = #llvm.fastmath<nnan, contract>}> : (vector<4xf32>, vector<4xf32>) -> vector<4xf32>
    %41 = "llvm.fsub"(%arg14, %arg14) <{fastmathFlags = #llvm.fastmath<nnan, contract>}> : (vector<[4]xf32>, vector<[4]xf32>) -> vector<[4]xf32>
    %42 = "llvm.fmul"(%arg2, %arg2) <{fastmathFlags = #llvm.fastmath<nnan, contract>}> : (f32, f32) -> f32
    %43 = "llvm.fmul"(%arg9, %arg9) <{fastmathFlags = #llvm.fastmath<nnan, contract>}> : (vector<4xf32>, vector<4xf32>) -> vector<4xf32>
    %44 = "llvm.fmul"(%arg14, %arg14) <{fastmathFlags = #llvm.fastmath<nnan, contract>}> : (vector<[4]xf32>, vector<[4]xf32>) -> vector<[4]xf32>
    %45 = "llvm.fdiv"(%arg2, %arg2) <{fastmathFlags = #llvm.fastmath<nnan, contract>}> : (f32, f32) -> f32
    %46 = "llvm.fdiv"(%arg9, %arg9) <{fastmathFlags = #llvm.fastmath<nnan, contract>}> : (vector<4xf32>, vector<4xf32>) -> vector<4xf32>
    %47 = "llvm.fdiv"(%arg14, %arg14) <{fastmathFlags = #llvm.fastmath<nnan, contract>}> : (vector<[4]xf32>, vector<[4]xf32>) -> vector<[4]xf32>
    %48 = "llvm.frem"(%arg2, %arg2) <{fastmathFlags = #llvm.fastmath<nnan, contract>}> : (f32, f32) -> f32
    %49 = "llvm.frem"(%arg9, %arg9) <{fastmathFlags = #llvm.fastmath<nnan, contract>}> : (vector<4xf32>, vector<4xf32>) -> vector<4xf32>
    %50 = "llvm.frem"(%arg14, %arg14) <{fastmathFlags = #llvm.fastmath<nnan, contract>}> : (vector<[4]xf32>, vector<[4]xf32>) -> vector<[4]xf32>
    %51 = "llvm.fneg"(%arg2) <{fastmathFlags = #llvm.fastmath<nnan, contract>}> : (f32) -> f32
    %52 = "llvm.fneg"(%arg9) <{fastmathFlags = #llvm.fastmath<nnan, contract>}> : (vector<4xf32>) -> vector<4xf32>
    %53 = "llvm.fneg"(%arg14) <{fastmathFlags = #llvm.fastmath<nnan, contract>}> : (vector<[4]xf32>) -> vector<[4]xf32>
    %54 = "llvm.icmp"(%arg0, %arg0) <{predicate = 0 : i64}> {tag = "preserve"} : (i32, i32) -> i1
    %55 = "llvm.icmp"(%arg5, %arg5) <{predicate = 0 : i64}> {tag = "preserve"} : (!llvm.ptr, !llvm.ptr) -> i1
    %56 = "llvm.icmp"(%arg6, %arg6) <{predicate = 0 : i64}> {tag = "preserve"} : (!llvm.ptr<3>, !llvm.ptr<3>) -> i1
    %57 = "llvm.icmp"(%arg8, %arg8) <{predicate = 0 : i64}> {tag = "preserve"} : (vector<4xi32>, vector<4xi32>) -> vector<4xi1>
    %58 = "llvm.icmp"(%arg10, %arg10) <{predicate = 0 : i64}> {tag = "preserve"} : (vector<4x!llvm.ptr>, vector<4x!llvm.ptr>) -> vector<4xi1>
    %59 = "llvm.icmp"(%arg13, %arg13) <{predicate = 0 : i64}> {tag = "preserve"} : (vector<[4]x!llvm.ptr>, vector<[4]x!llvm.ptr>) -> vector<[4]xi1>
    %60 = "llvm.fcmp"(%arg2, %arg2) <{fastmathFlags = #llvm.fastmath<fast>, predicate = 1 : i64}> : (f32, f32) -> i1
    %61 = "llvm.fcmp"(%arg9, %arg9) <{fastmathFlags = #llvm.fastmath<fast>, predicate = 1 : i64}> : (vector<4xf32>, vector<4xf32>) -> vector<4xi1>
    %62 = "llvm.fcmp"(%arg14, %arg14) <{fastmathFlags = #llvm.fastmath<fast>, predicate = 1 : i64}> : (vector<[4]xf32>, vector<[4]xf32>) -> vector<[4]xi1>
    %63 = "llvm.trunc"(%arg0) <{overflowFlags = 1 : i32}> : (i32) -> i8
    %64 = "llvm.trunc"(%arg8) <{overflowFlags = 1 : i32}> : (vector<4xi32>) -> vector<4xi8>
    %65 = "llvm.zext"(%arg0) <{nonNeg}> : (i32) -> i64
    %66 = "llvm.zext"(%arg8) <{nonNeg}> : (vector<4xi32>) -> vector<4xi64>
    %67 = "llvm.sext"(%arg0) : (i32) -> i64
    %68 = "llvm.sext"(%arg8) : (vector<4xi32>) -> vector<4xi64>
    %69 = "llvm.sitofp"(%arg0) : (i32) -> f32
    %70 = "llvm.sitofp"(%arg8) : (vector<4xi32>) -> vector<4xf32>
    %71 = "llvm.fptosi"(%arg2) : (f32) -> i32
    %72 = "llvm.fptosi"(%arg9) : (vector<4xf32>) -> vector<4xi32>
    %73 = "llvm.fptrunc"(%arg4) : (f64) -> f32
    %74 = "llvm.fpext"(%arg2) : (f32) -> f64
    %75 = "llvm.fpext"(%arg9) : (vector<4xf32>) -> vector<4xf64>
    %76 = "llvm.ptrtoint"(%arg6) : (!llvm.ptr<3>) -> i64
    %77 = "llvm.ptrtoint"(%arg10) : (vector<4x!llvm.ptr>) -> vector<4xi64>
    %78 = "llvm.inttoptr"(%arg0) <{dereferenceable = #llvm.dereferenceable<bytes = 8, mayBeNull = true>}> : (i32) -> !llvm.ptr<3>
    %79 = "llvm.inttoptr"(%arg8) : (vector<4xi32>) -> vector<4x!llvm.ptr<3>>
    %80 = "llvm.load"(%arg5) <{alignment = 4 : i64, invariant, invariantGroup, nontemporal, ordering = 0 : i64, volatile_}> : (!llvm.ptr) -> i32
    %81 = "llvm.load"(%arg5) <{alignment = 4 : i64, ordering = 4 : i64, syncscope = "singlethread"}> : (!llvm.ptr) -> i32
    "llvm.store"(%arg0, %arg5) <{alignment = 4 : i64, invariantGroup, nontemporal, ordering = 5 : i64, syncscope = "singlethread", volatile_}> : (i32, !llvm.ptr) -> ()
    %82 = "llvm.alloca"(%arg0) <{alignment = 8 : i64, elem_type = i32, inalloca}> : (i32) -> !llvm.ptr<3>
    %83 = "llvm.getelementptr"(%arg5, %arg0) <{elem_type = i32, noWrapFlags = 3 : i32, rawConstantIndices = array<i32: -2147483648>}> : (!llvm.ptr, i32) -> !llvm.ptr
    %84 = "llvm.getelementptr"(%arg10, %arg8) <{elem_type = i32, noWrapFlags = 0 : i32, rawConstantIndices = array<i32: -2147483648>}> : (vector<4x!llvm.ptr>, vector<4xi32>) -> vector<4x!llvm.ptr>
    %85 = "llvm.insertvalue"(%2, %arg0) <{position = array<i64: 0>}> {tag = "preserve"} : (!llvm.struct<(i32, array<2 x i32>)>, i32) -> !llvm.struct<(i32, array<2 x i32>)>
    %86 = "llvm.extractvalue"(%85) <{position = array<i64: 0>}> {tag = "preserve"} : (!llvm.struct<(i32, array<2 x i32>)>) -> i32
    %87 = "llvm.select"(%arg7, %arg5, %arg5) <{fastmathFlags = #llvm.fastmath<none>}> : (i1, !llvm.ptr, !llvm.ptr) -> !llvm.ptr
    %88 = "llvm.select"(%arg11, %arg8, %arg8) <{fastmathFlags = #llvm.fastmath<none>}> : (vector<4xi1>, vector<4xi32>, vector<4xi32>) -> vector<4xi32>
    %89 = "llvm.call"(%arg0) <{CConv = #llvm.cconv<ccc>, TailCallKind = #llvm.tailcallkind<none>, callee = @callee, fastmathFlags = #llvm.fastmath<none>, op_bundle_sizes = array<i32>, operandSegmentSizes = array<i32: 1, 0>}> : (i32) -> i32
    "llvm.call"() <{CConv = #llvm.cconv<ccc>, TailCallKind = #llvm.tailcallkind<none>, callee = @void_callee, fastmathFlags = #llvm.fastmath<none>, op_bundle_sizes = array<i32>, operandSegmentSizes = array<i32: 0, 0>}> : () -> ()
    %90 = "llvm.call"(%arg5, %arg0) <{CConv = #llvm.cconv<ccc>, TailCallKind = #llvm.tailcallkind<none>, fastmathFlags = #llvm.fastmath<none>, op_bundle_sizes = array<i32>, operandSegmentSizes = array<i32: 2, 0>}> : (!llvm.ptr, i32) -> i32
    "llvm.call"(%arg5) <{CConv = #llvm.cconv<ccc>, TailCallKind = #llvm.tailcallkind<none>, fastmathFlags = #llvm.fastmath<none>, op_bundle_sizes = array<i32>, operandSegmentSizes = array<i32: 1, 0>}> : (!llvm.ptr) -> ()
    "llvm.call"(%arg0, %arg1) <{CConv = #llvm.cconv<ccc>, TailCallKind = #llvm.tailcallkind<none>, callee = @vararg, fastmathFlags = #llvm.fastmath<none>, op_bundle_sizes = array<i32>, operandSegmentSizes = array<i32: 2, 0>, var_callee_type = !llvm.func<void (i32, ...)>}> : (i32, i32) -> ()
    "llvm.cond_br"(%arg7, %arg0, %arg1)[^bb1, ^bb2] <{branch_weights = array<i32: 3, 2>, operandSegmentSizes = array<i32: 1, 1, 1>}> : (i1, i32, i32) -> ()
  ^bb1(%91: i32):  // pred: ^bb0
    "llvm.br"(%91)[^bb3] : (i32) -> ()
  ^bb2(%92: i32):  // pred: ^bb0
    "llvm.br"(%92)[^bb3] : (i32) -> ()
  ^bb3(%93: i32):  // 2 preds: ^bb1, ^bb2
    "llvm.return"() : () -> ()
  }) : () -> ()
  "llvm.func"() <{CConv = #llvm.cconv<ccc>, function_type = !llvm.func<void ()>, linkage = #llvm.linkage<internal>, sym_name = "dead", unnamed_addr = 0 : i64, visibility_ = 0 : i64}> ({
    "llvm.unreachable"() : () -> ()
  }) : () -> ()
}) : () -> ()


// CHECK: "llvm.func"
// CHECK: "llvm.icmp"({{.*}}) <{predicate = 0 : i64}> {tag = "preserve"} : (!llvm.ptr, !llvm.ptr) -> i1
// CHECK: "llvm.icmp"({{.*}}) <{predicate = 0 : i64}> {tag = "preserve"} : (vector<[4]x!llvm.ptr>, vector<[4]x!llvm.ptr>) -> vector<[4]xi1>
// CHECK: "llvm.insertvalue"
// CHECK-SAME: position = array<i64: 0>
// CHECK: "llvm.call"
// CHECK: "llvm.cond_br"
// CHECK: "llvm.unreachable"
