// RUN: scair-opt %s | filecheck %s

%0, %1, %i8val, %i64val, %f64val = "test.op"() : () -> (!llvm.ptr, i32, i8, i64, f64)
%2 = llvm.load %0 : !llvm.ptr -> i32
%3 = llvm.load %0 : !llvm.ptr -> index
%4 = "llvm.getelementptr"(%0, %2) <{"elem_type" = i32, "rawConstantIndices" = array<i32: -2147483648>}> : (!llvm.ptr, i32) -> !llvm.ptr
"llvm.store"(%1, %4) : (i32, !llvm.ptr) -> ()
%5 = "llvm.mlir.constant"() <{value = 7 : i32}> : () -> i32
%6 = "llvm.mlir.constant"() <{value = 1 : index}> : () -> index
%7 = "llvm.mlir.zero"() : () -> !llvm.ptr
%8 = "llvm.add"(%1, %5) : (i32, i32) -> i32
%9 = "llvm.mul"(%8, %5) : (i32, i32) -> i32
%10 = "llvm.ptrtoint"(%7) : (!llvm.ptr) -> i64
%11 = "llvm.inttoptr"(%10) : (i64) -> !llvm.ptr
%12 = "llvm.mlir.constant"() <{value = 1.25 : f32}> : () -> f32
%13 = "llvm.mlir.constant"() <{value = 2.5 : f32}> : () -> f32
%14 = "llvm.fadd"(%12, %13) : (f32, f32) -> f32
%15 = "llvm.fmul"(%14, %12) : (f32, f32) -> f32

%16 = "llvm.mlir.poison"() : () -> !llvm.struct<(i32, !llvm.array<2 x i32>)>
%17 = "llvm.insertvalue"(%5, %16) <{position = array<i32: 0>}> : (i32, !llvm.struct<(i32, !llvm.array<2 x i32>)>) -> !llvm.struct<(i32, !llvm.array<2 x i32>)>
%18 = "llvm.extractvalue"(%17) <{position = array<i32: 0>}> : (!llvm.struct<(i32, !llvm.array<2 x i32>)>) -> i32

// Integer arithmetic
%sub  = "llvm.sub"(%1, %5)  : (i32, i32) -> i32
%sdiv = "llvm.sdiv"(%1, %5) : (i32, i32) -> i32
%srem = "llvm.srem"(%1, %5) : (i32, i32) -> i32

// Float arithmetic
%fsub = "llvm.fsub"(%12, %13) : (f32, f32) -> f32
%fdiv = "llvm.fdiv"(%12, %13) : (f32, f32) -> f32
%frem = "llvm.frem"(%12, %13) : (f32, f32) -> f32
%fneg = "llvm.fneg"(%12)      : (f32) -> f32

// Float comparison (predicate 1 = OEQ)
%fc = "llvm.fcmp"(%12, %13) <{predicate = 1 : i64}> : (f32, f32) -> i1

// Bitwise
%and = "llvm.and"(%1, %5) : (i32, i32) -> i32
%or  = "llvm.or"(%1, %5)  : (i32, i32) -> i32
%xor = "llvm.xor"(%1, %5) : (i32, i32) -> i32

// Shifts
%shl  = "llvm.shl"(%1, %5)  : (i32, i32) -> i32
%lshr = "llvm.lshr"(%1, %5) : (i32, i32) -> i32
%ashr = "llvm.ashr"(%1, %5) : (i32, i32) -> i32

// Select
%sel = "llvm.select"(%fc, %1, %5) : (i1, i32, i32) -> i32

// Integer casts
%tr  = "llvm.trunc"(%1)      : (i32) -> i8
%ze  = "llvm.zext"(%i8val)   : (i8) -> i64
%se  = "llvm.sext"(%i8val)   : (i8) -> i64

// Int <-> Float conversions
%sitofp_v = "llvm.sitofp"(%1)     : (i32) -> f32
%fptosi_v = "llvm.fptosi"(%12)    : (f32) -> i32

// Float precision changes
%fptrunc_v = "llvm.fptrunc"(%f64val) : (f64) -> f32
%fpext_v   = "llvm.fpext"(%12)       : (f32) -> f64

// Alloca
%ap = "llvm.alloca"(%5) <{elem_type = i32}> : (i32) -> !llvm.ptr

// Call indirect (void call)
"llvm.call_indirect"(%0) : (!llvm.ptr) -> ()

builtin.module {
  llvm.func @callee(%arg0: i32) -> i32 {
    %0 = "llvm.mlir.constant"() <{value = 0 : i32}> : () -> i32
    %1 = llvm.icmp "slt" %arg0, %0 : i32
    "llvm.cond_br"(%1)[^bb1, ^bb2] <{operandSegmentSizes = array<i32: 1, 0, 0>}> : (i1) -> ()
  ^bb1:
    "llvm.br"()[^bb3] : () -> ()
  ^bb2:
    "llvm.br"()[^bb3] : () -> ()
  ^bb3:
    %2 = "llvm.mlir.constant"() <{value = 1 : i32}> : () -> i32
    "llvm.return"(%2) : (i32) -> ()
  }

  llvm.func @caller() -> i32 {
    %0 = "llvm.mlir.constant"() <{value = 9 : i32}> : () -> i32
    %1 = "llvm.call"(%0) <{callee = @callee}> : (i32) -> i32
    "llvm.return"(%1) : (i32) -> ()
  }
}

// CHECK:       builtin.module {
// CHECK-NEXT:    %0, %1, %2, %3, %4 = "test.op"() : () -> (!llvm.ptr, i32, i8, i64, f64)
// CHECK-NEXT:    %5 = llvm.load %0 : !llvm.ptr -> i32
// CHECK-NEXT:    %6 = llvm.load %0 : !llvm.ptr -> index
// CHECK-NEXT:    %7 = "llvm.getelementptr"(%0, %5) <{rawConstantIndices = array<i32: -2147483648>, elem_type = i32}> : (!llvm.ptr, i32) -> !llvm.ptr
// CHECK-NEXT:    "llvm.store"(%1, %7) : (i32, !llvm.ptr) -> ()
// CHECK-NEXT:    %8 = "llvm.mlir.constant"() <{value = 7 : i32}> : () -> i32
// CHECK-NEXT:    %9 = "llvm.mlir.constant"() <{value = 1 : index}> : () -> index
// CHECK-NEXT:    %10 = "llvm.mlir.zero"() : () -> !llvm.ptr
// CHECK-NEXT:    %11 = "llvm.add"(%1, %8) : (i32, i32) -> i32
// CHECK-NEXT:    %12 = "llvm.mul"(%11, %8) : (i32, i32) -> i32
// CHECK-NEXT:    %13 = "llvm.ptrtoint"(%10) : (!llvm.ptr) -> i64
// CHECK-NEXT:    %14 = "llvm.inttoptr"(%13) : (i64) -> !llvm.ptr
// CHECK-NEXT:    %15 = "llvm.mlir.constant"() <{value = 1.25 : f32}> : () -> f32
// CHECK-NEXT:    %16 = "llvm.mlir.constant"() <{value = 2.5 : f32}> : () -> f32
// CHECK-NEXT:    %17 = "llvm.fadd"(%15, %16) : (f32, f32) -> f32
// CHECK-NEXT:    %18 = "llvm.fmul"(%17, %15) : (f32, f32) -> f32
// CHECK-NEXT:    %19 = llvm.mlir.poison : !llvm.struct<(i32, !llvm.array<2 x i32>)>
// CHECK-NEXT:    %20 = llvm.insertvalue %8, %19[0] : !llvm.struct<(i32, !llvm.array<2 x i32>)>
// CHECK-NEXT:    %21 = llvm.extractvalue %20[0] : !llvm.struct<(i32, !llvm.array<2 x i32>)>
// CHECK-NEXT:    %22 = "llvm.sub"(%1, %8) : (i32, i32) -> i32
// CHECK-NEXT:    %23 = "llvm.sdiv"(%1, %8) : (i32, i32) -> i32
// CHECK-NEXT:    %24 = "llvm.srem"(%1, %8) : (i32, i32) -> i32
// CHECK-NEXT:    %25 = "llvm.fsub"(%15, %16) : (f32, f32) -> f32
// CHECK-NEXT:    %26 = "llvm.fdiv"(%15, %16) : (f32, f32) -> f32
// CHECK-NEXT:    %27 = "llvm.frem"(%15, %16) : (f32, f32) -> f32
// CHECK-NEXT:    %28 = "llvm.fneg"(%15) : (f32) -> f32
// CHECK-NEXT:    %29 = "llvm.fcmp"(%15, %16) <{predicate = 1}> : (f32, f32) -> i1
// CHECK-NEXT:    %30 = "llvm.and"(%1, %8) : (i32, i32) -> i32
// CHECK-NEXT:    %31 = "llvm.or"(%1, %8) : (i32, i32) -> i32
// CHECK-NEXT:    %32 = "llvm.xor"(%1, %8) : (i32, i32) -> i32
// CHECK-NEXT:    %33 = "llvm.shl"(%1, %8) : (i32, i32) -> i32
// CHECK-NEXT:    %34 = "llvm.lshr"(%1, %8) : (i32, i32) -> i32
// CHECK-NEXT:    %35 = "llvm.ashr"(%1, %8) : (i32, i32) -> i32
// CHECK-NEXT:    %36 = "llvm.select"(%29, %1, %8) : (i1, i32, i32) -> i32
// CHECK-NEXT:    %37 = "llvm.trunc"(%1) : (i32) -> i8
// CHECK-NEXT:    %38 = "llvm.zext"(%2) : (i8) -> i64
// CHECK-NEXT:    %39 = "llvm.sext"(%2) : (i8) -> i64
// CHECK-NEXT:    %40 = "llvm.sitofp"(%1) : (i32) -> f32
// CHECK-NEXT:    %41 = "llvm.fptosi"(%15) : (f32) -> i32
// CHECK-NEXT:    %42 = "llvm.fptrunc"(%4) : (f64) -> f32
// CHECK-NEXT:    %43 = "llvm.fpext"(%15) : (f32) -> f64
// CHECK-NEXT:    %44 = "llvm.alloca"(%8) <{elem_type = i32}> : (i32) -> !llvm.ptr
// CHECK-NEXT:    "llvm.call_indirect"(%0) : (!llvm.ptr) -> ()
// CHECK-NEXT:    builtin.module {
// CHECK:         llvm.func @callee(%{{.*}}: i32) -> i32 {
// CHECK:           %{{.*}} = "llvm.mlir.constant"() <{value = 0 : i32}> : () -> i32
// CHECK:           %{{.*}} = llvm.icmp "slt" %{{.*}}, %{{.*}} : i32
// CHECK:           "llvm.cond_br"(%{{.*}})[^bb0, ^bb1] <{operandSegmentSizes = array<i32: 1, 0, 0>}> : (i1) -> ()
// CHECK:         ^bb0:
// CHECK:           "llvm.br"()[^bb2] : () -> ()
// CHECK:         ^bb1:
// CHECK:           "llvm.br"()[^bb2] : () -> ()
// CHECK:         ^bb2:
// CHECK:           %{{.*}} = "llvm.mlir.constant"() <{value = 1 : i32}> : () -> i32
// CHECK:           "llvm.return"(%{{.*}}) : (i32) -> ()
// CHECK:         }
// CHECK:         llvm.func @caller() -> i32 {
// CHECK:           %{{.*}} = "llvm.mlir.constant"() <{value = 9 : i32}> : () -> i32
// CHECK:           %{{.*}} = llvm.call @callee(%{{.*}}) : (i32) -> i32
// CHECK:           "llvm.return"(%{{.*}}) : (i32) -> ()
// CHECK:         }
// CHECK-NEXT:    }
// CHECK-NEXT:  }
