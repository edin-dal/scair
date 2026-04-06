// RUN: scair-opt %s | filecheck %s

%0, %1 = "test.op"() : () -> (!llvm.ptr, i32)
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
// CHECK-NEXT:    %0, %1 = "test.op"() : () -> (!llvm.ptr, i32)
// CHECK-NEXT:    %2 = llvm.load %0 : !llvm.ptr -> i32
// CHECK-NEXT:    %3 = llvm.load %0 : !llvm.ptr -> index
// CHECK-NEXT:    %4 = "llvm.getelementptr"(%0, %2) <{rawConstantIndices = array<i32: -2147483648>, elem_type = i32}> : (!llvm.ptr, i32) -> !llvm.ptr
// CHECK-NEXT:    "llvm.store"(%1, %4) : (i32, !llvm.ptr) -> ()
// CHECK-NEXT:    %5 = "llvm.mlir.constant"() <{value = 7 : i32}> : () -> i32
// CHECK-NEXT:    %6 = "llvm.mlir.constant"() <{value = 1 : index}> : () -> index
// CHECK-NEXT:    %7 = "llvm.mlir.zero"() : () -> !llvm.ptr
// CHECK-NEXT:    %8 = "llvm.add"(%1, %5) : (i32, i32) -> i32
// CHECK-NEXT:    %9 = "llvm.mul"(%8, %5) : (i32, i32) -> i32
// CHECK-NEXT:    %10 = "llvm.ptrtoint"(%7) : (!llvm.ptr) -> i64
// CHECK-NEXT:    %11 = "llvm.inttoptr"(%10) : (i64) -> !llvm.ptr
// CHECK-NEXT:    %12 = "llvm.mlir.constant"() <{value = 1.25 : f32}> : () -> f32
// CHECK-NEXT:    %13 = "llvm.mlir.constant"() <{value = 2.5 : f32}> : () -> f32
// CHECK-NEXT:    %14 = "llvm.fadd"(%12, %13) : (f32, f32) -> f32
// CHECK-NEXT:    %15 = "llvm.fmul"(%14, %12) : (f32, f32) -> f32
// CHECK-NEXT:    %16 = llvm.mlir.poison : !llvm.struct<(i32, !llvm.array<2 x i32>)>
// CHECK-NEXT:    %17 = llvm.insertvalue %5, %16[0] : !llvm.struct<(i32, !llvm.array<2 x i32>)>
// CHECK-NEXT:    %18 = llvm.extractvalue %17[0] : !llvm.struct<(i32, !llvm.array<2 x i32>)>
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
