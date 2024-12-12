// RUN: scair-opt %s | mlir-opt --mlir-print-op-generic | scair-opt | filecheck %s

%0:2 = "test.op"() : () -> (!llvm.ptr, i32)
%1 = "llvm.load"(%0#0) : (!llvm.ptr) -> !llvm.ptr
%2 = "llvm.load"(%0#0) : (!llvm.ptr) -> !llvm.ptr
%3 = "llvm.getelementptr"(%0#0, %0#1) <{"elem_type" = !llvm.ptr, "rawConstantIndices" = array<i32: -2147483648>}> : (!llvm.ptr, i32) -> !llvm.ptr

// CHECK:       builtin.module {
// CHECK-NEXT:  ^bb0():
// CHECK-NEXT:    %0, %1 = "test.op"() : () -> (!llvm.ptr, i32)
// CHECK-NEXT:    %2 = "llvm.load"(%0) : (!llvm.ptr) -> (i32)
// CHECK-NEXT:    %3 = "llvm.load"(%0) : (!llvm.ptr) -> (index)
// CHECK-NEXT:    %4 = "llvm.getelementptr"(%0, %2) <{elem_type = i32, rawConstantIndices = array<i32: -2147483648>}> : (!llvm.ptr, i32) -> (!llvm.ptr)
// CHECK-NEXT:  }
