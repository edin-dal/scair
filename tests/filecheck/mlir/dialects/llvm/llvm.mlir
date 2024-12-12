// RUN: scair-opt %s | mlir-opt --mlir-print-op-generic | scair-opt | filecheck %s

%0:2 = "test.op"() : () -> (!llvm.ptr, i32)
%1 = "llvm.load"(%0#0) : (!llvm.ptr) -> !llvm.ptr
%2 = "llvm.load"(%0#0) : (!llvm.ptr) -> !llvm.ptr
%3 = "llvm.getelementptr"(%0#0, %0#1) <{"elem_type" = !llvm.ptr, "rawConstantIndices" = array<i32: -2147483648>}> : (!llvm.ptr, i32) -> !llvm.ptr

// CHECK:       builtin.module {
// CHECK-NEXT:    %0, %1 = "test.op"() : () -> (!llvm.ptr, i32)
// CHECK-NEXT:    %2 = "llvm.load"(%0) <{ordering = 0}> : (!llvm.ptr) -> (!llvm.ptr)
// CHECK-NEXT:    %3 = "llvm.load"(%0) <{ordering = 0}> : (!llvm.ptr) -> (!llvm.ptr)
// CHECK-NEXT:    %4 = "llvm.getelementptr"(%0, %1) <{elem_type = !llvm.ptr, rawConstantIndices = array<i32: -2147483648>}> : (!llvm.ptr, i32) -> (!llvm.ptr)
// CHECK-NEXT:  }
