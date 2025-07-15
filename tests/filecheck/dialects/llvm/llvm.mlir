// RUN: scair-opt %s | filecheck %s

%0, %1 = "test.op"() : () -> (!llvm.ptr, i32)
%2 = llvm.load %0 : !llvm.ptr -> i32
%3 = llvm.load %0 : !llvm.ptr -> index
%4 = "llvm.getelementptr"(%0, %2) <{"elem_type" = i32, "rawConstantIndices" = array<i32: -2147483648>}> : (!llvm.ptr, i32) -> !llvm.ptr

// CHECK:       builtin.module {
// CHECK-NEXT:    %0, %1 = "test.op"() : () -> (!llvm.ptr, i32)
// CHECK-NEXT:    %2 = llvm.load %0 : !llvm.ptr -> i32
// CHECK-NEXT:    %3 = llvm.load %0 : !llvm.ptr -> index
// CHECK-NEXT:    %4 = "llvm.getelementptr"(%0, %2) <{rawConstantIndices = array<i32: -2147483648>, elem_type = i32}> : (!llvm.ptr, i32) -> (!llvm.ptr)
// CHECK-NEXT:  }