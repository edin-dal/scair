// RUN: scair-opt %s --verify-diagnostics | filecheck %s

builtin.module {
  %0, %1 = "test.op"() : () -> (!llvm.ptr, i32)
  %2 = "llvm.getelementptr"(%0) <{"elem_type" = i32, "rawConstantIndices" = array<i32: -2147483648>}> : (!llvm.ptr) -> !llvm.ptr
}

// CHECK: llvm.getelementptr: rawConstantIndices contain 1 dynamic markers but op has 0 dynamic indices
