%0, %17 = "test.op" : () -> (!llvm.ptr, i32)
%1 = "llvm.load"(%0) : (!llvm.ptr) -> i32
%2 = "llvm.load"(%0) : (!llvm.ptr) -> index
%3 = "llvm.getelementptr"(%0, %1) <{"elem_type" = i32, "rawConstantIndices" = array<i32: -2147483648>}> : (!llvm.ptr, i64) -> !llvm.ptr