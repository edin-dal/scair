// RUN: scair-opt %s -p convert-func-to-llvm | filecheck %s
// RUN: scair-opt %s -p convert-func-to-llvm | /home/dominic/dev/llvm-source/build/bin/mlir-translate --mlir-to-llvmir > /dev/null

builtin.module {
  func.func private @printNewline()

  func.func @callee() -> i32 {
    %c7 = "llvm.mlir.constant"() <{value = 7 : i32}> : () -> i32
    "func.return"(%c7) : (i32) -> ()
  }

  func.func @main() -> i32 {
    %v = "func.call"() <{callee = @callee}> : () -> i32
    "func.call"() <{callee = @printNewline}> : () -> ()
    "func.return"(%v) : (i32) -> ()
  }
}

// CHECK: builtin.module {
// CHECK-NEXT:   llvm.func @printNewline()
// CHECK-NEXT:   llvm.func @callee() -> i32 {
// CHECK-NEXT:     %0 = "llvm.mlir.constant"() <{value = 7 : i32}> : () -> i32
// CHECK-NEXT:     "llvm.return"(%0) : (i32) -> ()
// CHECK-NEXT:   }
// CHECK-NEXT:   llvm.func @main() -> i32 {
// CHECK-NEXT:     %0 = llvm.call @callee() : () -> i32
// CHECK-NEXT:     llvm.call @printNewline() : () -> ()
// CHECK-NEXT:     "llvm.return"(%0) : (i32) -> ()
// CHECK-NEXT:   }
// CHECK-NEXT: }
