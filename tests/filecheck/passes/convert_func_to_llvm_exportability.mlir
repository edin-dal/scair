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

// CHECK-LABEL: llvm.func @printNewline()
// CHECK-LABEL: llvm.func @callee() -> i32 {
// CHECK: "llvm.return"(%{{.*}}) : (i32) -> ()
// CHECK-LABEL: llvm.func @main() -> i32 {
// CHECK: llvm.call @callee() : () -> i32
// CHECK: llvm.call @printNewline() : () -> ()
// CHECK-NOT: func.func
// CHECK-NOT: func.call
