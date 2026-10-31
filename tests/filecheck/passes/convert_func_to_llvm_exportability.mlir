// RUN: scair-opt %s -p convert-func-to-llvm | filecheck %s

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

// CHECK: "llvm.func"()
// CHECK-SAME: sym_name = "printNewline"
// CHECK: "llvm.func"()
// CHECK-SAME: sym_name = "callee"
// CHECK-SAME: function_type = !llvm.func<i32 ()>
// CHECK: "llvm.mlir.constant"() <{value = 7 : i32}>
// CHECK: "llvm.return"
// CHECK: "llvm.func"()
// CHECK-SAME: sym_name = "main"
// CHECK-SAME: function_type = !llvm.func<i32 ()>
// CHECK: "llvm.call"
// CHECK-SAME: callee = @callee
// CHECK: "llvm.call"
// CHECK-SAME: callee = @printNewline
// CHECK: "llvm.return"
