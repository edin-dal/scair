// RUN: scair-run %s | filecheck %s

builtin.module {

  func.func @main() -> i32 {
    %3 = "arith.constant"() <{value = 3 : i32}> : () -> i32
    %4 = "func.call"(%3) <{"callee" = @add2}> : (i32) -> i32
    func.return %4 : i32
  }

  func.func @add2(%0: i32) -> i32 {
      %1 = "arith.constant"() <{value = 2 : i32}> : () -> i32
      %2 = "arith.addi"(%0, %1) : (i32, i32) -> i32
      func.return %2 : i32
  }
}

// CHECK: Result: 5