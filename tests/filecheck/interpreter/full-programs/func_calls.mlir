// RUN: scair-run %s | filecheck %s

builtin.module {
  func.func @inc(%x: i32) -> i32 {
    %one = "arith.constant"() <{value = 1 : i32}> : () -> i32
    %r = "arith.addi"(%x, %one) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
    func.return %r : i32
  }

  func.func @main() -> (i32) {
    %lb   = "arith.constant"() <{value = 0 : i32}> : () -> i32
    %ub   = "arith.constant"() <{value = 100 : i32}> : () -> i32
    %step = "arith.constant"() <{value = 1 : i32}> : () -> i32
    %init = "arith.constant"() <{value = 0 : i32}> : () -> i32

    %result = "scf.for"(%lb, %ub, %step, %init) ({
    ^bb0(%iv: i32, %acc: i32):
      %next = "func.call"(%acc) <{"callee" = @inc}> : (i32) -> i32
      "scf.yield"(%next) : (i32) -> ()
    }) : (i32, i32, i32, i32) -> i32

    func.return %result : i32
  }
}

// CHECK: Result: 10