// RUN: scair-run %s | filecheck %s

builtin.module {
  func.func @main() -> (i32) {
    %lb   = "arith.constant"() <{value = 0 : i32}> : () -> i32
    %ub   = "arith.constant"() <{value = 1000000 : i32}> : () -> i32
    %step = "arith.constant"() <{value = 1 : i32}> : () -> i32
    %init = "arith.constant"() <{value = 0 : i32}> : () -> i32
    %one  = "arith.constant"() <{value = 1 : i32}> : () -> i32

    // Block args:
    //   %iv  = induction variable
    //   %acc = loop-carried accumulator
    %result = "scf.for"(%lb, %ub, %step, %init) ({
    ^bb0(%iv: i32, %acc: i32):
      %next = "arith.addi"(%acc, %one) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      "scf.yield"(%next) : (i32) -> ()
    }) : (i32, i32, i32, i32) -> i32

    func.return %result : i32
  }
}

// CHECK: Result: 5