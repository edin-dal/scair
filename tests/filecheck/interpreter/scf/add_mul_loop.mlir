// RUN: scair-run %s | filecheck %s

builtin.module {
  func.func @main() -> (i32) {
    %lb   = "arith.constant"() <{value = 0 : i32}> : () -> i32
    %ub   = "arith.constant"() <{value = 10 : i32}> : () -> i32
    %step = "arith.constant"() <{value = 1 : i32}> : () -> i32
    %init = "arith.constant"() <{value = 0 : i32}> : () -> i32
    %one  = "arith.constant"() <{value = 1 : i32}> : () -> i32
    %two  = "arith.constant"() <{value = 2 : i32}> : () -> i32

    %result = "scf.for"(%lb, %ub, %step, %init) ({
    ^bb0(%iv: i32, %acc: i32):
      %tmp1 = "arith.addi"(%acc, %one) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %tmp2 = "arith.muli"(%tmp1, %two) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      "scf.yield"(%tmp2) : (i32) -> ()
    }) : (i32, i32, i32, i32) -> i32

    func.return %result : i32
  }
}

// CHECK: Result: 20