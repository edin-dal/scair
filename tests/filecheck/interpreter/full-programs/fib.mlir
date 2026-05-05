// RUN: scair-run %s | filecheck %s

builtin.module {
  func.func @main() -> (i32) {
    %n    = "arith.constant"() <{value = 10 : i32}> : () -> i32
    %lb   = "arith.constant"() <{value = 0 : i32}> : () -> i32
    %step = "arith.constant"() <{value = 1 : i32}> : () -> i32

    %a0   = "arith.constant"() <{value = 0 : i32}> : () -> i32
    %b0   = "arith.constant"() <{value = 1 : i32}> : () -> i32

    %a_res, %b_res = "scf.for"(%lb, %n, %step, %a0, %b0) ({
    ^bb0(%iv: i32, %a: i32, %b: i32):
      %next = "arith.addi"(%a, %b) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32

      "scf.yield"(%b, %next) : (i32, i32) -> ()
    }) : (i32, i32, i32, i32, i32) -> (i32, i32)

    %fib_n = "arith.addi"(%a_res, %lb) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32

    func.return %fib_n : i32
  }
}

// CHECK: Result: 55