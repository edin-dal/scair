// RUN: scair-run %s | filecheck %s

builtin.module {

  func.func @rec(%n: i32) -> i32 {

    %c0 = "arith.constant"() <{value = 0 : i32}> : () -> i32
    %c1 = "arith.constant"() <{value = 1 : i32}> : () -> i32

    %cond = "arith.cmpi"(%n, %c0) <{predicate = 0 : i64}> : (i32, i32) -> i1

    %res = "scf.if"(%cond) ({
      scf.yield %c0 : i32
    }, {
      %n1 = "arith.subi"(%n, %c1) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %r = "func.call"(%n1) <{"callee" = @rec}> : (i32) -> i32
      %out = "arith.addi"(%r, %c1) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      scf.yield %out : i32
    }) : (i1) -> i32

    func.return %res : i32
  }

  func.func @main() -> (i32) {

    %c = "arith.constant"() <{value = 487 : i32}> : () -> i32
    %res = "func.call"(%c) <{"callee" = @rec}> : (i32) -> i32

    func.return %res : i32
  }
}

// CHECK: Result: