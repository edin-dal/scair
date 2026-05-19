// RUN: scair-run %s | filecheck %s

builtin.module {
  func.func @main() -> (i32) {

    %lb   = "arith.constant"() <{value = 0 : i32}> : () -> i32
    %ub   = "arith.constant"() <{value = 1000 : i32}> : () -> i32
    %step = "arith.constant"() <{value = 1 : i32}> : () -> i32
    %one  = "arith.constant"() <{value = 1 : i32}> : () -> i32

    "scf.for"(%lb, %ub, %step) ({
    ^bb0(%i: i32):

      "scf.for"(%lb, %ub, %step) ({
      ^bb1(%j: i32):
        %a = "arith.addi"(%i, %j)
          <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        %b = "arith.addi"(%a, %one)
          <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        "scf.yield"() : () -> ()
      }) : (i32, i32, i32) -> ()
      "scf.yield"() : () -> ()
    }) : (i32, i32, i32) -> ()

    %final = "arith.constant"() <{value = 0 : i32}> : () -> i32
    func.return %final : i32
  }
}

// CHECK: Result: