// RUN: scair-run %s | filecheck %s

builtin.module {
  func.func @main() -> (i32) {
    %c1 = "arith.constant"() <{value = 1 : i32}> : () -> i32
    %c0 = "arith.constant"() <{value = 0 : i32}> : () -> i32
    %c6 = "arith.constant"() <{value = 6 : i32}> : () -> i32
    "llvm.br"(%c1, %c0) [^loop] : (i32, i32) -> ()
  ^loop(%i : i32, %acc : i32):
    %cond = "arith.cmpi"(%i, %c6) <{"predicate" = 2 : i64}> : (i32, i32) -> i1
    "llvm.cond_br"(%cond, %i, %acc, %i, %acc) [^body, ^exit] <{operandSegmentSizes = array<i32: 1, 2, 2>}> : (i1, i32, i32, i32, i32) -> ()
  ^body(%j : i32, %a : i32):
    %next = "arith.addi"(%j, %c1) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
    %sum = "arith.addi"(%a, %j) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
    "llvm.br"(%next, %sum) [^loop] : (i32, i32) -> ()
  ^exit(%k : i32, %s : i32):
    func.return %s : i32
  }
}

// CHECK: Result: 15
