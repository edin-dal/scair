// RUN: scair-run %s | filecheck %s

builtin.module {
  func.func @main() -> (i32) {
    %c0 = "arith.constant"() <{value = 0 : i32}> : () -> i32
    %c5 = "arith.constant"() <{value = 5 : i32}> : () -> i32
    %c1 = "arith.constant"() <{value = 1 : i32}> : () -> i32
    "llvm.br"(%c0) [^loop] : (i32) -> ()
  ^loop(%i : i32):
    %cond = "arith.cmpi"(%i, %c5) <{"predicate" = 2 : i64}> : (i32, i32) -> i1
    "llvm.cond_br"(%cond, %i, %i) [^body, ^exit] <{operandSegmentSizes = array<i32: 1, 1, 1>}> : (i1, i32, i32) -> ()
  ^body(%j : i32):
    %next = "arith.addi"(%j, %c1) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
    "llvm.br"(%next) [^loop] : (i32) -> ()
  ^exit(%k : i32):
    func.return %k : i32
  }
}

// CHECK: Result: 5
