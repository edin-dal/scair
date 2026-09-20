// RUN: scair-run %s | filecheck %s

builtin.module {
  func.func @main() -> (i32) {
    %c3 = "arith.constant"() <{value = 3 : i32}> : () -> i32
    %c7 = "arith.constant"() <{value = 7 : i32}> : () -> i32
    %cond = "arith.cmpi"(%c7, %c3) <{"predicate" = 4 : i64}> : (i32, i32) -> i1
    "llvm.cond_br"(%cond) [^then, ^else] <{operandSegmentSizes = array<i32: 1, 0, 0>}> : (i1) -> ()
  ^then:
    %t = "arith.muli"(%c3, %c7) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
    "llvm.br"(%t) [^join] : (i32) -> ()
  ^else:
    %e = "arith.addi"(%c3, %c7) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
    "llvm.br"(%e) [^join] : (i32) -> ()
  ^join(%res : i32):
    func.return %res : i32
  }
}

// CHECK: Result: 21
