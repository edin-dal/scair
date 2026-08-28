// RUN: scair-run %s | filecheck %s

builtin.module {
  func.func @main() -> (i32) {
    %c10 = "arith.constant"() <{value = 10 : i32}> : () -> i32
    %c20 = "arith.constant"() <{value = 20 : i32}> : () -> i32
    %c1 = "arith.constant"() <{value = 1 : i32}> : () -> i32
    %c2 = "arith.constant"() <{value = 2 : i32}> : () -> i32
    %cond = "arith.cmpi"(%c1, %c2) <{"predicate" = 2 : i64}> : (i32, i32) -> i1
    "llvm.cond_br"(%cond) [^then, ^else] <{operandSegmentSizes = array<i32: 1, 0, 0>}> : (i1) -> ()
  ^then:
    "llvm.br"(%c10) [^join] : (i32) -> ()
  ^else:
    "llvm.br"(%c20) [^join] : (i32) -> ()
  ^join(%res : i32):
    func.return %res : i32
  }
}

// CHECK: Result: 10
