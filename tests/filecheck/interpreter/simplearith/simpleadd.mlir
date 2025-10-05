// RUN: scair-run %s | filecheck %s

builtin.module {
  func.func @main() -> (i64) {
    %0 = "arith.constant"() <{value = 30 : i64}> : () -> (i64)
    %1 = "arith.constant"() <{value = 28 : i64}> : () -> (i64)
    %2 = "arith.addi"(%0, %1) : (i64, i64) -> (i64)
    func.return %2 : i64
  }
}

// CHECK: 58