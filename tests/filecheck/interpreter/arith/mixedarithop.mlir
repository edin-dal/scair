// RUN: scair-run %s | filecheck %s

builtin.module {
  func.func @main() -> (i64) {
    %0 = "arith.constant"() <{value = 3 : i64}> : () -> (i64)
    %1 = "arith.constant"() <{value = 4 : i64}> : () -> (i64)
    %2 = "arith.constant"() <{value = 3 : i64}> : () -> (i64)
    %3 = "arith.addi"(%0, %1) : (i64, i64) -> (i64)
    %4 = "arith.muli"(%3, %2) : (i64, i64) -> (i64)
    func.return %4 : i64
  }
}

// CHECK: Result: 21