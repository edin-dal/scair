// RUN: scair-run %s | filecheck %s

builtin.module {
  func.func @main() -> (i64) {
    %0 = "arith.constant"() <{value = 1 : i64}> : () -> (i64)
    %1 = "arith.constant"() <{value = 0 : i64}> : () -> (i64)
    %2 = "arith.constant"() <{value = 1 : i64}> : () -> (i64)
    %3 = "arith.andi"(%0, %1) : (i64, i64) -> (i64)
    %4 = "arith.ori"(%3, %2) : (i64, i64) -> (i64)
    func.return %4 : i64
  }
}

// CHECK: Result: true