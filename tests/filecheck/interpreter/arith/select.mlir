// RUN: scair-run %s | filecheck %s

builtin.module {
  func.func @main() -> i64 {
    %0 = "arith.constant"() <{value = 1 : i64}> : () -> i64
    %1 = "arith.constant"() <{value = 6 : i64}> : () -> i64
    %2 = "arith.constant"() <{value = 7 : i64}> : () -> i64
    %3 = "arith.select"(%0, %1, %2) : (i64, i64, i64) -> i32
    func.return %3 : i32
  }
}

// CHECK: 6