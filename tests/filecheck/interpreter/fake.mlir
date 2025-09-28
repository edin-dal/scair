// RUN: scair-run %s | filecheck %s

builtin.module {
  func.func @main() -> (i64) {
    %0 = "arith.constant"() <{value = 42 : i64}> : () -> (i64)
    %1 = "arith.constant"() <{value = 45 : i64}> : () -> (i64)
    func.return %1 : i64
  }
}

// CHECK: Such interpretation, much wow