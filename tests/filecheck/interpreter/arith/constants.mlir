// RUN: scair-run %s | filecheck %s

builtin.module {
  func.func @main() -> i64 {
    %0 = "arith.constant"() <{value = 42 : i64}> : () -> i64
    func.return %0 : i64
  }
}

// CHECK: Result: 42