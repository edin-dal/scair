// RUN: scair-run %s | filecheck %s

builtin.module {
  func.func @main() -> (i64) {
    %0 = "arith.constant"() <{value = 3 : i64}> : () -> (i64)
    %1 = "arith.constant"() <{value = 4 : i64}> : () -> (i64)
    %2 = "arith.muli"(%0, %1) : (i64, i64) -> (i64)
    func.return %2 : i64
  }
}

// CHECK: 12