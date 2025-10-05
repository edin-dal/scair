// RUN: scair-run %s | filecheck %s

builtin.module {
  func.func @main() -> (i64) {
    %0 = "arith.constant"() <{value = 5 : i64}> : () -> (i64)
    %1 = "arith.constant"() <{value = 5 : i64}> : () -> (i64)
    %2 = "arith.cmpi"(%0, %1) <{"predicate" = 1 : i64}> : (i64, i64) -> (i1)
    func.return %2 : i1
  }
}

// CHECK: 0