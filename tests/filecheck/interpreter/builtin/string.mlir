// RUN: scair-run %s | filecheck %s

builtin.module {
  func.func @main() -> (i64) {
    %0 = "builtin.string"() <{value = hey : }> : () -> (i64)
    func.return %0 : i64
  }
}

// CHECK: hey