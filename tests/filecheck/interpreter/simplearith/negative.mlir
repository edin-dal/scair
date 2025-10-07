// RUN: scair-run %s | filecheck %s

builtin.module {
  func.func @main() -> (si8) {
    %0 = "arith.constant"() <{value = 129 : si8}> : () -> (si8)
    func.return %0 : si8
  }
}

// CHECK: 42