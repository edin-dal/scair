// RUN: scair-run %s | filecheck %s

builtin.module {
  func.func @main() -> () {
    %0 = "test.op"() : () -> (i32)
    func.return
  }
}

// CHECK: Such interpretation, much wow