// RUN: scair-run %s | filecheck %s

builtin.module {
  func.func @square(%x: i64) -> (i64) {
    %result = "arith.muli"(%x, %x) : (i64, i64) -> i64
    func.return %result : i64
  }
  
  func.func @main() -> (i64) {
    %n = "arith.constant"() <{value = 7 : i64}> : () -> i64
    %squared = "func.call"(%n) <{"callee" = @square}> : (i64) -> i64
    func.return %squared : i64
  }
}

// CHECK: 81