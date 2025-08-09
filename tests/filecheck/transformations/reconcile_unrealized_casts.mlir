// RUN: scair-opt %s -p reconcile-unrealized-casts | filecheck %s

builtin.module {
  func.func @unusedCast(%0: i64) -> i64 {
    %1 = "builtin.unrealized_conversion_cast"(%0) : (i64) -> (i32)
    func.return %0 : i64
  }
  func.func @sameTypes(%0: i64) -> i64 {
    %1 = "test.op"() : () -> (i64)
    %2 = "builtin.unrealized_conversion_cast"(%1) : (i64) -> (i64)
    func.return %2 : i64
  }
}

// CHECK:       builtin.module {
// CHECK-NEXT:    func.func @unusedCast(%0: i64) -> i64 {
// CHECK-NEXT:      func.return %0 : i64
// CHECK-NEXT:    }
// CHECK-NEXT:    func.func @sameTypes(%0: i64) -> i64 {
// CHECK-NEXT:      %1 = "test.op"() : () -> (i64)
// CHECK-NEXT:      func.return %1 : i64
// CHECK-NEXT:    }
// CHECK-NEXT:  }
