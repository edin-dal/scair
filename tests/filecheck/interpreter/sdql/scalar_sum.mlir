// RUN: scair-run %s | filecheck %s

builtin.module {
  func.func @main() -> i64 {
    %k1 = "arith.constant"() <{value = 1 : i32}> : () -> i32

    %v1 = "arith.constant"() <{value = 3 : i32}> : () -> i32
    %k2 = "arith.constant"() <{value = 6 : i32}> : () -> i32
    %v2 = "arith.constant"() <{value = 13 : i32}> : () -> i32

    %0 = sdql.create_dictionary %k1, %v1, %k2, %v2 : i32, i32, i32, i32 -> dictionary<i32, i32>

    %1 = sdql.sum %0 : dictionary<i32, i32> -> i32 {
    ^bb0(%k: i32, %v: i32):
      sdql.yield %v : i32
    }
    func.return %1 : i32
  }
}

// CHECK: 16