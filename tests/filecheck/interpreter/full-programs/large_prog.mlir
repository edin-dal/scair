// AUTO-GENERATED BENCHMARK
builtin.module {
  func.func @main() -> (i32) {

    %c0 = "arith.constant"() <{value = 0 : i32}> : () -> i32
    %c1 = "arith.constant"() <{value = 1 : i32}> : () -> i32

    %v1 = "arith.addi"(%c0, %c1) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
    %v2 = "arith.addi"(%v1, %c1) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
    %v3 = "arith.addi"(%v2, %c1) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
    %v4 = "arith.addi"(%v3, %c1) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
    %v5 = "arith.addi"(%v4, %c1) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
    %v6 = "arith.addi"(%v5, %c1) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
    %v7 = "arith.addi"(%v6, %c1) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
    %v8 = "arith.addi"(%v7, %c1) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
    %v9 = "arith.addi"(%v8, %c1) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
    %v10 = "arith.addi"(%v9, %c1) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32

    func.return %v10 : i32
  }
}