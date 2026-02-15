// RUN: scair-run %s | filecheck %s

builtin.module {
    %0 = "arith.constant"() <{value = 3 : i64}> : () -> i64
    %1 = "arith.constant"() <{value = 2 : i64}> : () -> i64
    %2 = "arith.addi"(%0, %1) : (i64, i64) -> i64
    "func.call"(%2) <{"callee" = @print}> : (i64) -> ()
}

// CHECK: Result: 5