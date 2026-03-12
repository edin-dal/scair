// RUN: scair-run %s | filecheck %s

builtin.module {
    %0 = "arith.constant"() <{value = 42 : i64}> : () -> i64
    "func.call"(%0) <{"callee" = @print}> : (i64) -> ()
}

// CHECK: Result: 42