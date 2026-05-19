// RUN: scair-run %s | filecheck %s

builtin.module {
    func.func @main() -> () {
        %0 = "arith.constant"() <{value = 42 : i64}> : () -> i64
        "func.call"(%0) <{"callee" = @print}> : (i64) -> ()
    }
}

// CHECK: 42