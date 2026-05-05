// RUN: scair-opt %s --verify-diagnostics | filecheck %s

func.func @zero() -> i64 {
    %zero = "arith.constant"() <{ value = 0 : i64 }> : () -> i64
    func.return %zero : i64
}

%zerof = "func.constant"() <{ value = @zero }> : () -> () -> i64
%zero = "func.call_indirect"(%zerof): (() -> i64) -> i32

// CHECK:       "builtin.module"() ({
// CHECK-NEXT:    "func.func"() <{sym_name = "zero", function_type = () -> i64}> ({
// CHECK-NEXT:      %0 = "arith.constant"() <{value = 0}> : () -> i64
// CHECK-NEXT:      "func.return"(%0) : (i64) -> ()
// CHECK-NEXT:    }) : () -> ()
// CHECK-NEXT:    %0 = "func.constant"() <{value = @zero}> : () -> () -> i64
// CHECK-NEXT:    ^^
// CHECK-NEXT:    > func.call_indirect: result types List(i32) do not match callee output types List(i64)
// CHECK-NEXT:    %1 = "func.call_indirect"(%0) : (() -> i64) -> i32
// CHECK-NEXT:                              ^^
// CHECK-NEXT:                              > func.call_indirect: result types List(i32) do not match callee output types List(i64)
// CHECK-NEXT:  }) : () -> ()