// RUN: scair-opt %s -p reconcile-unrealized-casts | filecheck %s

builtin.module {
//   "func.func"() <{function_type = (i64) -> i64, sym_name = "unusedCast"}> ({
//   ^bb0(%arg0: i64):
//     %0 = "builtin.unrealized_conversion_cast"(%arg0) : (i64) -> i32
//     "func.return"(%arg0) : (i64) -> ()
//   }) : () -> ()
  "func.func"() <{function_type = (i64) -> i64, sym_name = "sameTypes"}> ({
  ^bb0(%arg0: i64):
    %1 = "test.op"() : () -> i64
    %0 = "builtin.unrealized_conversion_cast"(%1) : (i64) -> i64
    "func.return"(%0) : (i64) -> ()
  }) : () -> ()
}