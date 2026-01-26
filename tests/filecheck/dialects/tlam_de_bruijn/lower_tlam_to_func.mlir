// RUN: scair-opt %s --allow-unregistered-dialect -p monomorphize,erase-tlam,lower-tlam-to-func | filecheck %s

builtin.module {
  %0 = "tlam.tlambda"() ({
    %poly_id = "tlam.tlambda"() ({
      %id = "tlam.vlambda"() <{funAttr = !tlam.fun<!tlam.bvar<0>, !tlam.bvar<0>>}> ({
      ^bb0(%x: !tlam.bvar<0>):
        "tlam.vreturn"(%x) <{expected = !tlam.bvar<0>}> : (!tlam.bvar<0>) -> ()
      }) : () -> (!tlam.fun<!tlam.bvar<0>, !tlam.bvar<0>>)
      "tlam.treturn"(%id) <{expected = !tlam.fun<!tlam.bvar<0>, !tlam.bvar<0>>}> : (!tlam.fun<!tlam.bvar<0>, !tlam.bvar<0>>) -> ()
    }) : () -> (!tlam.forall<!tlam.fun<!tlam.bvar<0>, !tlam.bvar<0>>>)

    %spec = "tlam.tapply"(%poly_id) <{tyArg = i64}> : (!tlam.forall<!tlam.fun<!tlam.bvar<0>, !tlam.bvar<0>>>) -> (!tlam.fun<i64, i64>)
    "tlam.treturn"(%spec) <{expected = !tlam.fun<i64, i64>}> : (!tlam.fun<i64, i64>) -> ()
  }) : () -> (!tlam.forall<!tlam.fun<i64, i64>>)
}

// CHECK:       builtin.module {
// CHECK-NOT:     "tlam."
// CHECK:         func.func @lifted_1(%0: i64) -> i64 {
// CHECK-NEXT:      func.return %0 : i64 
// CHECK-NEXT:    }
// CHECK:         %0 = "func.constant"() <{callee = @lifted_1}> : () -> (i64) -> i64 
// CHECK:       }
