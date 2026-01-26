// RUN: scair-opt %s --allow-unregistered-dialect -p monomorphize,erase-tlam | filecheck %s

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
// CHECK-NOT:     "tlam.tlambda"
// CHECK-NOT:     "tlam.treturn"
// CHECK-NEXT:    %0 = "tlam.vlambda"() ({
// CHECK-NEXT:    ^bb0(%1: i64): 
// CHECK-NEXT:      "tlam.vreturn"(%1) : (i64) -> () 
// CHECK-NEXT:    }) : () -> !tlam.fun<i64, i64>
// CHECK-NEXT:  }
