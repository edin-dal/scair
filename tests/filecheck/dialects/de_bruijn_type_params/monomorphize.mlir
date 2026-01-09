// RUN: scair-opt -p=monomorphize %s | filecheck %s --dump-input=fail --dump-input-filter all

builtin.module {
  %0 = "tlam.tlambda"() ({
    %1 = "tlam.tlambda"() ({
      %2 = "tlam.vlambda"() <{funAttr = !tlam.fun<!tlam.bvar<0>, !tlam.bvar<0>>}> ({
      ^bb0(%3: !tlam.bvar<0>):
        "tlam.vreturn"(%3) <{expected = !tlam.bvar<0>}> : (!tlam.bvar<0>) -> ()
      }) : () -> (!tlam.fun<!tlam.bvar<0>, !tlam.bvar<0>>)
      "tlam.treturn"(%2) <{expected = !tlam.fun<!tlam.bvar<0>, !tlam.bvar<0>>}> : (!tlam.fun<!tlam.bvar<0>, !tlam.bvar<0>>) -> ()
    }) : () -> (!tlam.forall<!tlam.fun<!tlam.bvar<0>, !tlam.bvar<0>>>)
    %2 = "tlam.tapply"(%1) <{argType = !tlam.bvar<0>}> : (!tlam.forall<!tlam.fun<!tlam.bvar<0>, !tlam.bvar<0>>>) -> (!tlam.fun<!tlam.bvar<0>, !tlam.bvar<0>>)
    "tlam.treturn"(%2) <{expected = !tlam.fun<!tlam.bvar<0>, !tlam.bvar<0>>}> : (!tlam.fun<!tlam.bvar<0>, !tlam.bvar<0>>) -> ()
  }) : () -> (!tlam.forall<!tlam.fun<!tlam.bvar<0>, !tlam.bvar<0>>>)
}

// CHECK-NOT: "tlam.tapply"
// CHECK: builtin.module { 
// CHECK:   %0 = "tlam.tlambda"() ({ 
// CHECK:     %1 = "tlam.vlambda"() <{funAttr = !tlam.fun<!tlam.bvar<0>, !tlam.bvar<0>>}> ({ 
// CHECK:     ^bb0(%2: !tlam.bvar<0>): 
// CHECK:        "tlam.vreturn"(%2) <{expected = !tlam.bvar<0>}> : (!tlam.bvar<0>) -> () 
// CHECK:      }) : () -> !tlam.fun<!tlam.bvar<0>, !tlam.bvar<0>> 
// CHECK:     "tlam.treturn"(%1) <{expected = !tlam.fun<!tlam.bvar<0>, !tlam.bvar<0>>}> : (!tlam.fun<!tlam.bvar<0>, !tlam.bvar<0>>) -> () 
// CHECK:      }) : () -> !tlam.forall<!tlam.fun<!tlam.bvar<0>, !tlam.bvar<0>>> 
// CHECK:  }
