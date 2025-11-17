// RUN: scair-opt -p=monomorphize | filecheck %s

builtin.module {
  %0 = "dlam.tlambda"() ({
    %1 = "dlam.tlambda"() ({
      %2 = "dlam.vlambda"() <{funAttr = !dlam.fun<!dlam.bvar<0>, !dlam.bvar<0>>}> ({
      ^bb0(%3: !dlam.bvar<0>):
        "dlam.vreturn"(%3) <{expected = !dlam.bvar<0>}> : (!dlam.bvar<0>) -> ()
      }) : () -> (!dlam.fun<!dlam.bvar<0>, !dlam.bvar<0>>)
      "dlam.treturn"(%2) <{expected = !dlam.fun<!dlam.bvar<0>, !dlam.bvar<0>>}> : (!dlam.fun<!dlam.bvar<0>, !dlam.bvar<0>>) -> ()
    }) : () -> (!dlam.forall<!dlam.fun<!dlam.bvar<0>, !dlam.bvar<0>>>)
    %2 = "dlam.tapply"(%1) <{argType = !dlam.bvar<0>}> : (!dlam.forall<!dlam.fun<!dlam.bvar<0>, !dlam.bvar<0>>>) -> (!dlam.fun<!dlam.bvar<0>, !dlam.bvar<0>>)
    "dlam.treturn"(%2) <{expected = !dlam.fun<!dlam.bvar<0>, !dlam.bvar<0>>}> : (!dlam.fun<!dlam.bvar<0>, !dlam.bvar<0>>) -> ()
  }) : () -> (!dlam.forall<!dlam.fun<!dlam.bvar<0>, !dlam.bvar<0>>>)
}

// CHECK-NOT: "dlam.tapply"
