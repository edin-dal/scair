// RUN: scair-opt %s | filecheck %s

"builtin.module"() ({
  "op1"() ({
  ^bb0():
    "test.op"()[^bb1] : () -> ()
  ^bb1():
    "test.op"() : () -> ()
  }) : () -> ()

  // CHECK:      "op1"() ({
  // CHECK-NEXT: ^bb1():
  // CHECK-NEXT:   "test.op"()[^bb2] : () -> ()
  // CHECK-NEXT: ^bb2():
  // CHECK-NEXT:   "test.op"() : () -> ()
  // CHECK-NEXT: }) : () -> ()

  "op1"() ({
  ^bb0():
    "test.op"() : () -> ()
  ^bb1():
    "test.op"()[^bb0] : () -> ()
  }) : () -> ()

  // CHECK:      "op1"() ({
  // CHECK-NEXT: ^bb3():
  // CHECK-NEXT:   "test.op"() : () -> ()
  // CHECK-NEXT: ^bb4():
  // CHECK-NEXT:   "test.op"()[^bb3] : () -> ()
  // CHECK-NEXT: }) : () -> ()

}) : () -> ()