// RUN: scair-opt %s | filecheck %s

"test.op"() ({
^bb0():
  "test.op"()[^bb1] : () -> ()
^bb1():
  "test.op"()[^bb0] : () -> ()
}) : () -> ()

// CHECK: builtin.module {

// CHECK:      "test.op"() ({
// CHECK-NEXT: ^bb1():
// CHECK-NEXT:   "test.op"()[^bb2] : () -> ()
// CHECK-NEXT: ^bb2():
// CHECK-NEXT:   "test.op"()[^bb1] : () -> ()
// CHECK-NEXT: }) : () -> ()

"test.op"() ({
^bb0():
  "test.op"() : () -> ()
^bb1():
  "test.op"()[^bb0] : () -> ()
}) : () -> ()

// CHECK:      "test.op"() ({
// CHECK-NEXT: ^bb3():
// CHECK-NEXT:   "test.op"() : () -> ()
// CHECK-NEXT: ^bb4():
// CHECK-NEXT:   "test.op"()[^bb3] : () -> ()
// CHECK-NEXT: }) : () -> ()

// CHECK-NEXT: }

