// RUN: scair-opt %s | filecheck %s

"test.op"() ({
^bb0():
^bb1():
  "test.op"()[^bb2] : () -> ()
^bb2():
  "test.op"()[^bb1] : () -> ()
}) : () -> ()

// CHECK: builtin.module {

// CHECK:      "test.op"() ({
// CHECK-NEXT: ^bb0():
// CHECK:      ^bb1():
// CHECK-NEXT:   "test.op"()[^bb2] : () -> ()
// CHECK-NEXT: ^bb2():
// CHECK-NEXT:   "test.op"()[^bb1] : () -> ()
// CHECK-NEXT: }) : () -> ()

"test.op"() ({
^bb0():
^bb1():
  "test.op"() : () -> ()
^bb2():
  "test.op"()[^bb1] : () -> ()
}) : () -> ()

// CHECK:      "test.op"() ({
// CHECK-NEXT: ^bb0():
// CHECK:      ^bb1():
// CHECK-NEXT:   "test.op"() : () -> ()
// CHECK-NEXT: ^bb2():
// CHECK-NEXT:   "test.op"()[^bb1] : () -> ()
// CHECK-NEXT: }) : () -> ()

// CHECK-NEXT: }

