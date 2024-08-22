// RUN: scair-opt %s -p test-ins-pass | filecheck %s

"test.op"() : () -> ()
"test.op"() : () -> ()
"tobereplaced"() : () -> ()
"test.op"() : () -> ()
"test.op"() : () -> ()

// CHECK: builtin.module {
// CHECK: ^bb0():
// CHECK:   "test.op"() {replaced = "false"} : () -> ()
// CHECK:   "test.op"() {replaced = "false"} : () -> ()
// CHECK:   "dummy1"() : () -> ()
// CHECK:   "dummy2"() : () -> ()
// CHECK:   "dummy3"() : () -> ()
// CHECK:   "dummy4"() : () -> ()
// CHECK:   "test.op"() {replaced = "false"} : () -> ()
// CHECK:   "test.op"() {replaced = "false"} : () -> ()
// CHECK: }
