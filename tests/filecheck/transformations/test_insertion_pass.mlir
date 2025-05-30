// RUN: scair-opt %s -p test-ins-pass --allow-unregistered-dialect | filecheck %s

"test.op"() : () -> ()
"test.op"() : () -> ()
"tobereplaced"() : () -> ()
"test.op"() : () -> ()
"test.op"() : () -> ()

// CHECK: builtin.module {
// CHECK:   "test.op"() {replaced = "false"} : () -> ()
// CHECK:   "test.op"() {replaced = "false"} : () -> ()
// CHECK:   "dummy1"() {replaced = "false"} : () -> ()
// CHECK:   "dummy2"() {replaced = "false"} : () -> ()
// CHECK:   "dummy3"() {replaced = "false"} : () -> ()
// CHECK:   "dummy4"() {replaced = "false"} : () -> ()
// CHECK:   "test.op"() {replaced = "false"} : () -> ()
// CHECK:   "test.op"() {replaced = "false"} : () -> ()
// CHECK: }
