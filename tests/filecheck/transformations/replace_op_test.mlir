// RUN: scair-opt %s -p test-rep-pass --allow-unregistered-dialect | filecheck %s

"test.op"(%0) : (i32) -> ()
"test.op"(%0) : (i32) -> ()
%0, %1 = "tobereplaced"() : () -> (i32, i64)
"test.op"(%1) : (i64) -> ()
"test.op"(%1) : (i64) -> ()

// CHECK: builtin.module {
// CHECK:   "test.op"(%0) {replaced = "false"} : ("replaced(i32)") -> ()
// CHECK:   "test.op"(%0) {replaced = "false"} : ("replaced(i32)") -> ()
// CHECK:   "replacedOp"() ({
// CHECK:     "dummy-op1"() : () -> ()
// CHECK:     "dummy-op2"() : () -> ()
// CHECK:     %0, %1 = "dummy-return"() : () -> ("replaced(i32)", "replaced(i64)")
// CHECK:   }) : () -> ()
// CHECK:   "test.op"(%1) {replaced = "false"} : ("replaced(i64)") -> ()
// CHECK:   "test.op"(%1) {replaced = "false"} : ("replaced(i64)") -> ()
// CHECK: }
