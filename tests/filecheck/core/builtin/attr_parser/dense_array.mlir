// RUN: scair-opt %s | filecheck %s

"test.op"() {hello = array<i32: 0>} : () -> ()
"test.op"() {hello = array<i32: 1>} : () -> ()
"test.op"() {hello = array<i32: -1>} : () -> ()

// CHECK: "test.op"() {hello = array<i32: 0>} : () -> ()
// CHECK: "test.op"() {hello = array<i32: 1>} : () -> ()
// CHECK: "test.op"() {hello = array<i32: -1>} : () -> ()