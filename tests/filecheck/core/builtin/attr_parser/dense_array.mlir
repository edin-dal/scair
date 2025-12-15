// RUN: scair-opt %s | filecheck %s

"test.op"() {hello = array<i32: 0>} : () -> ()
"test.op"() {hello = array<i32: 1>} : () -> ()
"test.op"() {hello = array<i32: -1>} : () -> ()

// CHECK: "test.op"() {hello = array<i32: 0>} : () -> ()
// CHECK: "test.op"() {hello = array<i32: 1>} : () -> ()
// CHECK: "test.op"() {hello = array<i32: -1>} : () -> ()

%0 = "arith.constant"() <{value = 5}> : () -> i32
%1 = "arith.constant"() <{value = 5}> : () -> i32
%2 = "arith.addi"(%0, %1) : (i32, i32) -> i32
func.call @print(%2) : (i32) -> ()