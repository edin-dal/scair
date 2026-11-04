// A string attribute has a type, which must not be lost.

// RUN: mlir-opt %s --allow-unregistered-dialect --mlir-print-op-generic | filecheck %s --strict-whitespace
// RUN: scair-opt %s --allow-unregistered-dialect --print-generic | filecheck %s --strict-whitespace

// CHECK: "builtin.module"() ({

// Types are kept, whatever the contents.
"test.op"() {s = "a" : i32} : () -> ()
// CHECK-NEXT:   "test.op"() {s = "a" : i32} : () -> ()

"test.op"() {s = "" : i1} : () -> ()
// CHECK-NEXT:   "test.op"() {s = "" : i1} : () -> ()

"test.op"() {s = "\FF" : i8} : () -> ()
// CHECK-NEXT:   "test.op"() {s = "\FF" : i8} : () -> ()

"test.op"() {s = "\22" : index} : () -> ()
// CHECK-NEXT:   "test.op"() {s = "\22" : index} : () -> ()

"test.op"() {s = "\FF" : tensor<1xi8>} : () -> ()
// CHECK-NEXT:   "test.op"() {s = "\FF" : tensor<1xi8>} : () -> ()

"test.op"() {s = "a" : memref<2xf32>} : () -> ()
// CHECK-NEXT:   "test.op"() {s = "a" : memref<2xf32>} : () -> ()

"test.op"() {s = "a" : f32} : () -> ()
// CHECK-NEXT:   "test.op"() {s = "a" : f32} : () -> ()

"test.op"() {s = "a" : () -> ()} : () -> ()
// CHECK-NEXT:   "test.op"() {s = "a" : () -> ()} : () -> ()

// The none type is the default, and is elided.
"test.op"() {s = "\FF" : none} : () -> ()
// CHECK-NEXT:   "test.op"() {s = "\FF"} : () -> ()

// In containers.
"test.op"() {s = ["a" : i32, "b"]} : () -> ()
// CHECK-NEXT:   "test.op"() {s = ["a" : i32, "b"]} : () -> ()

"test.op"() {"a b" = "c" : i8} : () -> ()
// CHECK-NEXT:   "test.op"() {"a b" = "c" : i8} : () -> ()

"test.op"() <{s = "a" : i32}> : () -> ()
// CHECK-NEXT:   "test.op"() <{s = "a" : i32}> : () -> ()

// CHECK-NEXT: }) : () -> ()
