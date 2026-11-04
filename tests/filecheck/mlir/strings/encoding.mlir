// Neither the input path nor the locale may change the bytes of a string.

// RUN: mlir-opt %s --allow-unregistered-dialect --mlir-print-op-generic | filecheck %s --strict-whitespace
// RUN: scair-opt %s --allow-unregistered-dialect --print-generic | filecheck %s --strict-whitespace
// RUN: scair-opt %s --allow-unregistered-dialect --print-generic -p 'mlir-opt[]' | filecheck %s --strict-whitespace
// RUN: LC_ALL=C scair-opt %s --allow-unregistered-dialect --print-generic -p 'mlir-opt[]' | filecheck %s --strict-whitespace
// RUN: scair-opt --allow-unregistered-dialect --print-generic < %s | filecheck %s --strict-whitespace
// RUN: LC_ALL=C scair-opt %s --allow-unregistered-dialect --print-generic | filecheck %s --strict-whitespace
// RUN: LC_ALL=C scair-opt --allow-unregistered-dialect --print-generic < %s | filecheck %s --strict-whitespace

// CHECK: "builtin.module"() ({

"test.op"() {s = "é"} : () -> ()
// CHECK-NEXT:   "test.op"() {s = "\C3\A9"} : () -> ()

"test.op"() {s = "€"} : () -> ()
// CHECK-NEXT:   "test.op"() {s = "\E2\82\AC"} : () -> ()

"test.op"() {s = "😀"} : () -> ()
// CHECK-NEXT:   "test.op"() {s = "\F0\9F\98\80"} : () -> ()

"test.op"() {s = "\FF"} : () -> ()
// CHECK-NEXT:   "test.op"() {s = "\FF"} : () -> ()

"test.op"() {s = "\C3\A9"} : () -> ()
// CHECK-NEXT:   "test.op"() {s = "\C3\A9"} : () -> ()

"test.op"() {s = "a\00é\FF€"} : () -> ()
// CHECK-NEXT:   "test.op"() {s = "a\00\C3\A9\FF\E2\82\AC"} : () -> ()

"test.op"() {"é" = "\FF"} : () -> ()
// CHECK-NEXT:   "test.op"() {"\C3\A9" = "\FF"} : () -> ()

"test.op"() {s = @"é"} : () -> ()
// CHECK-NEXT:   "test.op"() {s = @"\C3\A9"} : () -> ()

// CHECK-NEXT: }) : () -> ()
