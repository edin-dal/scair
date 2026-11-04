// The lexer's messages reach the user, as they do from MLIR.

// RUN: mlir-opt %s --split-input-file --allow-unregistered-dialect 2>&1 | filecheck %s
// RUN: scair-opt %s --split-input-file --parsing-diagnostics --allow-unregistered-dialect 2>&1 | filecheck %s

"test.op"() {s = "\r"} : () -> ()
// CHECK: unknown escape in string literal

// -----

"test.op"() {s = "\4g"} : () -> ()
// CHECK: unknown escape in string literal

// -----

"test.op"() {"\q" = 1} : () -> ()
// CHECK: unknown escape in string literal

// -----

"test.op"() {s = "a
b"} : () -> ()
// CHECK: expected '"' in string literal
