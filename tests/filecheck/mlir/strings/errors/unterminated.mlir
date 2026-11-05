// A string cut short by the end of the input, at any point of an escape, is an
// error and not a crash. The CHECK lines come first, for each chunk to end on its
// string. Each echoed source line proves that its own chunk was rejected.

// RUN: mlir-opt %s --split-input-file --allow-unregistered-dialect 2>&1 | filecheck %s
// RUN: scair-opt %s --split-input-file --parsing-diagnostics --allow-unregistered-dialect 2>&1 | filecheck %s

// CHECK: {{: error: |^Parse error at }}
// CHECK: "test.op"() {{[{]}}s = "
"test.op"() {s = "
// -----

// CHECK: {{: error: |^Parse error at }}
// CHECK: "test.op"() {{[{]}}s = "abc
"test.op"() {s = "abc
// -----

// CHECK: {{: error: |^Parse error at }}
// CHECK: "test.op"() {{[{]}}s = "abc\
"test.op"() {s = "abc\
// -----

// CHECK: {{: error: |^Parse error at }}
// CHECK: "test.op"() {{[{]}}s = "abc\4
"test.op"() {s = "abc\4
// -----

// CHECK: {{: error: |^Parse error at }}
// CHECK: "test.op"() {{[{]}}s = "abc\41
"test.op"() {s = "abc\41
// -----

// CHECK: {{: error: |^Parse error at }}
// CHECK: "test.op"() {{[{]}}s = "abc\"
"test.op"() {s = "abc\"
// -----

// CHECK: {{: error: |^Parse error at }}
// CHECK: "test.op"() {{[{]}}s = "abc\n
"test.op"() {s = "abc\n
// -----

// CHECK: {{: error: |^Parse error at }}
// CHECK: "test.op"() {{[{]}}s = "abc\\
"test.op"() {s = "abc\\
// -----

// CHECK: {{: error: |^Parse error at }}
// CHECK: "test.op"() {{[{]}}"abc\
"test.op"() {"abc\
// -----

// CHECK: {{: error: |^Parse error at }}
// CHECK: "test.op"() {{[{]}}s = @"abc\
"test.op"() {s = @"abc\
// -----

// CHECK: {{: error: |^Parse error at }}
// CHECK: "foo\
"foo\
