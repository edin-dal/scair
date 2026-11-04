// Keys are compared after decoding, and must not be empty. Neither may operation
// names, which also exclude NUL. A symbol with a dash must be quoted. Each echoed source line proves that
// its own chunk was rejected.

// RUN: mlir-opt %s --split-input-file --allow-unregistered-dialect 2>&1 | filecheck %s
// RUN: scair-opt %s --split-input-file --parsing-diagnostics --allow-unregistered-dialect 2>&1 | filecheck %s

"test.op"() {a = 1, a = 2} : () -> ()
// CHECK: {{: error: |^Parse error at }}
// CHECK: {a = 1, a = 2}

// -----

"test.op"() {a = 1, "a" = 2} : () -> ()
// CHECK: {{: error: |^Parse error at }}
// CHECK: {a = 1, "a" = 2}

// -----

"test.op"() {"a" = 1, "\61" = 2} : () -> ()
// CHECK: {{: error: |^Parse error at }}
// CHECK: {"a" = 1, "\61" = 2}

// -----

"test.op"() {s = {"a" = 1, "\61" = 2}} : () -> ()
// CHECK: {{: error: |^Parse error at }}
// CHECK: {s = {"a" = 1, "\61" = 2}}

// -----

"test.op"() {"" = 1} : () -> ()
// CHECK: {{: error: |^Parse error at }}
// CHECK: {"" = 1}

// -----

""() : () -> ()
// CHECK: {{: error: |^Parse error at }}
// CHECK: ""() : () -> ()

// -----

"foo.b\00r"() : () -> ()
// CHECK: {{: error: |^Parse error at }}
// CHECK: "foo.b\00r"() : () -> ()

// -----

"test.op"() {s = @a-b} : () -> ()
// CHECK: {{: error: |^Parse error at }}
// CHECK: {s = @a-b}
