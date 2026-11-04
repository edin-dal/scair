// Only the escapes of the MLIR lexer are accepted, in every position, and a raw
// newline ends a string with an error. Each echoed source line proves that its
// own chunk was rejected.

// RUN: mlir-opt %s --split-input-file --allow-unregistered-dialect 2>&1 | filecheck %s
// RUN: scair-opt %s --split-input-file --parsing-diagnostics --allow-unregistered-dialect 2>&1 | filecheck %s

"test.op"() {s = "\r"} : () -> ()
// CHECK: {{: error: |^Parse error at }}
// CHECK: {s = "\r"} : () -> ()

// -----

"test.op"() {s = "\v"} : () -> ()
// CHECK: {{: error: |^Parse error at }}
// CHECK: {s = "\v"} : () -> ()

// -----

"test.op"() {s = "\f"} : () -> ()
// CHECK: {{: error: |^Parse error at }}
// CHECK: {s = "\f"} : () -> ()

// -----

"test.op"() {s = "\b"} : () -> ()
// CHECK: {{: error: |^Parse error at }}
// CHECK: {s = "\b"} : () -> ()

// -----

"test.op"() {s = "\e"} : () -> ()
// CHECK: {{: error: |^Parse error at }}
// CHECK: {s = "\e"} : () -> ()

// -----

"test.op"() {s = "\a"} : () -> ()
// CHECK: {{: error: |^Parse error at }}
// CHECK: {s = "\a"} : () -> ()

// -----

"test.op"() {s = "\0"} : () -> ()
// CHECK: {{: error: |^Parse error at }}
// CHECK: {s = "\0"} : () -> ()

// -----

"test.op"() {s = "\9"} : () -> ()
// CHECK: {{: error: |^Parse error at }}
// CHECK: {s = "\9"} : () -> ()

// -----

"test.op"() {s = "\x41"} : () -> ()
// CHECK: {{: error: |^Parse error at }}
// CHECK: {s = "\x41"} : () -> ()

// -----

"test.op"() {s = "\u0041"} : () -> ()
// CHECK: {{: error: |^Parse error at }}
// CHECK: {s = "\u0041"} : () -> ()

// -----

"test.op"() {s = "\N"} : () -> ()
// CHECK: {{: error: |^Parse error at }}
// CHECK: {s = "\N"} : () -> ()

// -----

"test.op"() {s = "\T"} : () -> ()
// CHECK: {{: error: |^Parse error at }}
// CHECK: {s = "\T"} : () -> ()

// -----

"test.op"() {s = "\'"} : () -> ()
// CHECK: {{: error: |^Parse error at }}
// CHECK: {s = "\'"} : () -> ()

// -----

"test.op"() {s = "\ "} : () -> ()
// CHECK: {{: error: |^Parse error at }}
// CHECK: {s = "\ "} : () -> ()

// -----

"test.op"() {s = "\/"} : () -> ()
// CHECK: {{: error: |^Parse error at }}
// CHECK: {s = "\/"} : () -> ()

// -----

"test.op"() {s = "\?"} : () -> ()
// CHECK: {{: error: |^Parse error at }}
// CHECK: {s = "\?"} : () -> ()

// -----

"test.op"() {s = "\4g"} : () -> ()
// CHECK: {{: error: |^Parse error at }}
// CHECK: {s = "\4g"} : () -> ()

// -----

"test.op"() {s = "\g4"} : () -> ()
// CHECK: {{: error: |^Parse error at }}
// CHECK: {s = "\g4"} : () -> ()

// -----

"test.op"() {s = "\A"} : () -> ()
// CHECK: {{: error: |^Parse error at }}
// CHECK: {s = "\A"} : () -> ()

// -----

"test.op"() {s = "\Ag"} : () -> ()
// CHECK: {{: error: |^Parse error at }}
// CHECK: {s = "\Ag"} : () -> ()

// -----

"test.op"() {s = "\é"} : () -> ()
// CHECK: {{: error: |^Parse error at }}
// CHECK: {s = "\{{.+}}"} : () -> ()

// -----

"test.op"() {s = "a\qb"} : () -> ()
// CHECK: {{: error: |^Parse error at }}
// CHECK: {s = "a\qb"} : () -> ()

// -----

"test.op"() {s = "\\\q"} : () -> ()
// CHECK: {{: error: |^Parse error at }}
// CHECK: {s = "\\\q"} : () -> ()

// -----

"test.op"() {"\q" = 1} : () -> ()
// CHECK: {{: error: |^Parse error at }}
// CHECK: {"\q" = 1} : () -> ()

// -----

"test.op"() {s = @"\q"} : () -> ()
// CHECK: {{: error: |^Parse error at }}
// CHECK: s = @"\q"} : () -> ()

// -----

"foo.\q"() : () -> ()
// CHECK: {{: error: |^Parse error at }}
// CHECK: "foo.\q"() : () -> ()

// -----

"test.op"() {s = ["ok", "\q"]} : () -> ()
// CHECK: {{: error: |^Parse error at }}
// CHECK: ["ok", "\q"]

// -----

"test.op"() {s = "a
b"} : () -> ()
// CHECK: {{: error: |^Parse error at }}
// CHECK: {s = "a
