// A vertical tab, a form feed or a CRLF line break ends a string with an error. The
// RUN line decodes the hexadecimal escapes below, to keep those bytes out of the
// repository.

// RUN: printf '%b' "$(grep -v '^// CHECK' %s)" > %t
// RUN: mlir-opt %t --split-input-file --allow-unregistered-dialect 2>&1 | filecheck %s
// RUN: scair-opt %t --split-input-file --parsing-diagnostics --allow-unregistered-dialect 2>&1 | filecheck %s

"test.op"() {s = "a\x0Bb"} : () -> ()
// CHECK: {{: error: |^Parse error at }}

// -----

"test.op"() {s = "a\x0Cb"} : () -> ()
// CHECK: {{: error: |^Parse error at }}

// -----

"test.op"() {s = "a\r\nb"} : () -> ()
// CHECK: {{: error: |^Parse error at }}
