// Decoding and canonical printing of every escape form.

// RUN: mlir-opt %s --allow-unregistered-dialect --mlir-print-op-generic | filecheck %s --strict-whitespace
// RUN: scair-opt %s --allow-unregistered-dialect --print-generic | filecheck %s --strict-whitespace

// CHECK: "builtin.module"() ({

// Named escapes print as hexadecimal escapes, except for the backslash.
"test.op"() {s = "\n"} : () -> ()
// CHECK-NEXT:   "test.op"() {s = "\0A"} : () -> ()

"test.op"() {s = "\t"} : () -> ()
// CHECK-NEXT:   "test.op"() {s = "\09"} : () -> ()

"test.op"() {s = "\""} : () -> ()
// CHECK-NEXT:   "test.op"() {s = "\22"} : () -> ()

"test.op"() {s = "\\"} : () -> ()
// CHECK-NEXT:   "test.op"() {s = "\\"} : () -> ()

"test.op"() {s = "a\nb\tc\"d\\e"} : () -> ()
// CHECK-NEXT:   "test.op"() {s = "a\0Ab\09c\22d\\e"} : () -> ()

// Hexadecimal escapes accept any case, and print in upper case.
"test.op"() {s = "\0a"} : () -> ()
// CHECK-NEXT:   "test.op"() {s = "\0A"} : () -> ()

"test.op"() {s = "\0A"} : () -> ()
// CHECK-NEXT:   "test.op"() {s = "\0A"} : () -> ()

"test.op"() {s = "\ff"} : () -> ()
// CHECK-NEXT:   "test.op"() {s = "\FF"} : () -> ()

"test.op"() {s = "\fF"} : () -> ()
// CHECK-NEXT:   "test.op"() {s = "\FF"} : () -> ()

"test.op"() {s = "\Ff"} : () -> ()
// CHECK-NEXT:   "test.op"() {s = "\FF"} : () -> ()

"test.op"() {s = "\FF"} : () -> ()
// CHECK-NEXT:   "test.op"() {s = "\FF"} : () -> ()

"test.op"() {s = "\aB"} : () -> ()
// CHECK-NEXT:   "test.op"() {s = "\AB"} : () -> ()

"test.op"() {s = "\Ab"} : () -> ()
// CHECK-NEXT:   "test.op"() {s = "\AB"} : () -> ()

"test.op"() {s = "\c3\A9"} : () -> ()
// CHECK-NEXT:   "test.op"() {s = "\C3\A9"} : () -> ()

// Escapes that denote printable bytes print as those bytes.
"test.op"() {s = "\41"} : () -> ()
// CHECK-NEXT:   "test.op"() {s = "A"} : () -> ()

"test.op"() {s = "\20"} : () -> ()
// CHECK-NEXT:   "test.op"() {s = " "} : () -> ()

"test.op"() {s = "\7E"} : () -> ()
// CHECK-NEXT:   "test.op"() {s = "~"} : () -> ()

"test.op"() {s = "\41\42\43"} : () -> ()
// CHECK-NEXT:   "test.op"() {s = "ABC"} : () -> ()

"test.op"() {s = "\61bc"} : () -> ()
// CHECK-NEXT:   "test.op"() {s = "abc"} : () -> ()

"test.op"() {s = "a\62c"} : () -> ()
// CHECK-NEXT:   "test.op"() {s = "abc"} : () -> ()

"test.op"() {s = "ab\63"} : () -> ()
// CHECK-NEXT:   "test.op"() {s = "abc"} : () -> ()

// The escapes of the quote and backslash bytes print in their canonical forms.
"test.op"() {s = "\22"} : () -> ()
// CHECK-NEXT:   "test.op"() {s = "\22"} : () -> ()

"test.op"() {s = "\5C"} : () -> ()
// CHECK-NEXT:   "test.op"() {s = "\\"} : () -> ()

"test.op"() {s = "\5c"} : () -> ()
// CHECK-NEXT:   "test.op"() {s = "\\"} : () -> ()

"test.op"() {s = "\5C\5C"} : () -> ()
// CHECK-NEXT:   "test.op"() {s = "\\\\"} : () -> ()

"test.op"() {s = "\22\22"} : () -> ()
// CHECK-NEXT:   "test.op"() {s = "\22\22"} : () -> ()

// An escape consumes exactly two digits; following digits are literal.
"test.op"() {s = "\418"} : () -> ()
// CHECK-NEXT:   "test.op"() {s = "A8"} : () -> ()

"test.op"() {s = "\41F"} : () -> ()
// CHECK-NEXT:   "test.op"() {s = "AF"} : () -> ()

"test.op"() {s = "\000"} : () -> ()
// CHECK-NEXT:   "test.op"() {s = "\000"} : () -> ()

"test.op"() {s = "\0000"} : () -> ()
// CHECK-NEXT:   "test.op"() {s = "\0000"} : () -> ()

"test.op"() {s = "\FFF"} : () -> ()
// CHECK-NEXT:   "test.op"() {s = "\FFF"} : () -> ()

"test.op"() {s = "\FFFF"} : () -> ()
// CHECK-NEXT:   "test.op"() {s = "\FFFF"} : () -> ()

"test.op"() {s = "\FF\FF"} : () -> ()
// CHECK-NEXT:   "test.op"() {s = "\FF\FF"} : () -> ()

"test.op"() {s = "\0A0A"} : () -> ()
// CHECK-NEXT:   "test.op"() {s = "\0A0A"} : () -> ()

// A decoded byte never starts a new escape.
"test.op"() {s = "\\41"} : () -> ()
// CHECK-NEXT:   "test.op"() {s = "\\41"} : () -> ()

"test.op"() {s = "\\\41"} : () -> ()
// CHECK-NEXT:   "test.op"() {s = "\\A"} : () -> ()

"test.op"() {s = "\\\\41"} : () -> ()
// CHECK-NEXT:   "test.op"() {s = "\\\\41"} : () -> ()

"test.op"() {s = "\5C41"} : () -> ()
// CHECK-NEXT:   "test.op"() {s = "\\41"} : () -> ()

"test.op"() {s = "\5Cn"} : () -> ()
// CHECK-NEXT:   "test.op"() {s = "\\n"} : () -> ()

"test.op"() {s = "\5C\5Cn"} : () -> ()
// CHECK-NEXT:   "test.op"() {s = "\\\\n"} : () -> ()

"test.op"() {s = "\\n"} : () -> ()
// CHECK-NEXT:   "test.op"() {s = "\\n"} : () -> ()

"test.op"() {s = "\\\n"} : () -> ()
// CHECK-NEXT:   "test.op"() {s = "\\\0A"} : () -> ()

"test.op"() {s = "\\t"} : () -> ()
// CHECK-NEXT:   "test.op"() {s = "\\t"} : () -> ()

"test.op"() {s = "\\\""} : () -> ()
// CHECK-NEXT:   "test.op"() {s = "\\\22"} : () -> ()

"test.op"() {s = "\"41"} : () -> ()
// CHECK-NEXT:   "test.op"() {s = "\2241"} : () -> ()

"test.op"() {s = "\n41"} : () -> ()
// CHECK-NEXT:   "test.op"() {s = "\0A41"} : () -> ()

"test.op"() {s = "\t41"} : () -> ()
// CHECK-NEXT:   "test.op"() {s = "\0941"} : () -> ()

"test.op"() {s = "\22\\"} : () -> ()
// CHECK-NEXT:   "test.op"() {s = "\22\\"} : () -> ()

"test.op"() {s = "\\\\"} : () -> ()
// CHECK-NEXT:   "test.op"() {s = "\\\\"} : () -> ()

"test.op"() {s = "\\\\\\"} : () -> ()
// CHECK-NEXT:   "test.op"() {s = "\\\\\\"} : () -> ()

// Named and hexadecimal escapes of the same byte are the same string.
"test.op"() {s = "a\nb"} : () -> ()
// CHECK-NEXT:   "test.op"() {s = "a\0Ab"} : () -> ()

"test.op"() {s = "a\0Ab"} : () -> ()
// CHECK-NEXT:   "test.op"() {s = "a\0Ab"} : () -> ()

"test.op"() {s = "a\tb"} : () -> ()
// CHECK-NEXT:   "test.op"() {s = "a\09b"} : () -> ()

"test.op"() {s = "a\09b"} : () -> ()
// CHECK-NEXT:   "test.op"() {s = "a\09b"} : () -> ()

"test.op"() {s = "a\"b"} : () -> ()
// CHECK-NEXT:   "test.op"() {s = "a\22b"} : () -> ()

"test.op"() {s = "a\22b"} : () -> ()
// CHECK-NEXT:   "test.op"() {s = "a\22b"} : () -> ()

"test.op"() {s = "a\\b"} : () -> ()
// CHECK-NEXT:   "test.op"() {s = "a\\b"} : () -> ()

"test.op"() {s = "a\5Cb"} : () -> ()
// CHECK-NEXT:   "test.op"() {s = "a\\b"} : () -> ()

// Escapes at either end of the string, and only escapes.
"test.op"() {s = "\00abc"} : () -> ()
// CHECK-NEXT:   "test.op"() {s = "\00abc"} : () -> ()

"test.op"() {s = "abc\00"} : () -> ()
// CHECK-NEXT:   "test.op"() {s = "abc\00"} : () -> ()

"test.op"() {s = "\00abc\00"} : () -> ()
// CHECK-NEXT:   "test.op"() {s = "\00abc\00"} : () -> ()

"test.op"() {s = "\00"} : () -> ()
// CHECK-NEXT:   "test.op"() {s = "\00"} : () -> ()

"test.op"() {s = "\00\00\00"} : () -> ()
// CHECK-NEXT:   "test.op"() {s = "\00\00\00"} : () -> ()

// CHECK-NEXT: }) : () -> ()
