// Raw bytes that are not valid UTF-8 are string contents, and are legal in comments.
// The RUN line decodes the hexadecimal escapes below, as filecheck cannot read
// such a test file.

// RUN: printf '%b' "$(grep -v '^// CHECK' %s)" > %t
// RUN: mlir-opt %t --allow-unregistered-dialect --mlir-print-op-generic | filecheck %s --strict-whitespace
// RUN: scair-opt %t --allow-unregistered-dialect --print-generic | filecheck %s --strict-whitespace
// RUN: scair-opt %t --allow-unregistered-dialect --print-generic -p 'mlir-opt[]' | filecheck %s --strict-whitespace

// CHECK: "builtin.module"() ({

// comment with \xFF\xFE in it

"test.op"() {s = "\x80"} : () -> ()
// CHECK-NEXT:   "test.op"() {s = "\80"} : () -> ()

"test.op"() {s = "\xBF"} : () -> ()
// CHECK-NEXT:   "test.op"() {s = "\BF"} : () -> ()

"test.op"() {s = "\xC3"} : () -> ()
// CHECK-NEXT:   "test.op"() {s = "\C3"} : () -> ()

"test.op"() {s = "\xE2\x82"} : () -> ()
// CHECK-NEXT:   "test.op"() {s = "\E2\82"} : () -> ()

"test.op"() {s = "\xF0\x9F\x98"} : () -> ()
// CHECK-NEXT:   "test.op"() {s = "\F0\9F\98"} : () -> ()

"test.op"() {s = "\xC0\x80"} : () -> ()
// CHECK-NEXT:   "test.op"() {s = "\C0\80"} : () -> ()

"test.op"() {s = "\xC1\xBF"} : () -> ()
// CHECK-NEXT:   "test.op"() {s = "\C1\BF"} : () -> ()

"test.op"() {s = "\xED\xA0\x80"} : () -> ()
// CHECK-NEXT:   "test.op"() {s = "\ED\A0\80"} : () -> ()

"test.op"() {s = "\xF4\x90\x80\x80"} : () -> ()
// CHECK-NEXT:   "test.op"() {s = "\F4\90\80\80"} : () -> ()

"test.op"() {s = "\xFE"} : () -> ()
// CHECK-NEXT:   "test.op"() {s = "\FE"} : () -> ()

"test.op"() {s = "\xFF"} : () -> ()
// CHECK-NEXT:   "test.op"() {s = "\FF"} : () -> ()

"test.op"() {s = "\x61\xFF\x62"} : () -> ()
// CHECK-NEXT:   "test.op"() {s = "a\FFb"} : () -> ()

"test.op"() {s = "\xC3\xA9\xFF\xC3\xA9"} : () -> ()
// CHECK-NEXT:   "test.op"() {s = "\C3\A9\FF\C3\A9"} : () -> ()

"test.op"() {s = "\xC3\xC3\xA9"} : () -> ()
// CHECK-NEXT:   "test.op"() {s = "\C3\C3\A9"} : () -> ()

"test.op"() {s = "a"} : () -> () // trailing \x80
// CHECK-NEXT:   "test.op"() {s = "a"} : () -> ()

// CHECK-NEXT: }) : () -> ()
