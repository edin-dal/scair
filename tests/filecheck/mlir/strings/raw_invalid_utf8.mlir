// Raw bytes that are not valid UTF-8 are string contents, and are legal in comments.
// The input is generated, as filecheck cannot read such a test file.

// RUN: printf '%b' \
// RUN:   '// comment with \xFF\xFE in it\n' \
// RUN:   '"test.op"() {s = "\x80"} : () -> ()\n' \
// RUN:   '"test.op"() {s = "\xBF"} : () -> ()\n' \
// RUN:   '"test.op"() {s = "\xC3"} : () -> ()\n' \
// RUN:   '"test.op"() {s = "\xE2\x82"} : () -> ()\n' \
// RUN:   '"test.op"() {s = "\xF0\x9F\x98"} : () -> ()\n' \
// RUN:   '"test.op"() {s = "\xC0\x80"} : () -> ()\n' \
// RUN:   '"test.op"() {s = "\xC1\xBF"} : () -> ()\n' \
// RUN:   '"test.op"() {s = "\xED\xA0\x80"} : () -> ()\n' \
// RUN:   '"test.op"() {s = "\xF4\x90\x80\x80"} : () -> ()\n' \
// RUN:   '"test.op"() {s = "\xFE"} : () -> ()\n' \
// RUN:   '"test.op"() {s = "\xFF"} : () -> ()\n' \
// RUN:   '"test.op"() {s = "\x61\xFF\x62"} : () -> ()\n' \
// RUN:   '"test.op"() {s = "\xC3\xA9\xFF\xC3\xA9"} : () -> ()\n' \
// RUN:   '"test.op"() {s = "\xC3\xC3\xA9"} : () -> ()\n' \
// RUN:   '"test.op"() {s = "a"} : () -> () // trailing \x80\n' \
// RUN:   > %t
// RUN: mlir-opt %t --allow-unregistered-dialect --mlir-print-op-generic | filecheck %s --strict-whitespace
// RUN: scair-opt %t --allow-unregistered-dialect --print-generic | filecheck %s --strict-whitespace

// CHECK: "builtin.module"() ({

// CHECK-NEXT:   "test.op"() {s = "\80"} : () -> ()
// CHECK-NEXT:   "test.op"() {s = "\BF"} : () -> ()
// CHECK-NEXT:   "test.op"() {s = "\C3"} : () -> ()
// CHECK-NEXT:   "test.op"() {s = "\E2\82"} : () -> ()
// CHECK-NEXT:   "test.op"() {s = "\F0\9F\98"} : () -> ()
// CHECK-NEXT:   "test.op"() {s = "\C0\80"} : () -> ()
// CHECK-NEXT:   "test.op"() {s = "\C1\BF"} : () -> ()
// CHECK-NEXT:   "test.op"() {s = "\ED\A0\80"} : () -> ()
// CHECK-NEXT:   "test.op"() {s = "\F4\90\80\80"} : () -> ()
// CHECK-NEXT:   "test.op"() {s = "\FE"} : () -> ()
// CHECK-NEXT:   "test.op"() {s = "\FF"} : () -> ()
// CHECK-NEXT:   "test.op"() {s = "a\FFb"} : () -> ()
// CHECK-NEXT:   "test.op"() {s = "\C3\A9\FF\C3\A9"} : () -> ()
// CHECK-NEXT:   "test.op"() {s = "\C3\C3\A9"} : () -> ()
// CHECK-NEXT:   "test.op"() {s = "a"} : () -> ()
// CHECK-NEXT: }) : () -> ()
