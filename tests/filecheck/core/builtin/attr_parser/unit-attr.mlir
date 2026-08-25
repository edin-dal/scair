// RUN: scair-opt %s | filecheck %s
// RUN: scair-opt %s | scair-opt | filecheck %s

// A unit attribute is an entry with no value; it is both parsed and printed as a
// bare key. The explicit `= unit` spelling parses too, and prints back as the
// shorthand.

"test.op"() {flag} : () -> ()
"test.op"() {flag = unit} : () -> ()
"test.op"() {"quoted_flag"} : () -> ()
"test.op"() {hello = "world", flag, number = 42 : i32} : () -> ()
"test.op"() <{prop_flag}> {attr_flag} : () -> ()
"test.op"() {nested = [unit, "world"]} : () -> ()

// CHECK:      builtin.module {
// CHECK-NEXT:   "test.op"() {flag} : () -> ()
// CHECK-NEXT:   "test.op"() {flag} : () -> ()
// CHECK-NEXT:   "test.op"() {quoted_flag} : () -> ()
// CHECK-NEXT:   "test.op"() {hello = "world", flag, number = 42 : i32} : () -> ()
// CHECK-NEXT:   "test.op"() <{prop_flag}> {attr_flag} : () -> ()
// CHECK-NEXT:   "test.op"() {nested = [unit, "world"]} : () -> ()
// CHECK-NEXT: }
