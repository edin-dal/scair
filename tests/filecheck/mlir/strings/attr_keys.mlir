// Attribute keys are strings: quoted unless they are bare identifiers.

// RUN: mlir-opt %s --allow-unregistered-dialect --mlir-print-op-generic | filecheck %s --strict-whitespace
// RUN: scair-opt %s --allow-unregistered-dialect --print-generic | filecheck %s --strict-whitespace
// RUN: scair-opt %s --allow-unregistered-dialect --print-generic -p 'mlir-opt[]' | filecheck %s --strict-whitespace

// CHECK: "builtin.module"() ({

// Quotes are dropped when the key is a bare identifier, however it was written.
"test.op"() {"s" = 1} : () -> ()
// CHECK-NEXT:   "test.op"() {s = 1 : i64} : () -> ()

"test.op"() {"\73" = 1} : () -> ()
// CHECK-NEXT:   "test.op"() {s = 1 : i64} : () -> ()

"test.op"() {"a.b" = 1} : () -> ()
// CHECK-NEXT:   "test.op"() {a.b = 1 : i64} : () -> ()

"test.op"() {"a$b" = 1} : () -> ()
// CHECK-NEXT:   "test.op"() {a$b = 1 : i64} : () -> ()

"test.op"() {"_a" = 1} : () -> ()
// CHECK-NEXT:   "test.op"() {_a = 1 : i64} : () -> ()

"test.op"() {"A0" = 1} : () -> ()
// CHECK-NEXT:   "test.op"() {A0 = 1 : i64} : () -> ()

"test.op"() {"func" = 1} : () -> ()
// CHECK-NEXT:   "test.op"() {func = 1 : i64} : () -> ()

"test.op"() {"true" = 1} : () -> ()
// CHECK-NEXT:   "test.op"() {true = 1 : i64} : () -> ()

"test.op"() {"dense" = 1} : () -> ()
// CHECK-NEXT:   "test.op"() {dense = 1 : i64} : () -> ()

"test.op"() {"unit" = 1} : () -> ()
// CHECK-NEXT:   "test.op"() {unit = 1 : i64} : () -> ()

// Keys that are not bare identifiers stay quoted.
"test.op"() {"a b" = 1} : () -> ()
// CHECK-NEXT:   "test.op"() {"a b" = 1 : i64} : () -> ()

"test.op"() {"a-b" = 1} : () -> ()
// CHECK-NEXT:   "test.op"() {"a-b" = 1 : i64} : () -> ()

"test.op"() {"0a" = 1} : () -> ()
// CHECK-NEXT:   "test.op"() {"0a" = 1 : i64} : () -> ()

"test.op"() {".a" = 1} : () -> ()
// CHECK-NEXT:   "test.op"() {".a" = 1 : i64} : () -> ()

"test.op"() {"$a" = 1} : () -> ()
// CHECK-NEXT:   "test.op"() {"$a" = 1 : i64} : () -> ()

"test.op"() {"a/b" = 1} : () -> ()
// CHECK-NEXT:   "test.op"() {"a/b" = 1 : i64} : () -> ()

"test.op"() {"a:b" = 1} : () -> ()
// CHECK-NEXT:   "test.op"() {"a:b" = 1 : i64} : () -> ()

"test.op"() {"a=b" = 1} : () -> ()
// CHECK-NEXT:   "test.op"() {"a=b" = 1 : i64} : () -> ()

"test.op"() {" " = 1} : () -> ()
// CHECK-NEXT:   "test.op"() {" " = 1 : i64} : () -> ()

"test.op"() {"a " = 1} : () -> ()
// CHECK-NEXT:   "test.op"() {"a " = 1 : i64} : () -> ()

// Keys are escaped exactly as string values.
"test.op"() {"\22" = 1} : () -> ()
// CHECK-NEXT:   "test.op"() {"\22" = 1 : i64} : () -> ()

"test.op"() {"a\22b" = 1} : () -> ()
// CHECK-NEXT:   "test.op"() {"a\22b" = 1 : i64} : () -> ()

"test.op"() {"\5C" = 1} : () -> ()
// CHECK-NEXT:   "test.op"() {"\\" = 1 : i64} : () -> ()

"test.op"() {"a\\b" = 1} : () -> ()
// CHECK-NEXT:   "test.op"() {"a\\b" = 1 : i64} : () -> ()

"test.op"() {"a\nb" = 1} : () -> ()
// CHECK-NEXT:   "test.op"() {"a\0Ab" = 1 : i64} : () -> ()

"test.op"() {"a\tb" = 1} : () -> ()
// CHECK-NEXT:   "test.op"() {"a\09b" = 1 : i64} : () -> ()

"test.op"() {"\00" = 1} : () -> ()
// CHECK-NEXT:   "test.op"() {"\00" = 1 : i64} : () -> ()

"test.op"() {"a\00b" = 1} : () -> ()
// CHECK-NEXT:   "test.op"() {"a\00b" = 1 : i64} : () -> ()

"test.op"() {"\7F" = 1} : () -> ()
// CHECK-NEXT:   "test.op"() {"\7F" = 1 : i64} : () -> ()

"test.op"() {"\FF" = 1} : () -> ()
// CHECK-NEXT:   "test.op"() {"\FF" = 1 : i64} : () -> ()

"test.op"() {"\C3\A9" = 1} : () -> ()
// CHECK-NEXT:   "test.op"() {"\C3\A9" = 1 : i64} : () -> ()

"test.op"() {"é" = 1} : () -> ()
// CHECK-NEXT:   "test.op"() {"\C3\A9" = 1 : i64} : () -> ()

"test.op"() {"😀" = 1} : () -> ()
// CHECK-NEXT:   "test.op"() {"\F0\9F\98\80" = 1 : i64} : () -> ()

// Values are unaffected by their key being quoted.
"test.op"() {"a b" = "\FF"} : () -> ()
// CHECK-NEXT:   "test.op"() {"a b" = "\FF"} : () -> ()

// Unit attributes, and properties.
"test.op"() {"a b"} : () -> ()
// CHECK-NEXT:   "test.op"() {"a b"} : () -> ()

"test.op"() {"\5C"} : () -> ()
// CHECK-NEXT:   "test.op"() {"\\"} : () -> ()

"test.op"() <{"a b" = 1}> : () -> ()
// CHECK-NEXT:   "test.op"() <{"a b" = 1 : i64}> : () -> ()

// Characters at the boundary of a bare identifier, first in the key, then later.
"test.op"() {"1a" = 1} : () -> ()
// CHECK-NEXT:   "test.op"() {"1a" = 1 : i64} : () -> ()
"test.op"() {"_a" = 1} : () -> ()
// CHECK-NEXT:   "test.op"() {_a = 1 : i64} : () -> ()
"test.op"() {"$a" = 1} : () -> ()
// CHECK-NEXT:   "test.op"() {"$a" = 1 : i64} : () -> ()
"test.op"() {".a" = 1} : () -> ()
// CHECK-NEXT:   "test.op"() {".a" = 1 : i64} : () -> ()
"test.op"() {"-a" = 1} : () -> ()
// CHECK-NEXT:   "test.op"() {"-a" = 1 : i64} : () -> ()
"test.op"() {" a" = 1} : () -> ()
// CHECK-NEXT:   "test.op"() {" a" = 1 : i64} : () -> ()
"test.op"() {"a1" = 1} : () -> ()
// CHECK-NEXT:   "test.op"() {a1 = 1 : i64} : () -> ()
"test.op"() {"a_" = 1} : () -> ()
// CHECK-NEXT:   "test.op"() {a_ = 1 : i64} : () -> ()
"test.op"() {"a$" = 1} : () -> ()
// CHECK-NEXT:   "test.op"() {a$ = 1 : i64} : () -> ()
"test.op"() {"a." = 1} : () -> ()
// CHECK-NEXT:   "test.op"() {a. = 1 : i64} : () -> ()
"test.op"() {"a-" = 1} : () -> ()
// CHECK-NEXT:   "test.op"() {"a-" = 1 : i64} : () -> ()
"test.op"() {"a " = 1} : () -> ()
// CHECK-NEXT:   "test.op"() {"a " = 1 : i64} : () -> ()

// CHECK-NEXT: }) : () -> ()
