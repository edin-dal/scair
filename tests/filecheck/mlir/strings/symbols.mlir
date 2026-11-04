// Symbol names are strings: quoted unless they are bare identifiers.

// RUN: mlir-opt %s --allow-unregistered-dialect --mlir-print-op-generic | filecheck %s --strict-whitespace
// RUN: scair-opt %s --allow-unregistered-dialect --print-generic | filecheck %s --strict-whitespace
// RUN: scair-opt %s --allow-unregistered-dialect --print-generic -p 'mlir-opt[]' | filecheck %s --strict-whitespace

// CHECK: "builtin.module"() ({

// Quotes are dropped when the symbol is a bare identifier, however it was written.
"test.op"() {s = @"plain"} : () -> ()
// CHECK-NEXT:   "test.op"() {s = @plain} : () -> ()

"test.op"() {s = @"\70lain"} : () -> ()
// CHECK-NEXT:   "test.op"() {s = @plain} : () -> ()

"test.op"() {s = @"a.b"} : () -> ()
// CHECK-NEXT:   "test.op"() {s = @a.b} : () -> ()

"test.op"() {s = @"a$b"} : () -> ()
// CHECK-NEXT:   "test.op"() {s = @a$b} : () -> ()

"test.op"() {s = @"_a"} : () -> ()
// CHECK-NEXT:   "test.op"() {s = @_a} : () -> ()

// Symbols that are not bare identifiers stay quoted.
"test.op"() {s = @"a b"} : () -> ()
// CHECK-NEXT:   "test.op"() {s = @"a b"} : () -> ()

"test.op"() {s = @"a-b"} : () -> ()
// CHECK-NEXT:   "test.op"() {s = @"a-b"} : () -> ()

"test.op"() {s = @"0a"} : () -> ()
// CHECK-NEXT:   "test.op"() {s = @"0a"} : () -> ()

"test.op"() {s = @".a"} : () -> ()
// CHECK-NEXT:   "test.op"() {s = @".a"} : () -> ()

"test.op"() {s = @"$a"} : () -> ()
// CHECK-NEXT:   "test.op"() {s = @"$a"} : () -> ()

"test.op"() {s = @" "} : () -> ()
// CHECK-NEXT:   "test.op"() {s = @" "} : () -> ()

"test.op"() {s = @"a::b"} : () -> ()
// CHECK-NEXT:   "test.op"() {s = @"a::b"} : () -> ()

"test.op"() {s = @"@a"} : () -> ()
// CHECK-NEXT:   "test.op"() {s = @"@a"} : () -> ()

// Symbols are escaped exactly as string values.
"test.op"() {s = @"\22"} : () -> ()
// CHECK-NEXT:   "test.op"() {s = @"\22"} : () -> ()

"test.op"() {s = @"a\22b"} : () -> ()
// CHECK-NEXT:   "test.op"() {s = @"a\22b"} : () -> ()

"test.op"() {s = @"\5C"} : () -> ()
// CHECK-NEXT:   "test.op"() {s = @"\\"} : () -> ()

"test.op"() {s = @"a\\b"} : () -> ()
// CHECK-NEXT:   "test.op"() {s = @"a\\b"} : () -> ()

"test.op"() {s = @"a\nb"} : () -> ()
// CHECK-NEXT:   "test.op"() {s = @"a\0Ab"} : () -> ()

"test.op"() {s = @"a\tb"} : () -> ()
// CHECK-NEXT:   "test.op"() {s = @"a\09b"} : () -> ()

"test.op"() {s = @"\00"} : () -> ()
// CHECK-NEXT:   "test.op"() {s = @"\00"} : () -> ()

"test.op"() {s = @"\7F"} : () -> ()
// CHECK-NEXT:   "test.op"() {s = @"\7F"} : () -> ()

"test.op"() {s = @"\FF"} : () -> ()
// CHECK-NEXT:   "test.op"() {s = @"\FF"} : () -> ()

"test.op"() {s = @"\C3\A9"} : () -> ()
// CHECK-NEXT:   "test.op"() {s = @"\C3\A9"} : () -> ()

"test.op"() {s = @"é"} : () -> ()
// CHECK-NEXT:   "test.op"() {s = @"\C3\A9"} : () -> ()

"test.op"() {s = @"😀"} : () -> ()
// CHECK-NEXT:   "test.op"() {s = @"\F0\9F\98\80"} : () -> ()

// Nested references.
"test.op"() {s = @"a b"::@"c d"} : () -> ()
// CHECK-NEXT:   "test.op"() {s = @"a b"::@"c d"} : () -> ()

"test.op"() {s = @a::@"\FF"::@b} : () -> ()
// CHECK-NEXT:   "test.op"() {s = @a::@"\FF"::@b} : () -> ()

"test.op"() {s = @"a"::@"b"} : () -> ()
// CHECK-NEXT:   "test.op"() {s = @a::@b} : () -> ()

"test.op"() {s = @"a\nb"::@"\22"::@"\\"} : () -> ()
// CHECK-NEXT:   "test.op"() {s = @"a\0Ab"::@"\22"::@"\\"} : () -> ()

// In containers, and as a property.
"test.op"() {s = [@"a b", @c]} : () -> ()
// CHECK-NEXT:   "test.op"() {s = [@"a b", @c]} : () -> ()

"test.op"() <{s = @"\FF"}> : () -> ()
// CHECK-NEXT:   "test.op"() <{s = @"\FF"}> : () -> ()

// Characters at the boundary of a bare identifier, first in the symbol, then later.
"test.op"() {s = @"1a"} : () -> ()
// CHECK-NEXT:   "test.op"() {s = @"1a"} : () -> ()
"test.op"() {s = @"_a"} : () -> ()
// CHECK-NEXT:   "test.op"() {s = @_a} : () -> ()
"test.op"() {s = @"$a"} : () -> ()
// CHECK-NEXT:   "test.op"() {s = @"$a"} : () -> ()
"test.op"() {s = @".a"} : () -> ()
// CHECK-NEXT:   "test.op"() {s = @".a"} : () -> ()
"test.op"() {s = @"-a"} : () -> ()
// CHECK-NEXT:   "test.op"() {s = @"-a"} : () -> ()
"test.op"() {s = @" a"} : () -> ()
// CHECK-NEXT:   "test.op"() {s = @" a"} : () -> ()
"test.op"() {s = @"a1"} : () -> ()
// CHECK-NEXT:   "test.op"() {s = @a1} : () -> ()
"test.op"() {s = @"a_"} : () -> ()
// CHECK-NEXT:   "test.op"() {s = @a_} : () -> ()
"test.op"() {s = @"a$"} : () -> ()
// CHECK-NEXT:   "test.op"() {s = @a$} : () -> ()
"test.op"() {s = @"a."} : () -> ()
// CHECK-NEXT:   "test.op"() {s = @a.} : () -> ()
"test.op"() {s = @"a-"} : () -> ()
// CHECK-NEXT:   "test.op"() {s = @"a-"} : () -> ()
"test.op"() {s = @"a "} : () -> ()
// CHECK-NEXT:   "test.op"() {s = @"a "} : () -> ()

// CHECK-NEXT: }) : () -> ()
