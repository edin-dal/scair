// Operation names are strings, in the generic syntax.

// RUN: mlir-opt %s --allow-unregistered-dialect --mlir-print-op-generic | filecheck %s --strict-whitespace
// RUN: scair-opt %s --allow-unregistered-dialect --print-generic | filecheck %s --strict-whitespace
// RUN: scair-opt %s --allow-unregistered-dialect --print-generic -p 'mlir-opt[]' | filecheck %s --strict-whitespace

// CHECK: "builtin.module"() ({

// Escapes are decoded, so these name known forms.
"foo.bar"() : () -> ()
// CHECK-NEXT:   "foo.bar"() : () -> ()

"foo.b\61r"() : () -> ()
// CHECK-NEXT:   "foo.bar"() : () -> ()

"te\73t.op"() : () -> ()
// CHECK-NEXT:   "test.op"() : () -> ()

"\66oo.bar"() : () -> ()
// CHECK-NEXT:   "foo.bar"() : () -> ()

// Unusual but printable names.
"nodot"() : () -> ()
// CHECK-NEXT:   "nodot"() : () -> ()

"foo.b ar"() : () -> ()
// CHECK-NEXT:   "foo.b ar"() : () -> ()

"foo.bar baz"() : () -> ()
// CHECK-NEXT:   "foo.bar baz"() : () -> ()

"foo.$"() : () -> ()
// CHECK-NEXT:   "foo.$"() : () -> ()

"1.2"() : () -> ()
// CHECK-NEXT:   "1.2"() : () -> ()

// Names are escaped exactly as string values.
"foo.b\22r"() : () -> ()
// CHECK-NEXT:   "foo.b\22r"() : () -> ()

"foo.b\5Cr"() : () -> ()
// CHECK-NEXT:   "foo.b\\r"() : () -> ()

"foo.b\nr"() : () -> ()
// CHECK-NEXT:   "foo.b\0Ar"() : () -> ()

"foo.b\tr"() : () -> ()
// CHECK-NEXT:   "foo.b\09r"() : () -> ()

"foo.b\7Fr"() : () -> ()
// CHECK-NEXT:   "foo.b\7Fr"() : () -> ()

"foo.b\FFr"() : () -> ()
// CHECK-NEXT:   "foo.b\FFr"() : () -> ()

"foo.b\C3\A9r"() : () -> ()
// CHECK-NEXT:   "foo.b\C3\A9r"() : () -> ()

"foo.bér"() : () -> ()
// CHECK-NEXT:   "foo.b\C3\A9r"() : () -> ()

// CHECK-NEXT: }) : () -> ()
