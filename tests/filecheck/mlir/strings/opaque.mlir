// Attributes and types of unknown dialects hold strings ScaIR must not alter.

// RUN: mlir-opt %s --allow-unregistered-dialect --mlir-print-op-generic | filecheck %s --strict-whitespace
// RUN: scair-opt %s --allow-unregistered-dialect --print-generic | filecheck %s --strict-whitespace

// CHECK: "builtin.module"() ({

// The body of an attribute of an unknown dialect is kept as written.
"test.op"() {s = #foo<"\FF">} : () -> ()
// CHECK-NEXT:   "test.op"() {s = #foo<"\FF">} : () -> ()

"test.op"() {s = #foo.bar<"\FF">} : () -> ()
// CHECK-NEXT:   "test.op"() {s = #foo.bar<"\FF">} : () -> ()

"test.op"() {s = #foo.bar<"\\">} : () -> ()
// CHECK-NEXT:   "test.op"() {s = #foo.bar<"\\">} : () -> ()

"test.op"() {s = #foo.bar<"a\22b">} : () -> ()
// CHECK-NEXT:   "test.op"() {s = #foo.bar<"a\22b">} : () -> ()

"test.op"() {s = #foo.bar<a, "b", 1>} : () -> ()
// CHECK-NEXT:   "test.op"() {s = #foo.bar<a, "b", 1>} : () -> ()

"test.op"() {s = #foo.bar<"}>">} : () -> ()
// CHECK-NEXT:   "test.op"() {s = #foo.bar<"}>">} : () -> ()

"test.op"() {s = #foo.bar<"a\0Ab", "\00">} : () -> ()
// CHECK-NEXT:   "test.op"() {s = #foo.bar<"a\0Ab", "\00">} : () -> ()

// The same goes for types.
"test.op"() {s = !foo<"\FF">} : () -> ()
// CHECK-NEXT:   "test.op"() {s = !foo<"\FF">} : () -> ()

"test.op"() {s = !foo.bar<"\FF">} : () -> ()
// CHECK-NEXT:   "test.op"() {s = !foo.bar<"\FF">} : () -> ()

"test.op"() {s = !foo.bar<"a>b">} : () -> ()
// CHECK-NEXT:   "test.op"() {s = !foo.bar<"a>b">} : () -> ()

"test.op"() {s = !foo.bar<"\\">} : () -> ()
// CHECK-NEXT:   "test.op"() {s = !foo.bar<"\\">} : () -> ()

// CHECK-NEXT: }) : () -> ()
