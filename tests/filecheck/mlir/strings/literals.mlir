// Contents that need no escaping survive unchanged, spaces included.

// RUN: mlir-opt %s --allow-unregistered-dialect --mlir-print-op-generic | filecheck %s --strict-whitespace
// RUN: scair-opt %s --allow-unregistered-dialect --print-generic | filecheck %s --strict-whitespace
// RUN: scair-opt %s --allow-unregistered-dialect --print-generic -p 'mlir-opt[]' | filecheck %s --strict-whitespace

// CHECK: "builtin.module"() ({

// Empty and whitespace-only strings.
"test.op"() {s = ""} : () -> ()
// CHECK-NEXT:   "test.op"() {s = ""} : () -> ()

"test.op"() {s = " "} : () -> ()
// CHECK-NEXT:   "test.op"() {s = " "} : () -> ()

"test.op"() {s = "  "} : () -> ()
// CHECK-NEXT:   "test.op"() {s = "  "} : () -> ()

"test.op"() {s = "   a   "} : () -> ()
// CHECK-NEXT:   "test.op"() {s = "   a   "} : () -> ()

"test.op"() {s = "  two  spaces  "} : () -> ()
// CHECK-NEXT:   "test.op"() {s = "  two  spaces  "} : () -> ()

// Contents that look like MLIR syntax.
"test.op"() {s = "// comment"} : () -> ()
// CHECK-NEXT:   "test.op"() {s = "// comment"} : () -> ()

"test.op"() {s = "/* comment */"} : () -> ()
// CHECK-NEXT:   "test.op"() {s = "/* comment */"} : () -> ()

"test.op"() {s = "// -----"} : () -> ()
// CHECK-NEXT:   "test.op"() {s = "// -----"} : () -> ()

"test.op"() {s = "%0 @f #a ^b !t"} : () -> ()
// CHECK-NEXT:   "test.op"() {s = "%0 @f #a ^b !t"} : () -> ()

"test.op"() {s = "() -> ()"} : () -> ()
// CHECK-NEXT:   "test.op"() {s = "() -> ()"} : () -> ()

"test.op"() {s = "{a = 1}"} : () -> ()
// CHECK-NEXT:   "test.op"() {s = "{a = 1}"} : () -> ()

"test.op"() {s = "<{a = 1}>"} : () -> ()
// CHECK-NEXT:   "test.op"() {s = "<{a = 1}>"} : () -> ()

"test.op"() {s = "[1, 2]"} : () -> ()
// CHECK-NEXT:   "test.op"() {s = "[1, 2]"} : () -> ()

"test.op"() {s = "loc(unknown)"} : () -> ()
// CHECK-NEXT:   "test.op"() {s = "loc(unknown)"} : () -> ()

"test.op"() {s = ": i32"} : () -> ()
// CHECK-NEXT:   "test.op"() {s = ": i32"} : () -> ()

"test.op"() {s = "{{x}} [[y]]"} : () -> ()
// CHECK-NEXT:   "test.op"() {s = "{{[{][{]}}x}} {{\[\[}}y]]"} : () -> ()

"test.op"() {s = "'"} : () -> ()
// CHECK-NEXT:   "test.op"() {s = "'"} : () -> ()

"test.op"() {s = "`"} : () -> ()
// CHECK-NEXT:   "test.op"() {s = "`"} : () -> ()

"test.op"() {s = "$"} : () -> ()
// CHECK-NEXT:   "test.op"() {s = "$"} : () -> ()

"test.op"() {s = "0x41"} : () -> ()
// CHECK-NEXT:   "test.op"() {s = "0x41"} : () -> ()

"test.op"() {s = "\\n"} : () -> ()
// CHECK-NEXT:   "test.op"() {s = "\\n"} : () -> ()

"test.op"() {s = "\\00"} : () -> ()
// CHECK-NEXT:   "test.op"() {s = "\\00"} : () -> ()

// CHECK-NEXT: }) : () -> ()
