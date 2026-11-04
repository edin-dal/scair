// Attributes that hold strings do not change how those strings are printed.

// RUN: mlir-opt %s --allow-unregistered-dialect --mlir-print-op-generic | filecheck %s --strict-whitespace
// RUN: scair-opt %s --allow-unregistered-dialect --print-generic | filecheck %s --strict-whitespace
// RUN: scair-opt %s --allow-unregistered-dialect --print-generic -p 'mlir-opt{allowUnregisteredDialect=true arguments=--allow-unregistered-dialect,--mlir-print-op-generic}' | filecheck %s --strict-whitespace

// CHECK: "builtin.module"() ({

#byte = "\FF"
#array = ["\00", "\n"]

// Arrays.
"test.op"() {s = ["\FF"]} : () -> ()
// CHECK-NEXT:   "test.op"() {s = ["\FF"]} : () -> ()

"test.op"() {s = ["\FF", "a\nb", ["\00"]]} : () -> ()
// CHECK-NEXT:   "test.op"() {s = ["\FF", "a\0Ab", ["\00"]]} : () -> ()

"test.op"() {s = [["\22"], [["\5C"]]]} : () -> ()
// CHECK-NEXT:   "test.op"() {s = {{\[\[}}"\22"], {{\[\[}}"\\"]]]} : () -> ()

// Properties, on registered and unregistered operations.
"test.op"() <{s = "\FF"}> : () -> ()
// CHECK-NEXT:   "test.op"() <{s = "\FF"}> : () -> ()

"foo.op"() <{s = "\n"}> : () -> ()
// CHECK-NEXT:   "foo.op"() <{s = "\0A"}> : () -> ()

"test.op"() <{s = ["\22"]}> : () -> ()
// CHECK-NEXT:   "test.op"() <{s = ["\22"]}> : () -> ()

// Dictionaries. ScaIR prints two spaces before a nested dictionary, which is unrelated.
"test.op"() {s = {k = "\FF"}} : () -> ()
// CHECK-NEXT:   "test.op"() {s ={{ +}}{k = "\FF"}} : () -> ()

"test.op"() {s = {k = "\FF", n = {m = "\22"}}} : () -> ()
// CHECK-NEXT:   "test.op"() {s ={{ +}}{k = "\FF", n ={{ +}}{m = "\22"}}} : () -> ()

"test.op"() {s = [{a = "\n"}]} : () -> ()
// CHECK-NEXT:   "test.op"() {s = [{{ *}}{a = "\0A"}]} : () -> ()

// Attribute aliases.
"test.op"() {s = #byte} : () -> ()
// CHECK-NEXT:   "test.op"() {s = "\FF"} : () -> ()

"test.op"() {s = #array} : () -> ()
// CHECK-NEXT:   "test.op"() {s = ["\00", "\0A"]} : () -> ()

"test.op"() {s = [#byte, #byte]} : () -> ()
// CHECK-NEXT:   "test.op"() {s = ["\FF", "\FF"]} : () -> ()

// Regions.
"test.op"() ({
  "test.op"() {s = "\FF"} : () -> ()
}) : () -> ()
// CHECK-NEXT:   "test.op"() ({
// CHECK-NEXT:     "test.op"() {s = "\FF"} : () -> ()
// CHECK-NEXT:   }) : () -> ()

// Attributes of the LLVM dialect that hold strings.
"test.op"() {s = #llvm.target_features<["+a\22b", "-\FF"]>} : () -> ()
// CHECK-NEXT:   "test.op"() {s = #llvm.target_features<["+a\22b", "-\FF"]>} : () -> ()

// CHECK-NEXT: }) : () -> ()
