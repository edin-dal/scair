// Raw control bytes in the source are string contents. The RUN line decodes the
// hexadecimal escapes below, to keep those bytes out of the repository.

// RUN: printf '%b' "$(grep -v '^// CHECK' %s)" > %t
// RUN: mlir-opt %t --allow-unregistered-dialect --mlir-print-op-generic | filecheck %s --strict-whitespace
// RUN: scair-opt %t --allow-unregistered-dialect --print-generic | filecheck %s --strict-whitespace
// RUN: scair-opt %t --allow-unregistered-dialect --print-generic -p 'mlir-opt[]' | filecheck %s --strict-whitespace

// CHECK: "builtin.module"() ({

"test.op"() {s = "a\x00b"} : () -> ()
// CHECK-NEXT:   "test.op"() {s = "a\00b"} : () -> ()

"test.op"() {s = "a\x01b"} : () -> ()
// CHECK-NEXT:   "test.op"() {s = "a\01b"} : () -> ()

"test.op"() {s = "a\x02b"} : () -> ()
// CHECK-NEXT:   "test.op"() {s = "a\02b"} : () -> ()

"test.op"() {s = "a\x03b"} : () -> ()
// CHECK-NEXT:   "test.op"() {s = "a\03b"} : () -> ()

"test.op"() {s = "a\x04b"} : () -> ()
// CHECK-NEXT:   "test.op"() {s = "a\04b"} : () -> ()

"test.op"() {s = "a\x05b"} : () -> ()
// CHECK-NEXT:   "test.op"() {s = "a\05b"} : () -> ()

"test.op"() {s = "a\x06b"} : () -> ()
// CHECK-NEXT:   "test.op"() {s = "a\06b"} : () -> ()

"test.op"() {s = "a\x07b"} : () -> ()
// CHECK-NEXT:   "test.op"() {s = "a\07b"} : () -> ()

"test.op"() {s = "a\x08b"} : () -> ()
// CHECK-NEXT:   "test.op"() {s = "a\08b"} : () -> ()

"test.op"() {s = "a\x09b"} : () -> ()
// CHECK-NEXT:   "test.op"() {s = "a\09b"} : () -> ()

"test.op"() {s = "a\x0Db"} : () -> ()
// CHECK-NEXT:   "test.op"() {s = "a\0Db"} : () -> ()

"test.op"() {s = "a\x0Eb"} : () -> ()
// CHECK-NEXT:   "test.op"() {s = "a\0Eb"} : () -> ()

"test.op"() {s = "a\x0Fb"} : () -> ()
// CHECK-NEXT:   "test.op"() {s = "a\0Fb"} : () -> ()

"test.op"() {s = "a\x10b"} : () -> ()
// CHECK-NEXT:   "test.op"() {s = "a\10b"} : () -> ()

"test.op"() {s = "a\x11b"} : () -> ()
// CHECK-NEXT:   "test.op"() {s = "a\11b"} : () -> ()

"test.op"() {s = "a\x12b"} : () -> ()
// CHECK-NEXT:   "test.op"() {s = "a\12b"} : () -> ()

"test.op"() {s = "a\x13b"} : () -> ()
// CHECK-NEXT:   "test.op"() {s = "a\13b"} : () -> ()

"test.op"() {s = "a\x14b"} : () -> ()
// CHECK-NEXT:   "test.op"() {s = "a\14b"} : () -> ()

"test.op"() {s = "a\x15b"} : () -> ()
// CHECK-NEXT:   "test.op"() {s = "a\15b"} : () -> ()

"test.op"() {s = "a\x16b"} : () -> ()
// CHECK-NEXT:   "test.op"() {s = "a\16b"} : () -> ()

"test.op"() {s = "a\x17b"} : () -> ()
// CHECK-NEXT:   "test.op"() {s = "a\17b"} : () -> ()

"test.op"() {s = "a\x18b"} : () -> ()
// CHECK-NEXT:   "test.op"() {s = "a\18b"} : () -> ()

"test.op"() {s = "a\x19b"} : () -> ()
// CHECK-NEXT:   "test.op"() {s = "a\19b"} : () -> ()

"test.op"() {s = "a\x1Ab"} : () -> ()
// CHECK-NEXT:   "test.op"() {s = "a\1Ab"} : () -> ()

"test.op"() {s = "a\x1Bb"} : () -> ()
// CHECK-NEXT:   "test.op"() {s = "a\1Bb"} : () -> ()

"test.op"() {s = "a\x1Cb"} : () -> ()
// CHECK-NEXT:   "test.op"() {s = "a\1Cb"} : () -> ()

"test.op"() {s = "a\x1Db"} : () -> ()
// CHECK-NEXT:   "test.op"() {s = "a\1Db"} : () -> ()

"test.op"() {s = "a\x1Eb"} : () -> ()
// CHECK-NEXT:   "test.op"() {s = "a\1Eb"} : () -> ()

"test.op"() {s = "a\x1Fb"} : () -> ()
// CHECK-NEXT:   "test.op"() {s = "a\1Fb"} : () -> ()

"test.op"() {s = "a\x7Fb"} : () -> ()
// CHECK-NEXT:   "test.op"() {s = "a\7Fb"} : () -> ()

// CHECK-NEXT: }) : () -> ()
