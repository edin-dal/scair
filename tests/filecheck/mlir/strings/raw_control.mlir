// Raw control bytes in the source are string contents. The input is generated
// to keep them out of the repository.

// RUN: printf '%b' \
// RUN:   '"test.op"() {s = "a\x00b"} : () -> ()\n' \
// RUN:   '"test.op"() {s = "a\x01b"} : () -> ()\n' \
// RUN:   '"test.op"() {s = "a\x02b"} : () -> ()\n' \
// RUN:   '"test.op"() {s = "a\x03b"} : () -> ()\n' \
// RUN:   '"test.op"() {s = "a\x04b"} : () -> ()\n' \
// RUN:   '"test.op"() {s = "a\x05b"} : () -> ()\n' \
// RUN:   '"test.op"() {s = "a\x06b"} : () -> ()\n' \
// RUN:   '"test.op"() {s = "a\x07b"} : () -> ()\n' \
// RUN:   '"test.op"() {s = "a\x08b"} : () -> ()\n' \
// RUN:   '"test.op"() {s = "a\x09b"} : () -> ()\n' \
// RUN:   '"test.op"() {s = "a\x0Db"} : () -> ()\n' \
// RUN:   '"test.op"() {s = "a\x0Eb"} : () -> ()\n' \
// RUN:   '"test.op"() {s = "a\x0Fb"} : () -> ()\n' \
// RUN:   '"test.op"() {s = "a\x10b"} : () -> ()\n' \
// RUN:   '"test.op"() {s = "a\x11b"} : () -> ()\n' \
// RUN:   '"test.op"() {s = "a\x12b"} : () -> ()\n' \
// RUN:   '"test.op"() {s = "a\x13b"} : () -> ()\n' \
// RUN:   '"test.op"() {s = "a\x14b"} : () -> ()\n' \
// RUN:   '"test.op"() {s = "a\x15b"} : () -> ()\n' \
// RUN:   '"test.op"() {s = "a\x16b"} : () -> ()\n' \
// RUN:   '"test.op"() {s = "a\x17b"} : () -> ()\n' \
// RUN:   '"test.op"() {s = "a\x18b"} : () -> ()\n' \
// RUN:   '"test.op"() {s = "a\x19b"} : () -> ()\n' \
// RUN:   '"test.op"() {s = "a\x1Ab"} : () -> ()\n' \
// RUN:   '"test.op"() {s = "a\x1Bb"} : () -> ()\n' \
// RUN:   '"test.op"() {s = "a\x1Cb"} : () -> ()\n' \
// RUN:   '"test.op"() {s = "a\x1Db"} : () -> ()\n' \
// RUN:   '"test.op"() {s = "a\x1Eb"} : () -> ()\n' \
// RUN:   '"test.op"() {s = "a\x1Fb"} : () -> ()\n' \
// RUN:   '"test.op"() {s = "a\x7Fb"} : () -> ()\n' \
// RUN:   > %t
// RUN: mlir-opt %t --allow-unregistered-dialect --mlir-print-op-generic | filecheck %s --strict-whitespace
// RUN: scair-opt %t --allow-unregistered-dialect --print-generic | filecheck %s --strict-whitespace

// CHECK: "builtin.module"() ({

// CHECK-NEXT:   "test.op"() {s = "a\00b"} : () -> ()
// CHECK-NEXT:   "test.op"() {s = "a\01b"} : () -> ()
// CHECK-NEXT:   "test.op"() {s = "a\02b"} : () -> ()
// CHECK-NEXT:   "test.op"() {s = "a\03b"} : () -> ()
// CHECK-NEXT:   "test.op"() {s = "a\04b"} : () -> ()
// CHECK-NEXT:   "test.op"() {s = "a\05b"} : () -> ()
// CHECK-NEXT:   "test.op"() {s = "a\06b"} : () -> ()
// CHECK-NEXT:   "test.op"() {s = "a\07b"} : () -> ()
// CHECK-NEXT:   "test.op"() {s = "a\08b"} : () -> ()
// CHECK-NEXT:   "test.op"() {s = "a\09b"} : () -> ()
// CHECK-NEXT:   "test.op"() {s = "a\0Db"} : () -> ()
// CHECK-NEXT:   "test.op"() {s = "a\0Eb"} : () -> ()
// CHECK-NEXT:   "test.op"() {s = "a\0Fb"} : () -> ()
// CHECK-NEXT:   "test.op"() {s = "a\10b"} : () -> ()
// CHECK-NEXT:   "test.op"() {s = "a\11b"} : () -> ()
// CHECK-NEXT:   "test.op"() {s = "a\12b"} : () -> ()
// CHECK-NEXT:   "test.op"() {s = "a\13b"} : () -> ()
// CHECK-NEXT:   "test.op"() {s = "a\14b"} : () -> ()
// CHECK-NEXT:   "test.op"() {s = "a\15b"} : () -> ()
// CHECK-NEXT:   "test.op"() {s = "a\16b"} : () -> ()
// CHECK-NEXT:   "test.op"() {s = "a\17b"} : () -> ()
// CHECK-NEXT:   "test.op"() {s = "a\18b"} : () -> ()
// CHECK-NEXT:   "test.op"() {s = "a\19b"} : () -> ()
// CHECK-NEXT:   "test.op"() {s = "a\1Ab"} : () -> ()
// CHECK-NEXT:   "test.op"() {s = "a\1Bb"} : () -> ()
// CHECK-NEXT:   "test.op"() {s = "a\1Cb"} : () -> ()
// CHECK-NEXT:   "test.op"() {s = "a\1Db"} : () -> ()
// CHECK-NEXT:   "test.op"() {s = "a\1Eb"} : () -> ()
// CHECK-NEXT:   "test.op"() {s = "a\1Fb"} : () -> ()
// CHECK-NEXT:   "test.op"() {s = "a\7Fb"} : () -> ()
// CHECK-NEXT: }) : () -> ()
