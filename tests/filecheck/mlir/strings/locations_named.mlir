// Names in locations are strings.

// RUN: mlir-opt %s --allow-unregistered-dialect --mlir-print-op-generic --mlir-print-debuginfo --mlir-print-local-scope | filecheck %s --strict-whitespace
// RUN: scair-opt %s --allow-unregistered-dialect --print-generic --parse-locations --print-locations | filecheck %s --strict-whitespace

// CHECK: "builtin.module"() ({

// Named locations.
"test.op"() : () -> () loc("a")
// CHECK-NEXT:   "test.op"() : () -> () loc("a")

"test.op"() : () -> () loc("\FF")
// CHECK-NEXT:   "test.op"() : () -> () loc("\FF")

"test.op"() : () -> () loc("a\22b"("f.c":1:2))
// CHECK-NEXT:   "test.op"() : () -> () loc("a\22b"("f.c":1:2))

// Fused locations.
"test.op"() : () -> () loc(fused["\FF", "\n"])
// CHECK-NEXT:   "test.op"() : () -> () loc(fused["\FF", "\0A"])
