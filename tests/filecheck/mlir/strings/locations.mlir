// Filenames in locations are strings.

// RUN: mlir-opt %s --allow-unregistered-dialect --mlir-print-op-generic --mlir-print-debuginfo --mlir-print-local-scope | filecheck %s --strict-whitespace
// RUN: scair-opt %s --allow-unregistered-dialect --print-generic --parse-locations --print-locations | filecheck %s --strict-whitespace

// CHECK: "builtin.module"() ({

// Filenames are escaped exactly as string values.
"test.op"() : () -> () loc("a.mlir":1:2)
// CHECK-NEXT:   "test.op"() : () -> () loc("a.mlir":1:2)

"test.op"() : () -> () loc("a b.mlir":1:2)
// CHECK-NEXT:   "test.op"() : () -> () loc("a b.mlir":1:2)

"test.op"() : () -> () loc("a\"b.mlir":1:2)
// CHECK-NEXT:   "test.op"() : () -> () loc("a\22b.mlir":1:2)

"test.op"() : () -> () loc("a\22b.mlir":1:2)
// CHECK-NEXT:   "test.op"() : () -> () loc("a\22b.mlir":1:2)

"test.op"() : () -> () loc("a\\b.mlir":1:2)
// CHECK-NEXT:   "test.op"() : () -> () loc("a\\b.mlir":1:2)

"test.op"() : () -> () loc("a\nb.mlir":1:2)
// CHECK-NEXT:   "test.op"() : () -> () loc("a\0Ab.mlir":1:2)

"test.op"() : () -> () loc("a\tb.mlir":1:2)
// CHECK-NEXT:   "test.op"() : () -> () loc("a\09b.mlir":1:2)

"test.op"() : () -> () loc("\00.mlir":1:2)
// CHECK-NEXT:   "test.op"() : () -> () loc("\00.mlir":1:2)

"test.op"() : () -> () loc("\FF.mlir":1:2)
// CHECK-NEXT:   "test.op"() : () -> () loc("\FF.mlir":1:2)

"test.op"() : () -> () loc("\C3\A9.mlir":1:2)
// CHECK-NEXT:   "test.op"() : () -> () loc("\C3\A9.mlir":1:2)

"test.op"() : () -> () loc("é.mlir":1:2)
// CHECK-NEXT:   "test.op"() : () -> () loc("\C3\A9.mlir":1:2)

// Line and column ranges.
"test.op"() : () -> () loc("\FF.mlir":1:2 to :3)
// CHECK-NEXT:   "test.op"() : () -> () loc("\FF.mlir":1:2 to :3)

"test.op"() : () -> () loc("\FF.mlir":1:2 to 3:4)
// CHECK-NEXT:   "test.op"() : () -> () loc("\FF.mlir":1:2 to 3:4)

"test.op"() : () -> () loc("a\nb.mlir":1:2 to 3:4)
// CHECK-NEXT:   "test.op"() : () -> () loc("a\0Ab.mlir":1:2 to 3:4)
