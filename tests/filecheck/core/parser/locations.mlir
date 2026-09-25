// RUN: scair-opt %s --print-locations | filecheck %s -DSOURCE_FILE=%s

builtin.module {
  %0 = "test.op"() : () -> i32
  "test.op"() : () -> () loc("original.mlir":12:7)
  "test.op"() : () -> () loc("original.mlir":12:7 to :18)
  "test.op"() : () -> () loc("original.mlir":12:7 to 14:3)
  "test.op"() : () -> () loc(unknown)
}

// CHECK:      builtin.module {
// CHECK-NEXT:   %0 = "test.op"() : () -> i32 loc("[[SOURCE_FILE]]":4:8)
// CHECK-NEXT:   "test.op"() : () -> () loc("original.mlir":12:7)
// CHECK-NEXT:   "test.op"() : () -> () loc("original.mlir":12:7 to :18)
// CHECK-NEXT:   "test.op"() : () -> () loc("original.mlir":12:7 to 14:3)
// CHECK-NEXT:   "test.op"() : () -> () loc(unknown)
// CHECK-NEXT: } loc("[[SOURCE_FILE]]":3:1)
