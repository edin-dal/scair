// RUN: scair-opt %s --print-locations --parse-locations | filecheck %s -DSOURCE_FILE=%s
// RUN: scair-opt %s --print-locations | filecheck %s --check-prefix=NO-FLAG

builtin.module {
  %0 = "test.op"() : () -> i32
  "test.op"() : () -> () loc("original.mlir":12:7)
  "test.op"() : () -> () loc("original.mlir":12:7 to :18)
  "test.op"() : () -> () loc("original.mlir":12:7 to 14:3)
  "test.op"() : () -> () loc(unknown)
}

// CHECK:      builtin.module {
// CHECK-NEXT:   %0 = "test.op"() : () -> i32 loc("[[SOURCE_FILE]]":5:8)
// CHECK-NEXT:   "test.op"() : () -> () loc("original.mlir":12:7)
// CHECK-NEXT:   "test.op"() : () -> () loc("original.mlir":12:7 to :18)
// CHECK-NEXT:   "test.op"() : () -> () loc("original.mlir":12:7 to 14:3)
// CHECK-NEXT:   "test.op"() : () -> () loc(unknown)
// CHECK-NEXT: } loc("[[SOURCE_FILE]]":4:1)

// NO-FLAG:       builtin.module {
// NO-FLAG-NEXT:    %0 = "test.op"() : () -> i32 loc(unknown)
// NO-FLAG-NEXT:    "test.op"() : () -> () loc("original.mlir":12:7)
// NO-FLAG-NEXT:    "test.op"() : () -> () loc("original.mlir":12:7 to :18)
// NO-FLAG-NEXT:    "test.op"() : () -> () loc("original.mlir":12:7 to 14:3)
// NO-FLAG-NEXT:    "test.op"() : () -> () loc(unknown)
// NO-FLAG-NEXT:  } loc(unknown)
