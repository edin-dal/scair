// RUN: scair-opt %s | filecheck %s

"op1"() ({
^bb0():
  %0 = "test.op"() : () -> (!tuples.tuple<f32, f32>)
  %1 = "test.op"() : () -> (!tuples.tuplestream<f32, f32>)
  %2 = "test.op"() : () -> (#tuples.column_def<f32, f32>)
  %3 = "test.op"() : () -> (#tuples.column_ref<f32>)
  %4 = "test.op"() : () -> (i32)
  tuples.return %4 : i32
  %5 = tuples.getcol %0 @example::@one : i32

// CHECK: "op1"() ({
// CHECK: ^bb0():
// CHECK:   %0 = "test.op"() : () -> (!tuples.tuple<f32, f32>)
// CHECK:   %1 = "test.op"() : () -> (!tuples.tuplestream<f32, f32>)
// CHECK:   %2 = "test.op"() : () -> (#tuples.column_def<f32, f32>)
// CHECK:   %3 = "test.op"() : () -> (#tuples.column_ref<f32>)
// CHECK:   "tuples.return"(%4) : (i32) -> ()
// CHECK:   %5 = "tuples.getcol"(%0) {attr = @example::@one} : (!tuples.tuple<f32, f32>) -> (i32)

}) : () -> ()
