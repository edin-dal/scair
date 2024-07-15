// RUN: scair-opt %s | filecheck %s

"op1"() ({
^bb0():
  %0 = "test.op"() : () -> (!tuples.tuple<f32, f32>)
  %1 = "test.op"() : () -> (!tuples.tuplestream<f32, f32>)
  %2 = "test.op"() : () -> (#tuples.column_def<f32, f32>)
  %3 = "test.op"() : () -> (#tuples.column_ref<f32>)

// CHECK: "op1"() ({
// CHECK: ^bb0():
// CHECK:   %0 = "test.op"() : () -> (!tuples.tuple<f32, f32>)
// CHECK:   %1 = "test.op"() : () -> (!tuples.tuplestream<f32, f32>)
// CHECK:   %2 = "test.op"() : () -> (#tuples.column_def<f32, f32>)
// CHECK:   %3 = "test.op"() : () -> (#tuples.column_ref<f32>)

}) : () -> ()
