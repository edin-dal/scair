// RUN: scair-opt %s | filecheck %s

"op1"() ({
^bb0():
  %0 = "test.op"() : () -> (!tuples.tuple<f32, f32>)
  %1 = "test.op"() : () -> (!tuples.tuplestream<f32, f32>)
  %2 = "test.op"() : () -> (i32)
  tuples.return %2 : i32
  %4 = tuples.getcol %0 @example::@one : i32

// CHECK: builtin.module {
// CHECK: ^bb0():
// CHECK:   "op1"() ({
// CHECK:   ^bb1():
// CHECK:     %0 = "test.op"() : () -> (!tuples.tuple<f32, f32>)
// CHECK:     %1 = "test.op"() : () -> (!tuples.tuplestream<f32, f32>)
// CHECK:     %2 = "test.op"() : () -> (i32)
// CHECK:     "tuples.return"(%2) : (i32) -> ()
// CHECK:     %3 = "tuples.getcol"(%0) {attr = @example::@one} : (!tuples.tuple<f32, f32>) -> (i32)

}) : () -> ()
