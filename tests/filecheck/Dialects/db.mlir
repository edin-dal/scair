// RUN: scair-opt %s | filecheck %s

"op1"() ({
^bb0():
  %0 = "test.op"() : () -> (i32)
  %1 = db.constant(1 : i32) : !db.decimal<12, 2>
  %2 = db.constant(1 : i32) : !db.decimal<12, 2>
  %3 = db.sub %1 : !db.decimal<12, 2>, %2 : !db.decimal<12, 2>
  %4 = db.mul %1 : !db.decimal<12, 2>, %2 : !db.decimal<12, 2>
  %5 = db.add %1 : !db.decimal<12, 2>, %2 : !db.decimal<12, 2>
  %6 = db.compare lte %3 : i64, %5 : i64
  %7 = "test.op"() : () -> (!db.char<1>)
  %8 = "test.op"() : () -> (!db.date<day>)
  %9 = "test.op"() : () -> (!db.string<>)

// CHECK: "op1"() ({
// CHECK: ^bb0():
// CHECK:   %0 = "test.op"() : () -> (i32)
// CHECK:   %1 = "db.constant"() {value = 1 : i32} : () -> (!db.decimal<12, 2>)
// CHECK:   %2 = "db.constant"() {value = 1 : i32} : () -> (!db.decimal<12, 2>)
// CHECK:   %3 = "db.sub"(%1, %2) : (!db.decimal<12, 2>, !db.decimal<12, 2>) -> (i64)
// CHECK:   %4 = "db.mul"(%1, %2) : (!db.decimal<12, 2>, !db.decimal<12, 2>) -> (i64)
// CHECK:   %5 = "db.add"(%1, %2) : (!db.decimal<12, 2>, !db.decimal<12, 2>) -> (i64)
// CHECK:   %6 = "db.compare"(%3, %5) {predicate = lte} : (i64, i64) -> (i8)
// CHECK:   %7 = "test.op"() : () -> (!db.char<1>)
// CHECK:   %8 = "test.op"() : () -> (!db.date<day>)
// CHECK:   %9 = "test.op"() : () -> (!db.string<>)
// CHECK: }) : () -> ()

}) : () -> ()
