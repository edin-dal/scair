// RUN: scair-opt %s -s | filecheck %s

  %0 = "test.op"() : () -> (i32)
  %1 = db.constant(1 : i32) : !db.decimal<12, 2>
  %2 = db.constant(1 : i32) : !db.decimal<12, 2>
  %3 = db.sub %1 : !db.decimal<12, 2>, %2 : !db.decimal<12, 2>
  %4 = db.mul %1 : !db.decimal<12, 2>, %2 : !db.decimal<12, 2>
  %5 = db.add %1 : !db.decimal<12, 2>, %2 : !db.decimal<12, 2>
  %6 = db.compare lte %3 : !db.decimal<12, 2>, %5 : !db.decimal<12, 2>
  %7 = "test.op"() : () -> (!db.char<1>)
  %8 = "test.op"() : () -> (!db.date<day>)
  %9 = "test.op"() : () -> (!db.string)

// CHECK: builtin.module {
// CHECK:     %0 = "test.op"() : () -> (i32)
// CHECK:     %1 = "db.constant"() {value = 1 : i32} : () -> (!db.decimal<12, 2>)
// CHECK:     %2 = "db.constant"() {value = 1 : i32} : () -> (!db.decimal<12, 2>)
// CHECK:     %3 = "db.sub"(%1, %2) : (!db.decimal<12, 2>, !db.decimal<12, 2>) -> (!db.decimal<12, 2>)
// CHECK:     %4 = "db.mul"(%1, %2) : (!db.decimal<12, 2>, !db.decimal<12, 2>) -> (!db.decimal<24, 4>)
// CHECK:     %5 = "db.add"(%1, %2) : (!db.decimal<12, 2>, !db.decimal<12, 2>) -> (!db.decimal<12, 2>)
// CHECK:     %6 = "db.compare"(%3, %5) {predicate = lte} : (!db.decimal<12, 2>, !db.decimal<12, 2>) -> (i1)
// CHECK:     %7 = "test.op"() : () -> (!db.char<1>)
// CHECK:     %8 = "test.op"() : () -> (!db.date<day>)
// CHECK:     %9 = "test.op"() : () -> (!db.string)
// CHECK:   }

