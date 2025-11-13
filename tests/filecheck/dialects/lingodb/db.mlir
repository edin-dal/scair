// RUN: scair-opt %s -s | filecheck %s

  %0 = "test.op"() : () -> i32
  %1 = "db.constant"() {value = 1 : i32} : () -> !db.decimal<12, 2>
  %2 = "db.constant"() {value = 1 : i32} : () -> !db.decimal<12, 2>
  %3 = "db.sub"(%1, %2) : (!db.decimal<12, 2>, !db.decimal<12, 2>) -> !db.decimal<12, 2>
  %4 = "db.mul"(%1, %2) : (!db.decimal<12, 2>, !db.decimal<12, 2>) -> !db.decimal<24, 4>
  %5 = "db.add"(%1, %2) : (!db.decimal<12, 2>, !db.decimal<12, 2>) -> !db.decimal<12, 2>
  %6 = "db.compare"(%3, %5) {predicate = lte} : (!db.decimal<12, 2>, !db.decimal<12, 2>) -> i1
  %7 = "test.op"() : () -> !db.char<1>
  %8 = "test.op"() : () -> !db.date<day>
  %9 = "test.op"() : () -> !db.string

// CHECK: builtin.module {
// CHECK-NEXT:  %0 = "test.op"() : () -> i32
// CHECK-NEXT:  %1 = "db.constant"() {value = 1 : i32} : () -> !db.decimal<12, 2>
// CHECK-NEXT:  %2 = "db.constant"() {value = 1 : i32} : () -> !db.decimal<12, 2>
// CHECK-NEXT:  %3 = "db.sub"(%1, %2) : (!db.decimal<12, 2>, !db.decimal<12, 2>) -> !db.decimal<12, 2>
// CHECK-NEXT:  %4 = "db.mul"(%1, %2) : (!db.decimal<12, 2>, !db.decimal<12, 2>) -> !db.decimal<24, 4>
// CHECK-NEXT:  %5 = "db.add"(%1, %2) : (!db.decimal<12, 2>, !db.decimal<12, 2>) -> !db.decimal<12, 2>
// CHECK-NEXT:  %6 = "db.compare"(%3, %5) {predicate = lte} : (!db.decimal<12, 2>, !db.decimal<12, 2>) -> i1
// CHECK-NEXT:  %7 = "test.op"() : () -> !db.char<1>
// CHECK-NEXT:  %8 = "test.op"() : () -> !db.date<day>
// CHECK-NEXT:  %9 = "test.op"() : () -> !db.string
// CHECK-NEXT:   }

