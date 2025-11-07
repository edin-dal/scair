// RUN: scair-opt %s | filecheck %s

builtin.module {
    %0, %1, %2, %3 = "test.op"() : () -> (i1, i8, i16, i32)
    %4 = "builtin.unrealized_conversion_cast"() : () -> i32
    %5 = "builtin.unrealized_conversion_cast"(%0) : (i1) -> i32
    %6, %7 = "builtin.unrealized_conversion_cast"(%1) : (i8) -> (i32, i32)
    %8, %9 = "builtin.unrealized_conversion_cast"(%2, %3) : (i16, i32) -> (i32, i32)
}

// CHECK:       builtin.module {
// CHECK-NEXT:    %0, %1, %2, %3 = "test.op"() : () -> (i1, i8, i16, i32)
// CHECK-NEXT:    %4 = "builtin.unrealized_conversion_cast"() : () -> i32
// CHECK-NEXT:    %5 = "builtin.unrealized_conversion_cast"(%0) : (i1) -> i32
// CHECK-NEXT:    %6, %7 = "builtin.unrealized_conversion_cast"(%1) : (i8) -> (i32, i32)
// CHECK-NEXT:    %8, %9 = "builtin.unrealized_conversion_cast"(%2, %3) : (i16, i32) -> (i32, i32)
// CHECK-NEXT:  }
