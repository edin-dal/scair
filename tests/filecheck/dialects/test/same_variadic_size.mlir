// RUN: scair-opt %s | filecheck %s

// Several variadic operands or results, declared to all have the same size, are
// split evenly over their definitions. No segment sizes property is involved.

%x, %y, %z = "test.op"() : () -> (i32, i32, i32)

// One operand per variadic definition.
"test.same_variadic_operand"(%x, %y, %z) : (i32, i32, i32) -> ()
// CHECK:       "test.same_variadic_operand"(%0, %1, %2) : (i32, i32, i32) -> ()

// Two each, with the single operand still picked out from between them.
"test.same_variadic_operand"(%x, %y, %z, %x, %y) : (i32, i32, i32, i32, i32) -> ()
// CHECK-NEXT:  "test.same_variadic_operand"(%0, %1, %2, %0, %1) : (i32, i32, i32, i32, i32) -> ()

// None at all, leaving only the single operand.
"test.same_variadic_operand"(%x) : (i32) -> ()
// CHECK-NEXT:  "test.same_variadic_operand"(%0) : (i32) -> ()

// Idem for results.
%a, %b, %c = "test.same_variadic_result"() : () -> (i32, i32, i32)
// CHECK-NEXT:  %3, %4, %5 = "test.same_variadic_result"() : () -> (i32, i32, i32)

%d, %e, %f, %g, %h = "test.same_variadic_result"() : () -> (i32, i32, i32, i32, i32)
// CHECK-NEXT:  %6, %7, %8, %9, %10 = "test.same_variadic_result"() : () -> (i32, i32, i32, i32, i32)
