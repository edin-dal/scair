// RUN: scair-opt %s --verify-diagnostics --split-input-file | filecheck %s

// A number of operands that does not split evenly over the variadic definitions.

%x, %y, %z, %w = "test.op"() : () -> (i32, i32, i32, i32)
"test.same_variadic_operand"(%x, %y, %z, %w) : (i32, i32, i32, i32) -> ()

// CHECK:         "test.same_variadic_operand"(%0, %1, %2, %3) : (i32, i32, i32, i32) -> ()
// CHECK-NEXT:    ^^
// CHECK-NEXT:    > Expected 1 operands plus a multiple of 2 same-sized variadic ones, got 4.

// -----

// Fewer operands than the op has non-variadic definitions.

"test.same_variadic_operand"() : () -> ()

// CHECK:         "test.same_variadic_operand"() : () -> ()
// CHECK-NEXT:    ^^
// CHECK-NEXT:    > Expected 1 operands plus a multiple of 2 same-sized variadic ones, got 0.

// -----

// Idem for results.

%a, %b = "test.same_variadic_result"() : () -> (i32, i32)

// CHECK:         %0, %1 = "test.same_variadic_result"() : () -> (i32, i32)
// CHECK-NEXT:    ^^
// CHECK-NEXT:    > Expected 1 results plus a multiple of 2 same-sized variadic ones, got 2.
