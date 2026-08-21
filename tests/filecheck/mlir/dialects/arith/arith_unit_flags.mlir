// RUN: scair-opt %s | mlir-opt --mlir-print-op-generic | scair-opt | filecheck %s

// The `isExact` unit flag round-trips through real mlir-opt.

%lhs, %rhs = "test.op"() : () -> (i32, i32)

%divsi = "arith.divsi"(%lhs, %rhs) <{"isExact"}> : (i32, i32) -> i32
%divui = "arith.divui"(%lhs, %rhs) <{"isExact"}> : (i32, i32) -> i32
%shrsi = "arith.shrsi"(%lhs, %rhs) <{"isExact"}> : (i32, i32) -> i32
%shrui = "arith.shrui"(%lhs, %rhs) <{"isExact"}> : (i32, i32) -> i32

// A flagless op keeps printing without the flag.
%plain = "arith.divsi"(%lhs, %rhs) : (i32, i32) -> i32

// CHECK:       builtin.module {
// CHECK-NEXT:    %0, %1 = "test.op"() : () -> (i32, i32)
// CHECK-NEXT:    %2 = "arith.divsi"(%0, %1) <{isExact}> : (i32, i32) -> i32
// CHECK-NEXT:    %3 = "arith.divui"(%0, %1) <{isExact}> : (i32, i32) -> i32
// CHECK-NEXT:    %4 = "arith.shrsi"(%0, %1) <{isExact}> : (i32, i32) -> i32
// CHECK-NEXT:    %5 = "arith.shrui"(%0, %1) <{isExact}> : (i32, i32) -> i32
// CHECK-NEXT:    %6 = "arith.divsi"(%0, %1) : (i32, i32) -> i32
// CHECK-NEXT:  }
