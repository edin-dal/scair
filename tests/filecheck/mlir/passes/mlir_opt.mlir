// Shorthand to use mlir-opt within a ScaIR pipeline
// RUN: scair-opt %s -p 'mlir-opt[cse]' | filecheck %s

// The explicit syntax, allowing full control over the arguments
// RUN: scair-opt %s -p 'mlir-opt{arguments=--cse,--mlir-print-op-generic}' | filecheck %s

// mlir-opt can be combined with ScaIR passes; here its output is further canonicalized
// RUN: scair-opt %s -p 'Canonicalize,mlir-opt[cse],Canonicalize' | filecheck %s --check-prefix=COMBO

%0 = "arith.constant"() <{value = 1 : i32}> : () -> i32
%1 = "arith.constant"() <{value = 1 : i32}> : () -> i32
%2 = "arith.addi"(%0, %1) : (i32, i32) -> i32
"test.op"(%2) : (i32) -> ()

// CHECK:      builtin.module {
// CHECK-NEXT:   %0 = "arith.constant"() <{value = 1 : i32}> : () -> i32
// CHECK-NEXT:   %1 = "arith.addi"(%0, %0) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
// CHECK-NEXT:   "test.op"(%1) : (i32) -> ()
// CHECK-NEXT: }

// COMBO:      builtin.module {
// COMBO-NEXT:   %0 = "arith.constant"() <{value = 2 : i32}> : () -> i32
// COMBO-NEXT:   "test.op"(%0) : (i32) -> ()
// COMBO-NEXT: }
