// RUN: scair-opt %s | filecheck %s

%0, %1, %2, %3 = "test.op"() : () -> (i1, i32, i32, i32)

"scf.condition"(%0, %1, %2) : (i1, i32, i32) -> ()

%4 = "scf.execute_region"() ({}) : () -> (i32)

%5, %6, %7, %8 = "test.op"() : () -> (i32, i32, i32, i32)

%9, %10 = "scf.for"(%5, %6, %7, %8) ({}) : (i32, i32, i32, i32) -> (i1, i1)

%11, %12, %13 = "test.op"() : () -> (index, index, index)
%15, %16, %17 = "test.op"() : () -> (array<i1:0, 1>, array<i1:0, 1>, array<i1:0, 1>)
%18, %19 = "test.op"() : () -> (tensor<1xf32>, tensor<1xf32>)

%20, %21 = "scf.forall"(%11, %12, %13, %15, %16, %17, %18, %19) ({}) : (index, index, index, array<i1:0, 1>, array<i1:0, 1>, array<i1:0, 1>, tensor<1xf32>, tensor<1xf32>) -> (i1, i1)

"scf.forall.in_parallel"() ({}) : () -> ()

%22, %23 = "scf.if"(%20) ({}, {}) : (i1) -> (i1, i1)

%24, %25 = "scf.parallel"(%11, %12, %13, %22, %23) ({}) : (index, index, index, i1, i1) -> (i1, i1)

"scf.reduce"(%24, %25) ({
    %26 = "scf.reduce.return"() : () -> (i1)
}) : (i1, i1) -> ()

%27, %28 = "scf.while"(%24) ({}, {}) : (i1) -> (i1, i1)

%29, %30 = "scf.index_switch"(%13, %15) ({}, {}) : (index, array<i1:0, 1>) -> (i1, i1)

%31 = "scf.yield"() : () -> (i1)


// CHECK:       builtin.module {
// CHECK-NEXT:    %0, %1, %2, %3 = "test.op"() : () -> (i1, i32, i32, i32)
// CHECK-NEXT:    "scf.condition"(%0, %1, %2) : (i1, i32, i32) -> ()
// CHECK-NEXT:    %4 = "scf.execute_region"() ({

// CHECK:       }) : () -> (i32)
// CHECK-NEXT:    %5, %6, %7, %8 = "test.op"() : () -> (i32, i32, i32, i32)
// CHECK-NEXT:    %9, %10 = "scf.for"(%5, %6, %7, %8) ({

// CHECK:       }) : (i32, i32, i32, i32) -> (i1, i1)
// CHECK-NEXT:    %11, %12, %13 = "test.op"() : () -> (index, index, index)
// CHECK-NEXT:    %14, %15, %16 = "test.op"() : () -> (array<i1: 0, 1>, array<i1: 0, 1>, array<i1: 0, 1>)
// CHECK-NEXT:    %17, %18 = "test.op"() : () -> (tensor<1xf32>, tensor<1xf32>)
// CHECK-NEXT:    %19, %20 = "scf.forall"(%11, %12, %13, %14, %15, %16, %17, %18) ({

// CHECK:       }) : (index, index, index, array<i1: 0, 1>, array<i1: 0, 1>, array<i1: 0, 1>, tensor<1xf32>, tensor<1xf32>) -> (i1, i1)
// CHECK-NEXT:    "scf.forall.in_parallel"() ({

// CHECK:       }) : () -> ()
// CHECK-NEXT:    %21, %22 = "scf.if"(%19) ({

// CHECK:       }, {

// CHECK:       }) : (i1) -> (i1, i1)
// CHECK-NEXT:    %23, %24 = "scf.parallel"(%11, %12, %13, %21, %22) ({

// CHECK:       }) : (index, index, index, i1, i1) -> (i1, i1)
// CHECK-NEXT:    "scf.reduce"(%23, %24) ({
// CHECK-NEXT:    %25 = "scf.reduce.return"() : () -> (i1)
// CHECK-NEXT:  }) : (i1, i1) -> ()
// CHECK-NEXT:    %25, %26 = "scf.while"(%23) ({

// CHECK:       }, {

// CHECK:       }) : (i1) -> (i1, i1)
// CHECK-NEXT:    %27, %28 = "scf.index_switch"(%13, %14) ({

// CHECK:       }, {

// CHECK:       }) : (index, array<i1: 0, 1>) -> (i1, i1)
// CHECK-NEXT:    %29 = "scf.yield"() : () -> (i1)
// CHECK-NEXT:  }