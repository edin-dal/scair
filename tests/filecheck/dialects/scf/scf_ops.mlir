// RUN: scair-opt %s | filecheck %s

%0, %1, %2, %3 = "test.op"() : () -> (i1, i32, i32, i32)

%4 = "scf.execute_region"() ({}) : () -> (i32)

%5, %6, %7, %8 = "test.op"() : () -> (i32, i32, i32, i32)

%9, %10 = "scf.for"(%5, %6, %7, %8) ({}) : (i32, i32, i32, i32) -> (i1, i1)

%11, %12, %13 = "test.op"() : () -> (index, index, index)
%14, %15, %16 = "test.op"() : () -> (array<i1: 0, 1>, array<i1: 0, 1>, array<i1: 0, 1>)
%17, %18 = "test.op"() : () -> (tensor<1xf32>, tensor<1xf32>)

%19, %20 = "scf.forall"(%11, %12, %13, %17, %18) <{staticLowerBound = array<i1: 0, 1>, staticUpperBound = array<i1: 0, 1>, staticStep = array<i1: 0, 1>, operandSegmentSizes = array<i32: 1, 1, 1, 2>}> ({
  "scf.forall.in_parallel"() ({
  }) : () -> ()
}) : (index, index, index, tensor<1xf32>, tensor<1xf32>) -> (i1, i1)


%21, %22 = "scf.if"(%19) ({
}, {
}) : (i1) -> (i1, i1)

%23, %24 = "scf.parallel"(%11, %12, %13, %21, %22) <{operandSegmentSizes = array<i32: 1, 1, 1, 2>}> ({
  "scf.reduce"(%21, %22) ({
    %25 = "scf.reduce.return"() : () -> (i1)
  }) : (i1, i1) -> ()
}) : (index, index, index, i1, i1) -> (i1, i1)

%25, %26 = "scf.while"(%23) ({
  "scf.condition"(%0, %1, %2) : (i1, i32, i32) -> ()
}, {
}) : (i1) -> (i1, i1)

%27, %28 = "scf.index_switch"(%13, %14) ({
}, {
}) : (index, array<i1: 0, 1>) -> (i1, i1)

"scf.yield"(%27) : (i1) -> ()


// CHECK:       builtin.module {
// CHECK-NEXT:    %0, %1, %2, %3 = "test.op"() : () -> (i1, i32, i32, i32)
// CHECK-NEXT:    %4 = "scf.execute_region"() ({
// CHECK-NEXT:    }) : () -> (i32)
// CHECK-NEXT:    %5, %6, %7, %8 = "test.op"() : () -> (i32, i32, i32, i32)
// CHECK-NEXT:    %9, %10 = "scf.for"(%5, %6, %7, %8) ({
// CHECK-NEXT:    }) : (i32, i32, i32, i32) -> (i1, i1)
// CHECK-NEXT:    %11, %12, %13 = "test.op"() : () -> (index, index, index)
// CHECK-NEXT:    %14, %15, %16 = "test.op"() : () -> (array<i1: 0, 1>, array<i1: 0, 1>, array<i1: 0, 1>)
// CHECK-NEXT:    %17, %18 = "test.op"() : () -> (tensor<1xf32>, tensor<1xf32>)
// CHECK-NEXT:    %19, %20 = "scf.forall"(%11, %12, %13, %17, %18) <{staticLowerBound = array<i1: 0, 1>, staticUpperBound = array<i1: 0, 1>, staticStep = array<i1: 0, 1>, operandSegmentSizes = array<i32: 1, 1, 1, 2>}> ({
// CHECK-NEXT:      "scf.forall.in_parallel"() ({
// CHECK-NEXT:      }) : () -> ()
// CHECK-NEXT:    }) : (index, index, index, tensor<1xf32>, tensor<1xf32>) -> (i1, i1)
// CHECK-NEXT:    %21, %22 = "scf.if"(%19) ({
// CHECK-NEXT:    }, {
// CHECK-NEXT:    }) : (i1) -> (i1, i1)
// CHECK-NEXT:    %23, %24 = "scf.parallel"(%11, %12, %13, %21, %22) <{operandSegmentSizes = array<i32: 1, 1, 1, 2>}> ({
// CHECK-NEXT:      "scf.reduce"(%21, %22) ({
// CHECK-NEXT:        %25 = "scf.reduce.return"() : () -> (i1)
// CHECK-NEXT:      }) : (i1, i1) -> ()
// CHECK-NEXT:    }) : (index, index, index, i1, i1) -> (i1, i1)
// CHECK-NEXT:    %25, %26 = "scf.while"(%23) ({
// CHECK-NEXT:      "scf.condition"(%0, %1, %2) : (i1, i32, i32) -> ()
// CHECK-NEXT:    }, {
// CHECK-NEXT:    }) : (i1) -> (i1, i1)
// CHECK-NEXT:    %27, %28 = "scf.index_switch"(%13, %14) ({
// CHECK-NEXT:    }, {
// CHECK-NEXT:    }) : (index, array<i1: 0, 1>) -> (i1, i1)
// CHECK-NEXT:    "scf.yield"(%27) : (i1) -> ()
// CHECK-NEXT:  }
