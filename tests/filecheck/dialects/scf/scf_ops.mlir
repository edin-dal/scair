// RUN: scair-opt %s | filecheck %s

%0, %1, %2, %3 = "test.op"() : () -> (i1, i32, i32, i32)

%4 = "scf.execute_region"() ({}) : () -> i32

%5, %6, %7, %8 = "test.op"() : () -> (i32, i32, i32, i32)

%9, %10 = "scf.for"(%5, %6, %7, %8) ({}) : (i32, i32, i32, i32) -> (i1, i1)

%11, %12, %13 = "test.op"() : () -> (index, index, index)
%14, %15 = "test.op"() : () -> (tensor<1xf32>, tensor<1xf32>)

%16, %17 = "scf.forall"(%11, %12, %13, %14, %15) <{staticLowerBound = array<i1: 0, 1>, staticUpperBound = array<i1: 0, 1>, staticStep = array<i1: 0, 1>, operandSegmentSizes = array<i32: 1, 1, 1, 2>}> ({
  "scf.forall.in_parallel"() ({
  }) : () -> ()
}) : (index, index, index, tensor<1xf32>, tensor<1xf32>) -> (i1, i1)


%18, %19 = "scf.if"(%16) ({
}, {
}) : (i1) -> (i1, i1)

%20, %21 = "scf.parallel"(%11, %12, %13, %18, %19) <{operandSegmentSizes = array<i32: 1, 1, 1, 2>}> ({
^bb0(%22: i1, %23: i1):
  scf.reduce(%22, %23 : i1, i1) {
  ^bb1(%24: i1, %25: i1):
    scf.reduce.return %24 : i1
  }, {
  ^bb1(%24: i1, %25: i1):
    scf.reduce.return %25 : i1
  }
}) : (index, index, index, i1, i1) -> (i1, i1)

%22, %23 = "scf.while"(%20) ({
  "scf.condition"(%0, %1, %2) : (i1, i32, i32) -> ()
}, {
}) : (i1) -> (i1, i1)

%24, %25 = "scf.index_switch"(%13) <{cases = array<i1: 0, 1>}> ({
}, {
}) : (index) -> (i1, i1)

scf.yield %24 : i1


// CHECK:       builtin.module {
// CHECK-NEXT:    %0, %1, %2, %3 = "test.op"() : () -> (i1, i32, i32, i32)
// CHECK-NEXT:    %4 = "scf.execute_region"() ({
// CHECK-NEXT:    }) : () -> i32
// CHECK-NEXT:    %5, %6, %7, %8 = "test.op"() : () -> (i32, i32, i32, i32)
// CHECK-NEXT:    %9, %10 = "scf.for"(%5, %6, %7, %8) ({
// CHECK-NEXT:    }) : (i32, i32, i32, i32) -> (i1, i1)
// CHECK-NEXT:    %11, %12, %13 = "test.op"() : () -> (index, index, index)
// CHECK-NEXT:    %14, %15 = "test.op"() : () -> (tensor<1xf32>, tensor<1xf32>)
// CHECK-NEXT:    %16, %17 = "scf.forall"(%11, %12, %13, %14, %15) <{staticLowerBound = array<i1: 0, 1>, staticUpperBound = array<i1: 0, 1>, staticStep = array<i1: 0, 1>, operandSegmentSizes = array<i32: 1, 1, 1, 2>}> ({
// CHECK-NEXT:      "scf.forall.in_parallel"() ({
// CHECK-NEXT:      }) : () -> ()
// CHECK-NEXT:    }) : (index, index, index, tensor<1xf32>, tensor<1xf32>) -> (i1, i1)
// CHECK-NEXT:    %18, %19 = "scf.if"(%16) ({
// CHECK-NEXT:    }, {
// CHECK-NEXT:    }) : (i1) -> (i1, i1)
// CHECK-NEXT:    %20, %21 = "scf.parallel"(%11, %12, %13, %18, %19) <{operandSegmentSizes = array<i32: 1, 1, 1, 2>}> ({
// CHECK-NEXT:    ^bb0(%22: i1, %23: i1):
// CHECK-NEXT:      scf.reduce(%22, %23 : i1, i1) {
// CHECK-NEXT:      ^bb1(%24: i1, %25: i1):
// CHECK-NEXT:        scf.reduce.return %24 : i1
// CHECK-NEXT:      }, {
// CHECK-NEXT:      ^bb1(%24: i1, %25: i1):
// CHECK-NEXT:        scf.reduce.return %25 : i1
// CHECK-NEXT:      }
// CHECK-NEXT:    }) : (index, index, index, i1, i1) -> (i1, i1)
// CHECK-NEXT:    %22, %23 = "scf.while"(%20) ({
// CHECK-NEXT:      "scf.condition"(%0, %1, %2) : (i1, i32, i32) -> ()
// CHECK-NEXT:    }, {
// CHECK-NEXT:    }) : (i1) -> (i1, i1)
// CHECK-NEXT:    %24, %25 = "scf.index_switch"(%13) <{cases = array<i1: 0, 1>}> ({
// CHECK-NEXT:    }, {
// CHECK-NEXT:    }) : (index) -> (i1, i1)
// CHECK-NEXT:    scf.yield %24 : i1
// CHECK-NEXT:  }
