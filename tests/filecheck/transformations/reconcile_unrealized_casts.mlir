// RUN: scair-opt %s --allow-unregistered-dialect -p reconcile-unrealized-casts | filecheck %s

builtin.module {
  func.func @unusedCast(%0: i64) -> i64 {
    %1 = "builtin.unrealized_conversion_cast"(%0) : (i64) -> (i32)
    func.return %0 : i64
  }
  func.func @sameTypes(%0: i64) -> i64 {
    %1 = "test.op"() : () -> (i64)
    %2 = "builtin.unrealized_conversion_cast"(%1) : (i64) -> (i64)
    func.return %2 : i64
  }
  func.func @pair(%arg0: i64) -> i64 {
    %0 = "builtin.unrealized_conversion_cast"(%arg0) : (i64) -> i32
    %1 = "builtin.unrealized_conversion_cast"(%0) : (i32) -> i64
    func.return %1 : i64
  }
  func.func @symmetricChain(%arg0: i64) -> i64 {
      %0 = "builtin.unrealized_conversion_cast"(%arg0) : (i64) -> i32
      %1 = "builtin.unrealized_conversion_cast"(%0) : (i32) -> i1
      %2 = "builtin.unrealized_conversion_cast" (%1) : (i1) -> i32
      %3 = "builtin.unrealized_conversion_cast"(%2) : (i32) -> i64
      func.return %3 : i64
  }
  func.func @asymmetricChain(%arg0: i64) -> i64 {
      %0 = "builtin.unrealized_conversion_cast"(%arg0) : (i64) -> i32
      %1 = "builtin.unrealized_conversion_cast"(%0) : (i32) -> i1
      %2 = "builtin.unrealized_conversion_cast"(%1) : (i1) -> i64
      func.return %2 : i64
  }
  func.func @unusedChain(%arg0: i64) -> i64 {
      %0 = "builtin.unrealized_conversion_cast"(%arg0) : (i64) -> i32
      %1 = "builtin.unrealized_conversion_cast"(%0) : (i32) -> i1
      func.return %arg0 : i64
  }
  func.func @bifurcation(%arg0: i64) -> i64 {
      %0 = "builtin.unrealized_conversion_cast"(%arg0) : (i64) -> i32
      %1 = "builtin.unrealized_conversion_cast"(%0) : (i32) -> i1
      %2 = "builtin.unrealized_conversion_cast"(%1) : (i1) -> i64
      %3 = "builtin.unrealized_conversion_cast" (%1) : (i1) -> i32
      %4 = "builtin.unrealized_conversion_cast"(%3) : (i32) -> i64
      %5 = "arith.addi"(%2, %4) : (i64, i64) -> i64
      func.return %5 : i64
  }
  func.func @unusedBifurcation(%arg0: i64) -> i64 {
      %0 = "builtin.unrealized_conversion_cast"(%arg0) : (i64) -> i32
      %1 = "builtin.unrealized_conversion_cast"(%0) : (i32) -> i1
      %2 = "builtin.unrealized_conversion_cast"(%1) : (i1) -> i64
      %3 = "builtin.unrealized_conversion_cast"(%0) : (i32) -> i64
      %4 = "arith.addi"(%arg0, %3) : (i64, i64) -> i64
      func.return %4 : i64
  }
  func.func @liveSingleCast(%arg0: i64) -> i32 {
      %0 = "builtin.unrealized_conversion_cast"(%arg0) : (i64) -> i32
      func.return %0 : i32
  }
  func.func @liveChain(%arg0: i64) -> i32 {
      %0 = "builtin.unrealized_conversion_cast"(%arg0) : (i64) -> i1
      %1 = "builtin.unrealized_conversion_cast"(%0) : (i1) -> i32
      func.return %1 : i32
  }
  func.func @liveBifurcation(%arg0: i64) -> i64 {
      %0 = "builtin.unrealized_conversion_cast"(%arg0) : (i64) -> i32
      %1 = "builtin.unrealized_conversion_cast"(%0) : (i32) -> i64
      %2 = "builtin.unrealized_conversion_cast"(%0) : (i32) -> i1
      %3 = "arith.extsi"(%2) : (i1) -> i64
      %4 = "arith.addi"(%1, %3) : (i64, i64) -> i64
      func.return %4 : i64
  }
  func.func @deadNToOneCast(%arg0: index, %arg1: index) {
      %0 = "builtin.unrealized_conversion_cast"(%arg0, %arg1) : (index, index) -> i64
      func.return
  }
  func.func @swappingOperands(%arg0: index, %arg1: index) -> (index, index) {
      %0:2 = "builtin.unrealized_conversion_cast"(%arg0, %arg1) : (index, index) -> (i64, i64)
      %1:2 = "builtin.unrealized_conversion_cast"(%0#1, %0#0) : (i64, i64) -> (i32, i32)
      %2:2 = "builtin.unrealized_conversion_cast"(%1#0, %1#1) : (i32, i32) -> (index, index)
      func.return %2#0, %2#1 : index, index
  }
  func.func @matchingOperands(%arg0: index, %arg1: index) -> (index, index) {
      %0:2 = "builtin.unrealized_conversion_cast"(%arg0, %arg1) : (index, index) -> (i64, i64)
      %1:3 = "builtin.unrealized_conversion_cast"(%0#0, %0#1) : (i64, i64) -> (i32, i32, i32)
      %2:2 = "builtin.unrealized_conversion_cast"(%1#0, %1#1, %1#2) : (i32, i32, i32) -> (index, index)
      func.return %2#0, %2#1 : index, index
  }
  func.func @emptyCast() -> index {
      %0 = "builtin.unrealized_conversion_cast"() : () -> index
      func.return %0 : index
  }

  // Infra is a bit too brittle at the moment to debug those; also not really desired at time of writing.

  // "test.graph_region"() ({
  //   %0 = "builtin.unrealized_conversion_cast"(%2) : (i32) -> i64
  //   %1 = "builtin.unrealized_conversion_cast"(%0) : (i64) -> i16
  //   %2 = "builtin.unrealized_conversion_cast"(%1) : (i16) -> i32
  //   "test.return"() : () -> ()
  // }) : () -> ()
  // "test.graph_region"() ({
  //   %0 = "builtin.unrealized_conversion_cast"(%2) : (i32) -> i64
  //   %1 = "builtin.unrealized_conversion_cast"(%0) : (i64) -> i16
  //   %2 = "builtin.unrealized_conversion_cast"(%1) : (i16) -> i32
  //   "test.user"(%2) : (i32) -> ()
  //   "test.return"() : () -> ()
  // }) : () -> ()
  // "test.graph_region"() ({
  //   %0 = "builtin.unrealized_conversion_cast"(%0) : (i32) -> i32
  //   "test.return"() : () -> ()
  // }) : () -> ()
  // "test.graph_region"() ({
  //   %cst = "arith.constant"() <{value = 0 : i32}> : () -> i32
  //   %0, %1 = "builtin.unrealized_conversion_cast"(%cst, %1) : (i32, i32) -> (i32, i32)
  //   "test.user"(%0) : (i32) -> ()
  //   "test.return"() : () -> ()
  // }) : () -> ()
}

// CHECK:       builtin.module {
// CHECK-NEXT:    func.func @unusedCast(%0: i64) -> i64 {
// CHECK-NEXT:      func.return %0 : i64
// CHECK-NEXT:    }
// CHECK-NEXT:    func.func @sameTypes(%0: i64) -> i64 {
// CHECK-NEXT:      %1 = "test.op"() : () -> (i64)
// CHECK-NEXT:      func.return %1 : i64
// CHECK-NEXT:    }
// CHECK-NEXT:    func.func @pair(%0: i64) -> i64 {
// CHECK-NEXT:      func.return %0 : i64
// CHECK-NEXT:    }
// CHECK-NEXT:    func.func @symmetricChain(%0: i64) -> i64 {
// CHECK-NEXT:      func.return %0 : i64
// CHECK-NEXT:    }
// CHECK-NEXT:    func.func @asymmetricChain(%0: i64) -> i64 {
// CHECK-NEXT:      func.return %0 : i64
// CHECK-NEXT:    }
// CHECK-NEXT:    func.func @unusedChain(%0: i64) -> i64 {
// CHECK-NEXT:      func.return %0 : i64
// CHECK-NEXT:    }
// CHECK-NEXT:    func.func @bifurcation(%0: i64) -> i64 {
// CHECK-NEXT:      %1 = "arith.addi"(%0, %0) : (i64, i64) -> (i64)
// CHECK-NEXT:      func.return %1 : i64
// CHECK-NEXT:    }
// CHECK-NEXT:    func.func @unusedBifurcation(%0: i64) -> i64 {
// CHECK-NEXT:      %1 = "arith.addi"(%0, %0) : (i64, i64) -> (i64)
// CHECK-NEXT:      func.return %1 : i64
// CHECK-NEXT:    }
// CHECK-NEXT:    func.func @liveSingleCast(%0: i64) -> i32 {
// CHECK-NEXT:      %1 = "builtin.unrealized_conversion_cast"(%0) : (i64) -> (i32)
// CHECK-NEXT:      func.return %1 : i32
// CHECK-NEXT:    }
// CHECK-NEXT:    func.func @liveChain(%0: i64) -> i32 {
// CHECK-NEXT:      %1 = "builtin.unrealized_conversion_cast"(%0) : (i64) -> (i1)
// CHECK-NEXT:      %2 = "builtin.unrealized_conversion_cast"(%1) : (i1) -> (i32)
// CHECK-NEXT:      func.return %2 : i32
// CHECK-NEXT:    }
// CHECK-NEXT:    func.func @liveBifurcation(%0: i64) -> i64 {
// CHECK-NEXT:      %1 = "builtin.unrealized_conversion_cast"(%0) : (i64) -> (i32)
// CHECK-NEXT:      %2 = "builtin.unrealized_conversion_cast"(%1) : (i32) -> (i1)
// CHECK-NEXT:      %3 = "arith.extsi"(%2) : (i1) -> (i64)
// CHECK-NEXT:      %4 = "arith.addi"(%0, %3) : (i64, i64) -> (i64)
// CHECK-NEXT:      func.return %4 : i64
// CHECK-NEXT:    }
// CHECK-NEXT:    func.func @deadNToOneCast(%0: index, %1: index) {
// CHECK-NEXT:      func.return
// CHECK-NEXT:    }
// CHECK-NEXT:    func.func @swappingOperands(%0: index, %1: index) -> (index, index) {
// CHECK-NEXT:      %2, %3 = "builtin.unrealized_conversion_cast"(%0, %1) : (index, index) -> (i64, i64)
// CHECK-NEXT:      %4, %5 = "builtin.unrealized_conversion_cast"(%3, %2) : (i64, i64) -> (i32, i32)
// CHECK-NEXT:      %6, %7 = "builtin.unrealized_conversion_cast"(%4, %5) : (i32, i32) -> (index, index)
// CHECK-NEXT:      func.return %6, %7 : index, index
// CHECK-NEXT:    }
// CHECK-NEXT:    func.func @matchingOperands(%0: index, %1: index) -> (index, index) {
// CHECK-NEXT:      func.return %0, %1 : index, index
// CHECK-NEXT:    }
// CHECK-NEXT:    func.func @emptyCast() -> index {
// CHECK-NEXT:      %0 = "builtin.unrealized_conversion_cast"() : () -> (index)
// CHECK-NEXT:      func.return %0 : index
// CHECK-NEXT:    }
// CHECK-NEXT:  }
