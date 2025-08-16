// RUN: scair-opt %s -p canonicalize | filecheck %s

// CHECK:       builtin.module {

func.func @tripleAddAdd(%0: index) -> index {
  %1 = "arith.constant"() <{value = 17 : index}> : () -> (index)
  %2 = "arith.constant"() <{value = 42 : index}> : () -> (index)
  %3 = "arith.addi"(%1, %0) : (index, index) -> (index)
  %4 = "arith.addi"(%2, %3) : (index, index) -> (index)
  func.return %4 : index
}
// CHECK-NEXT:    func.func @tripleAddAdd(%0: index) -> index {
// CHECK-NEXT:      %1 = "arith.constant"() <{value = 59 : index}> : () -> (index)
// CHECK-NEXT:      %2 = "arith.addi"(%0, %1) : (index, index) -> (index)
// CHECK-NEXT:      func.return %2 : index
// CHECK-NEXT:    }

func.func @tripleAddSub0(%0: index) -> index {
  %1 = "arith.constant"() <{value = 17 : index}> : () -> (index)
  %2 = "arith.constant"() <{value = 42 : index}> : () -> (index)
  %3 = "arith.subi"(%1, %0) : (index, index) -> (index)
  %4 = "arith.addi"(%2, %3) : (index, index) -> (index)
  func.return %4 : index
}
// CHECK-NEXT:    func.func @tripleAddSub0(%0: index) -> index {
// CHECK-NEXT:      %1 = "arith.constant"() <{value = 59 : index}> : () -> (index)
// CHECK-NEXT:      %2 = "arith.subi"(%1, %0) : (index, index) -> (index)
// CHECK-NEXT:      func.return %2 : index
// CHECK-NEXT:    }

func.func @tripleAddSub1(%0: index) -> index {
  %1 = "arith.constant"() <{value = 17 : index}> : () -> (index)
  %2 = "arith.constant"() <{value = 42 : index}> : () -> (index)
  %3 = "arith.subi"(%0, %1) : (index, index) -> (index)
  %4 = "arith.addi"(%2, %3) : (index, index) -> (index)
  func.return %4 : index
}
// CHECK-NEXT:    func.func @tripleAddSub1(%0: index) -> index {
// CHECK-NEXT:      %1 = "arith.constant"() <{value = -25 : index}> : () -> (index)
// CHECK-NEXT:      %2 = "arith.addi"(%0, %1) : (index, index) -> (index)
// CHECK-NEXT:      func.return %2 : index
// CHECK-NEXT:    }

func.func @tripleSubAdd0(%0: index) -> index {
  %1 = "arith.constant"() <{value = 17 : index}> : () -> (index)
  %2 = "arith.constant"() <{value = 42 : index}> : () -> (index)
  %3 = "arith.addi"(%1, %0) : (index, index) -> (index)
  %4 = "arith.subi"(%2, %3) : (index, index) -> (index)
  func.return %4 : index
}
// CHECK-NEXT:    func.func @tripleSubAdd0(%0: index) -> index {
// CHECK-NEXT:      %1 = "arith.constant"() <{value = 25 : index}> : () -> (index)
// CHECK-NEXT:      %2 = "arith.subi"(%1, %0) : (index, index) -> (index)
// CHECK-NEXT:      func.return %2 : index
// CHECK-NEXT:    }

func.func @tripleSubAdd1(%0: index) -> index {
  %1 = "arith.constant"() <{value = 17 : index}> : () -> (index)
  %2 = "arith.constant"() <{value = 42 : index}> : () -> (index)
  %3 = "arith.addi"(%1, %0) : (index, index) -> (index)
  %4 = "arith.subi"(%3, %2) : (index, index) -> (index)
  func.return %4 : index
}
// CHECK-NEXT:    func.func @tripleSubAdd1(%0: index) -> index {
// CHECK-NEXT:      %1 = "arith.constant"() <{value = -25 : index}> : () -> (index)
// CHECK-NEXT:      %2 = "arith.addi"(%0, %1) : (index, index) -> (index)
// CHECK-NEXT:      func.return %2 : index
// CHECK-NEXT:    }

func.func @subSub0(%0: index, %1: index) -> index {
  %2 = "arith.subi"(%0, %1) : (index, index) -> (index)
  %3 = "arith.subi"(%2, %0) : (index, index) -> (index)
  func.return %3 : index
}
// CHECK-NEXT:    func.func @subSub0(%0: index, %1: index) -> index {
// CHECK-NEXT:      %2 = "arith.constant"() <{value = 0 : index}> : () -> (index)
// CHECK-NEXT:      %3 = "arith.subi"(%2, %1) : (index, index) -> (index)
// CHECK-NEXT:      func.return %3 : index
// CHECK-NEXT:    }

func.func @tripleSubSub0(%0: index) -> index {
  %1 = "arith.constant"() <{value = 17 : index}> : () -> (index)
  %2 = "arith.constant"() <{value = 42 : index}> : () -> (index)
  %3 = "arith.subi"(%1, %0) : (index, index) -> (index)
  %4 = "arith.subi"(%2, %3) : (index, index) -> (index)
  func.return %4 : index
}
// CHECK-NEXT:    func.func @tripleSubSub0(%0: index) -> index {
// CHECK-NEXT:      %1 = "arith.constant"() <{value = 25 : index}> : () -> (index)
// CHECK-NEXT:      %2 = "arith.addi"(%0, %1) : (index, index) -> (index)
// CHECK-NEXT:      func.return %2 : index
// CHECK-NEXT:    }

func.func @tripleSubSub1(%0: index) -> index {
  %1 = "arith.constant"() <{value = 17 : index}> : () -> (index)
  %2 = "arith.constant"() <{value = 42 : index}> : () -> (index)
  %3 = "arith.subi"(%1, %0) : (index, index) -> (index)
  %4 = "arith.subi"(%3, %2) : (index, index) -> (index)
  func.return %4 : index
}
// CHECK-NEXT:    func.func @tripleSubSub1(%0: index) -> index {
// CHECK-NEXT:      %1 = "arith.constant"() <{value = -25 : index}> : () -> (index)
// CHECK-NEXT:      %2 = "arith.subi"(%1, %0) : (index, index) -> (index)
// CHECK-NEXT:      func.return %2 : index
// CHECK-NEXT:    }

func.func @tripleSubSub2(%0: index) -> index {
  %1 = "arith.constant"() <{value = 17 : index}> : () -> (index)
  %2 = "arith.constant"() <{value = 42 : index}> : () -> (index)
  %3 = "arith.subi"(%0, %1) : (index, index) -> (index)
  %4 = "arith.subi"(%2, %3) : (index, index) -> (index)
  func.return %4 : index
}
// CHECK-NEXT:    func.func @tripleSubSub2(%0: index) -> index {
// CHECK-NEXT:      %1 = "arith.constant"() <{value = 59 : index}> : () -> (index)
// CHECK-NEXT:      %2 = "arith.subi"(%1, %0) : (index, index) -> (index)
// CHECK-NEXT:      func.return %2 : index
// CHECK-NEXT:    }

func.func @tripleSubSub3(%0: index) -> index {
  %1 = "arith.constant"() <{value = 17 : index}> : () -> (index)
  %2 = "arith.constant"() <{value = 42 : index}> : () -> (index)
  %3 = "arith.subi"(%0, %1) : (index, index) -> (index)
  %4 = "arith.subi"(%3, %2) : (index, index) -> (index)
  func.return %4 : index
}
// CHECK-NEXT:    func.func @tripleSubSub3(%0: index) -> index {
// CHECK-NEXT:      %1 = "arith.constant"() <{value = 59 : index}> : () -> (index)
// CHECK-NEXT:      %2 = "arith.subi"(%0, %1) : (index, index) -> (index)
// CHECK-NEXT:      func.return %2 : index
// CHECK-NEXT:    }

// CHECK-NEXT:  }
