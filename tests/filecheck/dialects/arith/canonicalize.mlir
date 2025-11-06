// RUN: scair-opt %s -p canonicalize | filecheck %s

// CHECK:       builtin.module {

func.func @tripleAddAdd(%0: index) -> index {
  %1 = "arith.constant"() <{value = 17 : index}> : () -> index
  %2 = "arith.constant"() <{value = 42 : index}> : () -> index
  %3 = "arith.addi"(%1, %0) : (index, index) -> index
  %4 = "arith.addi"(%2, %3) : (index, index) -> index
  func.return %4 : index
}
// CHECK-NEXT:    func.func @tripleAddAdd(%0: index) -> index {
// CHECK-NEXT:      %1 = "arith.constant"() <{value = 59 : index}> : () -> index
// CHECK-NEXT:      %2 = "arith.addi"(%0, %1) : (index, index) -> index
// CHECK-NEXT:      func.return %2 : index
// CHECK-NEXT:    }

func.func @tripleAddSub0(%0: index) -> index {
  %1 = "arith.constant"() <{value = 17 : index}> : () -> index
  %2 = "arith.constant"() <{value = 42 : index}> : () -> index
  %3 = "arith.subi"(%1, %0) : (index, index) -> index
  %4 = "arith.addi"(%2, %3) : (index, index) -> index
  func.return %4 : index
}
// CHECK-NEXT:    func.func @tripleAddSub0(%0: index) -> index {
// CHECK-NEXT:      %1 = "arith.constant"() <{value = 59 : index}> : () -> index
// CHECK-NEXT:      %2 = "arith.subi"(%1, %0) : (index, index) -> index
// CHECK-NEXT:      func.return %2 : index
// CHECK-NEXT:    }

func.func @tripleAddSub1(%0: index) -> index {
  %1 = "arith.constant"() <{value = 17 : index}> : () -> index
  %2 = "arith.constant"() <{value = 42 : index}> : () -> index
  %3 = "arith.subi"(%0, %1) : (index, index) -> index
  %4 = "arith.addi"(%2, %3) : (index, index) -> index
  func.return %4 : index
}
// CHECK-NEXT:    func.func @tripleAddSub1(%0: index) -> index {
// CHECK-NEXT:      %1 = "arith.constant"() <{value = -25 : index}> : () -> index
// CHECK-NEXT:      %2 = "arith.addi"(%0, %1) : (index, index) -> index
// CHECK-NEXT:      func.return %2 : index
// CHECK-NEXT:    }

func.func @tripleSubAdd0(%0: index) -> index {
  %1 = "arith.constant"() <{value = 17 : index}> : () -> index
  %2 = "arith.constant"() <{value = 42 : index}> : () -> index
  %3 = "arith.addi"(%1, %0) : (index, index) -> index
  %4 = "arith.subi"(%2, %3) : (index, index) -> index
  func.return %4 : index
}
// CHECK-NEXT:    func.func @tripleSubAdd0(%0: index) -> index {
// CHECK-NEXT:      %1 = "arith.constant"() <{value = 25 : index}> : () -> index
// CHECK-NEXT:      %2 = "arith.subi"(%1, %0) : (index, index) -> index
// CHECK-NEXT:      func.return %2 : index
// CHECK-NEXT:    }

func.func @tripleSubAdd1(%0: index) -> index {
  %1 = "arith.constant"() <{value = 17 : index}> : () -> index
  %2 = "arith.constant"() <{value = 42 : index}> : () -> index
  %3 = "arith.addi"(%1, %0) : (index, index) -> index
  %4 = "arith.subi"(%3, %2) : (index, index) -> index
  func.return %4 : index
}
// CHECK-NEXT:    func.func @tripleSubAdd1(%0: index) -> index {
// CHECK-NEXT:      %1 = "arith.constant"() <{value = -25 : index}> : () -> index
// CHECK-NEXT:      %2 = "arith.addi"(%0, %1) : (index, index) -> index
// CHECK-NEXT:      func.return %2 : index
// CHECK-NEXT:    }

func.func @subSub0(%0: index, %1: index) -> index {
  %2 = "arith.subi"(%0, %1) : (index, index) -> index
  %3 = "arith.subi"(%2, %0) : (index, index) -> index
  func.return %3 : index
}
// CHECK-NEXT:    func.func @subSub0(%0: index, %1: index) -> index {
// CHECK-NEXT:      %2 = "arith.constant"() <{value = 0 : index}> : () -> index
// CHECK-NEXT:      %3 = "arith.subi"(%2, %1) : (index, index) -> index
// CHECK-NEXT:      func.return %3 : index
// CHECK-NEXT:    }

func.func @tripleSubSub0(%0: index) -> index {
  %1 = "arith.constant"() <{value = 17 : index}> : () -> index
  %2 = "arith.constant"() <{value = 42 : index}> : () -> index
  %3 = "arith.subi"(%1, %0) : (index, index) -> index
  %4 = "arith.subi"(%2, %3) : (index, index) -> index
  func.return %4 : index
}
// CHECK-NEXT:    func.func @tripleSubSub0(%0: index) -> index {
// CHECK-NEXT:      %1 = "arith.constant"() <{value = 25 : index}> : () -> index
// CHECK-NEXT:      %2 = "arith.addi"(%0, %1) : (index, index) -> index
// CHECK-NEXT:      func.return %2 : index
// CHECK-NEXT:    }

func.func @tripleSubSub1(%0: index) -> index {
  %1 = "arith.constant"() <{value = 17 : index}> : () -> index
  %2 = "arith.constant"() <{value = 42 : index}> : () -> index
  %3 = "arith.subi"(%1, %0) : (index, index) -> index
  %4 = "arith.subi"(%3, %2) : (index, index) -> index
  func.return %4 : index
}
// CHECK-NEXT:    func.func @tripleSubSub1(%0: index) -> index {
// CHECK-NEXT:      %1 = "arith.constant"() <{value = -25 : index}> : () -> index
// CHECK-NEXT:      %2 = "arith.subi"(%1, %0) : (index, index) -> index
// CHECK-NEXT:      func.return %2 : index
// CHECK-NEXT:    }

func.func @tripleSubSub2(%0: index) -> index {
  %1 = "arith.constant"() <{value = 17 : index}> : () -> index
  %2 = "arith.constant"() <{value = 42 : index}> : () -> index
  %3 = "arith.subi"(%0, %1) : (index, index) -> index
  %4 = "arith.subi"(%2, %3) : (index, index) -> index
  func.return %4 : index
}
// CHECK-NEXT:    func.func @tripleSubSub2(%0: index) -> index {
// CHECK-NEXT:      %1 = "arith.constant"() <{value = 59 : index}> : () -> index
// CHECK-NEXT:      %2 = "arith.subi"(%1, %0) : (index, index) -> index
// CHECK-NEXT:      func.return %2 : index
// CHECK-NEXT:    }

func.func @tripleSubSub3(%0: index) -> index {
  %1 = "arith.constant"() <{value = 17 : index}> : () -> index
  %2 = "arith.constant"() <{value = 42 : index}> : () -> index
  %3 = "arith.subi"(%0, %1) : (index, index) -> index
  %4 = "arith.subi"(%3, %2) : (index, index) -> index
  func.return %4 : index
}
// CHECK-NEXT:    func.func @tripleSubSub3(%0: index) -> index {
// CHECK-NEXT:      %1 = "arith.constant"() <{value = 59 : index}> : () -> index
// CHECK-NEXT:      %2 = "arith.subi"(%0, %1) : (index, index) -> index
// CHECK-NEXT:      func.return %2 : index
// CHECK-NEXT:    }

func.func @subAdd1(%0: index, %1: index) -> index {
  %2 = "arith.addi"(%0, %1) : (index, index) -> index
  %3 = "arith.subi"(%2, %1) : (index, index) -> index
  func.return %3 : index
}
// CHECK-NEXT:    func.func @subAdd1(%0: index, %1: index) -> index {
// CHECK-NEXT:      func.return %0 : index
// CHECK-NEXT:    }

func.func @subAdd2(%0: index, %1: index) -> index {
  %2 = "arith.addi"(%0, %1) : (index, index) -> index
  %3 = "arith.subi"(%2, %0) : (index, index) -> index
  func.return %3 : index
}
// CHECK-NEXT:    func.func @subAdd2(%0: index, %1: index) -> index {
// CHECK-NEXT:      func.return %1 : index
// CHECK-NEXT:    }

func.func @doubleAddSub1(%0: index, %1: index) -> index {
  %2 = "arith.subi"(%0, %1) : (index, index) -> index
  %3 = "arith.addi"(%2, %1) : (index, index) -> index
  func.return %3 : index
}
// CHECK-NEXT:    func.func @doubleAddSub1(%0: index, %1: index) -> index {
// CHECK-NEXT:      func.return %0 : index
// CHECK-NEXT:    }

func.func @doubleAddSub2(%0: index, %1: index) -> index {
  %2 = "arith.subi"(%0, %1) : (index, index) -> index
  %3 = "arith.addi"(%1, %2) : (index, index) -> index
  func.return %3 : index
}
// CHECK-NEXT:    func.func @doubleAddSub2(%0: index, %1: index) -> index {
// CHECK-NEXT:      func.return %0 : index
// CHECK-NEXT:    }

func.func @tripleMulIMulIIndex(%0: index) -> index {
  %1 = "arith.constant"() <{value = 3 : index}> : () -> index
  %2 = "arith.constant"() <{value = 5 : index}> : () -> index
  %3 = "arith.muli"(%0, %1) : (index, index) -> index
  %4 = "arith.muli"(%3, %2) : (index, index) -> index
  func.return %4 : index
}
// CHECK-NEXT:    func.func @tripleMulIMulIIndex(%0: index) -> index {
// CHECK-NEXT:      %1 = "arith.constant"() <{value = 15 : index}> : () -> index
// CHECK-NEXT:      %2 = "arith.muli"(%0, %1) : (index, index) -> index
// CHECK-NEXT:      func.return %2 : index
// CHECK-NEXT:    }


func.func @tripleMulIMulII32(%0: i32) -> i32 {
  %1 = "arith.constant"() <{value = -3 : i32}> : () -> i32
  %2 = "arith.constant"() <{value = 7 : i32}> : () -> i32
  %3 = "arith.muli"(%0, %1) : (i32, i32) -> i32
  %4 = "arith.muli"(%3, %2) : (i32, i32) -> i32
  func.return %4 : i32
}
// CHECK-NEXT:    func.func @tripleMulIMulII32(%0: i32) -> i32 {
// CHECK-NEXT:      %1 = "arith.constant"() <{value = -21 : i32}> : () -> i32
// CHECK-NEXT:      %2 = "arith.muli"(%0, %1) : (i32, i32) -> i32
// CHECK-NEXT:      func.return %2 : i32
// CHECK-NEXT:    }

func.func @tripleMulLargeInt(%0: i256) -> i256 {
  %1 = "arith.constant"() <{value = 3618502788666131213697322783095070105623107215331596699973092056135872020481 : i256}> : () -> i256
  %2 = "arith.constant"() <{value = 1 : i256}> : () -> i256
  %3 = "arith.addi"(%0, %1) : (i256, i256) -> i256
  %4 = "arith.addi"(%3, %2) : (i256, i256) -> i256
  func.return %4 : i256
}
// CHECK-NEXT:    func.func @tripleMulLargeInt(%0: i256) -> i256 {
// CHECK-NEXT:      %1 = "arith.constant"() <{value = 3618502788666131213697322783095070105623107215331596699973092056135872020482 : i256}> : () -> i256
// CHECK-NEXT:      %2 = "arith.addi"(%0, %1) : (i256, i256) -> i256
// CHECK-NEXT:      func.return %2 : i256
// CHECK-NEXT:    }

func.func @addiMuliToSubiRhsI32(%0: i32, %1: i32) -> i32 {
  %2 = "arith.constant"() <{value = -1 : i32}> : () -> i32
  %3 = "arith.muli"(%1, %2) : (i32, i32) -> i32
  %4 = "arith.addi"(%0, %3) : (i32, i32) -> i32
  func.return %4 : i32
}
// CHECK-NEXT:    func.func @addiMuliToSubiRhsI32(%0: i32, %1: i32) -> i32 {
// CHECK-NEXT:      %2 = "arith.subi"(%0, %1) : (i32, i32) -> i32
// CHECK-NEXT:      func.return %2 : i32
// CHECK-NEXT:    }

func.func @addiMuliToSubiRhsIndex(%0: index, %1: index) -> index {
  %2 = "arith.constant"() <{value = -1 : index}> : () -> index
  %3 = "arith.muli"(%1, %2) : (index, index) -> index
  %4 = "arith.addi"(%0, %3) : (index, index) -> index
  func.return %4 : index
}
// CHECK-NEXT:    func.func @addiMuliToSubiRhsIndex(%0: index, %1: index) -> index {
// CHECK-NEXT:      %2 = "arith.subi"(%0, %1) : (index, index) -> index
// CHECK-NEXT:      func.return %2 : index
// CHECK-NEXT:    }

// TODO: This requires our matching to abstract away from matching scalars specifically.
// Just not a priority here at the time of writing!
// func.func @addiMuliToSubiRhsVector(%0: vector<3xi64>, %1: vector<3xi64>) -> vector<3xi64> {
//   %2 = "arith.constant"() <{value = dense<-1> : vector<3xi64>}> : () -> vector<3xi64>
//   %3 = "arith.muli"(%1, %2) : (vector<3xi64>, vector<3xi64>) -> vector<3xi64>
//   %4 = "arith.addi"(%0, %3) : (vector<3xi64>, vector<3xi64>) -> vector<3xi64>
//   func.return %4 : vector<3xi64>
// }

func.func @addiMuliToSubiLhsI32(%0: i32, %1: i32) -> i32 {
  %2 = "arith.constant"() <{value = -1 : i32}> : () -> i32
  %3 = "arith.muli"(%1, %2) : (i32, i32) -> i32
  %4 = "arith.addi"(%3, %0) : (i32, i32) -> i32
  func.return %4 : i32
}
// CHECK-NEXT:    func.func @addiMuliToSubiLhsI32(%0: i32, %1: i32) -> i32 {
// CHECK-NEXT:      %2 = "arith.subi"(%0, %1) : (i32, i32) -> i32
// CHECK-NEXT:      func.return %2 : i32
// CHECK-NEXT:    }

func.func @addiMuliToSubiLhsIndex(%0: index, %1: index) -> index {
  %2 = "arith.constant"() <{value = -1 : index}> : () -> index
  %3 = "arith.muli"(%1, %2) : (index, index) -> index
  %4 = "arith.addi"(%3, %0) : (index, index) -> index
  func.return %4 : index
}
// CHECK-NEXT:    func.func @addiMuliToSubiLhsIndex(%0: index, %1: index) -> index {
// CHECK-NEXT:      %2 = "arith.subi"(%0, %1) : (index, index) -> index
// CHECK-NEXT:      func.return %2 : index
// CHECK-NEXT:    }


// TODO: This requires our matching to abstract away from matching scalars specifically.
// Just not a priority here at the time of writing!
// func.func @addiMuliToSubiLhsVector(%0: vector<3xi64>, %1: vector<3xi64>) -> vector<3xi64> {
//   %2 = "arith.constant"() <{value = dense<-1> : vector<3xi64>}> : () -> vector<3xi64>
//   %3 = "arith.muli"(%1, %2) : (vector<3xi64>, vector<3xi64>) -> vector<3xi64>
//   %4 = "arith.addi"(%3, %0) : (vector<3xi64>, vector<3xi64>) -> vector<3xi64>
//   func.return %4 : vector<3xi64>
// }

// CHECK-NEXT:  }
