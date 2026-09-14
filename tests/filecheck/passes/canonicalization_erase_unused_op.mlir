// RUN: scair-opt %s -p canonicalize | filecheck %s -DFILE=%s

func.func @dead_addi(%0: index) {
  %1 = "arith.constant"() <{value = 0 : index}> : () -> index
  %2 = "arith.addi"(%0, %1) <{overflowFlags = #arith.overflow<none>}> : (index, index) -> index
  func.return
}

// CHECK: builtin.module {
// CHECK-NEXT:   func.func @dead_addi(%0: index) {
// CHECK-NEXT:     func.return
// CHECK-NEXT:   }
// CHECK-NEXT: }
