// RUN: scair-opt %s -p arith-canonicalize | filecheck %s

// CHECK:       builtin.module {

"func.func"() <{sym_name = "tripleAddAdd", function_type = (index) -> index}> ({
^bb0(%0: index):
  %1 = "arith.constant"() <{value = 17 : index}> : () -> (index)
  %2 = "arith.constant"() <{value = 42 : index}> : () -> (index)
  %3 = "arith.addi"(%1, %0) : (index, index) -> (index)
  %4 = "arith.addi"(%2, %3) : (index, index) -> (index)
  func.return %4 : index
}) : () -> ()
// CHECK-NEXT:    "func.func"() <{sym_name = "tripleAddAdd", function_type = (index) -> index}> ({
// CHECK-NEXT:    ^bb0(%0: index):
// CHECK-NEXT:      %1 = "arith.constant"() <{value = 59 : index}> : () -> (index)
// CHECK-NEXT:      %2 = "arith.addi"(%0, %1) : (index, index) -> (index)
// CHECK-NEXT:      func.return %2 : index
// CHECK-NEXT:    }) : () -> ()

// CHECK-NEXT:  }