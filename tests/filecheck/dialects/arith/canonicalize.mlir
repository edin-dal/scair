// RUN: scair-opt %s -p arith-canonicalize | filecheck %s

"func.func"() <{sym_name = "tripleAddAdd", function_type = (index) -> index}> ({
^bb0(%0: index):
  %1 = "arith.constant"() <{value = 17 : index}> : () -> (index)
  %2 = "arith.constant"() <{value = 42 : index}> : () -> (index)
  %3 = "arith.addi"(%1, %0) : (index, index) -> (index)
  %4 = "arith.addi"(%2, %3) : (index, index) -> (index)
  func.return %4 : index
}) : () -> ()

// CHECK: DJOIADIWDJOIJD