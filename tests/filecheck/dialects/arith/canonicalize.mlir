// RUN: scair-opt %s -p canonicalize | filecheck %s

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

"func.func"() <{"sym_name" = "tripleAddSub0", "function_type" = (index) -> index}> ({
^0(%arg0 : index):
  %c17 = "arith.constant"() <{"value" = 17 : index}> : () -> index
  %c42 = "arith.constant"() <{"value" = 42 : index}> : () -> index
  %add1 = "arith.subi"(%c17, %arg0) : (index, index) -> index
  %add2 = "arith.addi"(%c42, %add1) : (index, index) -> index
  func.return %add2 : index
}) : () -> ()
// CHECK-NEXT:    "func.func"() <{sym_name = "tripleAddSub0", function_type = (index) -> index}> ({
// CHECK-NEXT:    ^bb0(%0: index):
// CHECK-NEXT:      %1 = "arith.constant"() <{value = 59 : index}> : () -> (index)
// CHECK-NEXT:      %2 = "arith.subi"(%1, %0) : (index, index) -> (index)
// CHECK-NEXT:      func.return %2 : index
// CHECK-NEXT:    }) : () -> ()

"func.func"() <{"sym_name" = "tripleAddSub1", "function_type" = (index) -> index}> ({
^0(%arg0 : index):
  %c17 = "arith.constant"() <{"value" = 17 : index}> : () -> index
  %c42 = "arith.constant"() <{"value" = 42 : index}> : () -> index
  %add1 = "arith.subi"(%arg0, %c17) : (index, index) -> index
  %add2 = "arith.addi"(%c42, %add1) : (index, index) -> index
  func.return %add2 : index
}) : () -> ()
// CHECK-NEXT:    "func.func"() <{sym_name = "tripleAddSub1", function_type = (index) -> index}> ({
// CHECK-NEXT:    ^bb0(%0: index):
// CHECK-NEXT:      %1 = "arith.constant"() <{value = -25 : index}> : () -> (index)
// CHECK-NEXT:      %2 = "arith.addi"(%0, %1) : (index, index) -> (index)
// CHECK-NEXT:      func.return %2 : index
// CHECK-NEXT:    }) : () -> ()

"func.func"() <{function_type = (index) -> index, sym_name = "tripleSubAdd0"}> ({
^bb0(%arg15: index):
  %30 = "arith.constant"() <{value = 17 : index}> : () -> index
  %31 = "arith.constant"() <{value = 42 : index}> : () -> index
  %32 = "arith.addi"(%30, %arg15) : (index, index) -> index
  %33 = "arith.subi"(%31, %32) : (index, index) -> index
  "func.return"(%33) : (index) -> ()
}) : () -> ()
// CHECK-NEXT:    "func.func"() <{sym_name = "tripleSubAdd0", function_type = (index) -> index}> ({
// CHECK-NEXT:    ^bb0(%0: index):
// CHECK-NEXT:      %1 = "arith.constant"() <{value = 25 : index}> : () -> (index)
// CHECK-NEXT:      %2 = "arith.subi"(%1, %0) : (index, index) -> (index)
// CHECK-NEXT:      func.return %2 : index
// CHECK-NEXT:    }) : () -> ()

"func.func"() <{function_type = (index) -> index, sym_name = "tripleSubAdd1"}> ({
^bb0(%arg14: index):
  %26 = "arith.constant"() <{value = 17 : index}> : () -> index
  %27 = "arith.constant"() <{value = 42 : index}> : () -> index
  %28 = "arith.addi"(%26, %arg14) : (index, index) -> index
  %29 = "arith.subi"(%28, %27) : (index, index) -> index
  "func.return"(%29) : (index) -> ()
}) : () -> ()
// CHECK-NEXT:    "func.func"() <{sym_name = "tripleSubAdd1", function_type = (index) -> index}> ({
// CHECK-NEXT:    ^bb0(%0: index):
// CHECK-NEXT:      %1 = "arith.constant"() <{value = -25 : index}> : () -> (index)
// CHECK-NEXT:      %2 = "arith.addi"(%0, %1) : (index, index) -> (index)
// CHECK-NEXT:      func.return %2 : index
// CHECK-NEXT:    }) : () -> ()

"func.func"() <{function_type = (index, index) -> index, sym_name = "subSub0"}> ({
^bb0(%arg12: index, %arg13: index):
  %24 = "arith.subi"(%arg12, %arg13) : (index, index) -> index
  %25 = "arith.subi"(%24, %arg12) : (index, index) -> index
  "func.return"(%25) : (index) -> ()
}) : () -> ()
// CHECK-NEXT:    "func.func"() <{sym_name = "subSub0", function_type = (index, index) -> index}> ({
// CHECK-NEXT:    ^bb0(%0: index, %1: index):
// CHECK-NEXT:      %2 = "arith.constant"() <{value = 0 : index}> : () -> (index)
// CHECK-NEXT:      %3 = "arith.subi"(%2, %1) : (index, index) -> (index)
// CHECK-NEXT:      func.return %3 : index
// CHECK-NEXT:    }) : () -> ()

"func.func"() <{function_type = (index) -> index, sym_name = "tripleSubSub0"}> ({
^bb0(%arg11: index):
  %20 = "arith.constant"() <{value = 17 : index}> : () -> index
  %21 = "arith.constant"() <{value = 42 : index}> : () -> index
  %22 = "arith.subi"(%20, %arg11) : (index, index) -> index
  %23 = "arith.subi"(%21, %22) : (index, index) -> index
  "func.return"(%23) : (index) -> ()
}) : () -> ()
// CHECK-NEXT:    "func.func"() <{sym_name = "tripleSubSub0", function_type = (index) -> index}> ({
// CHECK-NEXT:    ^bb0(%0: index):
// CHECK-NEXT:      %1 = "arith.constant"() <{value = 25 : index}> : () -> (index)
// CHECK-NEXT:      %2 = "arith.addi"(%0, %1) : (index, index) -> (index)
// CHECK-NEXT:      func.return %2 : index
// CHECK-NEXT:    }) : () -> ()

"func.func"() <{function_type = (index) -> index, sym_name = "tripleSubSub1"}> ({
^bb0(%arg10: index):
  %16 = "arith.constant"() <{value = 17 : index}> : () -> index
  %17 = "arith.constant"() <{value = 42 : index}> : () -> index
  %18 = "arith.subi"(%16, %arg10) : (index, index) -> index
  %19 = "arith.subi"(%18, %17) : (index, index) -> index
  "func.return"(%19) : (index) -> ()
}) : () -> ()// CHECK-NEXT:    "func.func"() <{sym_name = "tripleSubSub1", function_type = (index) -> index}> ({
// CHECK-NEXT:    ^bb0(%0: index):
// CHECK-NEXT:      %1 = "arith.constant"() <{value = -25 : index}> : () -> (index)
// CHECK-NEXT:      %2 = "arith.subi"(%1, %0) : (index, index) -> (index)
// CHECK-NEXT:      func.return %2 : index
// CHECK-NEXT:    }) : () -> ()

"func.func"() <{function_type = (index) -> index, sym_name = "tripleSubSub2"}> ({
^bb0(%arg9: index):
  %12 = "arith.constant"() <{value = 17 : index}> : () -> index
  %13 = "arith.constant"() <{value = 42 : index}> : () -> index
  %14 = "arith.subi"(%arg9, %12) : (index, index) -> index
  %15 = "arith.subi"(%13, %14) : (index, index) -> index
  "func.return"(%15) : (index) -> ()
}) : () -> ()
// CHECK-NEXT:    "func.func"() <{sym_name = "tripleSubSub2", function_type = (index) -> index}> ({
// CHECK-NEXT:    ^bb0(%0: index):
// CHECK-NEXT:      %1 = "arith.constant"() <{value = 59 : index}> : () -> (index)
// CHECK-NEXT:      %2 = "arith.subi"(%1, %0) : (index, index) -> (index)
// CHECK-NEXT:      func.return %2 : index
// CHECK-NEXT:    }) : () -> ()

"func.func"() <{function_type = (index) -> index, sym_name = "tripleSubSub3"}> ({
^bb0(%arg8: index):
  %8 = "arith.constant"() <{value = 17 : index}> : () -> index
  %9 = "arith.constant"() <{value = 42 : index}> : () -> index
  %10 = "arith.subi"(%arg8, %8) : (index, index) -> index
  %11 = "arith.subi"(%10, %9) : (index, index) -> index
  "func.return"(%11) : (index) -> ()
}) : () -> ()
// CHECK-NEXT:    "func.func"() <{sym_name = "tripleSubSub3", function_type = (index) -> index}> ({
// CHECK-NEXT:    ^bb0(%0: index):
// CHECK-NEXT:      %1 = "arith.constant"() <{value = 59 : index}> : () -> (index)
// CHECK-NEXT:      %2 = "arith.subi"(%0, %1) : (index, index) -> (index)
// CHECK-NEXT:      func.return %2 : index
// CHECK-NEXT:    }) : () -> ()

// CHECK-NEXT:  }
