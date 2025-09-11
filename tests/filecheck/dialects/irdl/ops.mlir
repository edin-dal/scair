// RUN: scair-opt %s | filecheck %s

"builtin.module"() ({
  "irdl.dialect"() <{sym_name = "cmath"}> ({
    "irdl.type"() <{sym_name = "complex"}> ({
        %2 = "irdl.any"() : () -> !irdl.attribute<>
        "irdl.parameters"(%2) <{names = ["elem"]}> : (!irdl.attribute<>) -> ()
    }) : () -> ()
    "irdl.operation"() <{sym_name = "norm"}> ({
      %3 = "irdl.any"() : () -> !irdl.attribute<>
      "irdl.operands"(%3) <{names = ["in"]}> : (!irdl.attribute<>) -> ()
      "irdl.results"(%3) <{names = ["out"]}> : (!irdl.attribute<>) -> ()
    }) : () -> ()
    "irdl.operation"() <{sym_name = "mul"}> ({
      %8 = "irdl.any"() : () -> !irdl.attribute<>
      "irdl.operands"(%8, %8) <{names = ["lhs", "rhs"]}> : (!irdl.attribute<>, !irdl.attribute<>) -> ()
      "irdl.results"(%8) <{names = ["res"]}> : (!irdl.attribute<>) -> ()
    }) : () -> ()
  }) : () -> ()
}) : () -> ()

// CHECK:       builtin.module {
// CHECK-NEXT:    "irdl.dialect"() <{sym_name = "cmath"}> ({
// CHECK-NEXT:      "irdl.type"() <{sym_name = "complex"}> ({
// CHECK-NEXT:        %0 = "irdl.any"() : () -> (#irdl.attribute)
// CHECK-NEXT:        "irdl.parameters"(%0) <{names = ["elem"]}> : (#irdl.attribute) -> ()
// CHECK-NEXT:      }) : () -> ()
// CHECK-NEXT:      "irdl.operation"() <{sym_name = "norm"}> ({
// CHECK-NEXT:        %0 = "irdl.any"() : () -> (#irdl.attribute)
// CHECK-NEXT:        "irdl.operands"(%0) <{names = ["in"]}> : (#irdl.attribute) -> ()
// CHECK-NEXT:        "irdl.results"(%0) <{names = ["out"]}> : (#irdl.attribute) -> ()
// CHECK-NEXT:      }) : () -> ()
// CHECK-NEXT:      "irdl.operation"() <{sym_name = "mul"}> ({
// CHECK-NEXT:        %0 = "irdl.any"() : () -> (#irdl.attribute)
// CHECK-NEXT:        "irdl.operands"(%0, %0) <{names = ["lhs", "rhs"]}> : (#irdl.attribute, #irdl.attribute) -> ()
// CHECK-NEXT:        "irdl.results"(%0) <{names = ["res"]}> : (#irdl.attribute) -> ()
// CHECK-NEXT:      }) : () -> ()
// CHECK-NEXT:    }) : () -> ()
// CHECK-NEXT:  }
