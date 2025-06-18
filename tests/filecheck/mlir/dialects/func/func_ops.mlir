// RUN: scair-opt %s | scair-opt | filecheck %s

"func.func"() <{"sym_name" = "noarg_void", "function_type" = () -> ()}> ({
    "func.return"() : () -> ()
  }) : () -> ()
  "func.func"() <{"sym_name" = "call_void", "function_type" = () -> ()}> ({
    "func.call"() <{"callee" = @call_void}> : () -> ()
    "func.return"() : () -> ()
  }) : () -> ()
  "func.func"() <{"sym_name" = "call_void_attributes", "function_type" = () -> ()}> ({
    "func.call"() <{"callee" = @call_void_attributes}> {"hello" = "world"} : () -> ()
    "func.return"() : () -> ()
  }) : () -> ()
  "func.func"() <{"sym_name" = "arg_rec", "function_type" = (index) -> index}> ({
  ^0(%0 : index):
    %1 = "func.call"(%0) <{"callee" = @arg_rec}> : (index) -> index
    "func.return"(%1) : (index) -> ()
  }) : () -> ()
  "func.func"() <{"sym_name" = "arg_rec_block", "function_type" = (index) -> index}> ({
  ^0(%0 : index):
    %1 = "func.call"(%0) <{"callee" = @arg_rec_block}> : (index) -> index
    "func.return"(%1) : (index) -> ()
  }) : () -> ()
  "func.func"() <{"sym_name" = "external_fn", "function_type" = (i32) -> (i32, i32), "sym_visibility" = "private"}> ({
  }) : () -> ()
  "func.func"() <{"sym_name" = "multi_return_body", "function_type" = (i32) -> (i32, i32)}> ({
  ^0(%a : i32):
    "func.return"(%a, %a) : (i32, i32) -> ()
  }) : () -> ()

// CHECK:       builtin.module {
// CHECK-NEXT:    "func.func"() <{sym_name = "noarg_void", function_type = () -> ()}> ({
// CHECK-NEXT:      "func.return"() : () -> ()
// CHECK-NEXT:    }) : () -> ()
// CHECK-NEXT:    "func.func"() <{sym_name = "call_void", function_type = () -> ()}> ({
// CHECK-NEXT:      "func.call"() <{callee = @call_void}> : () -> ()
// CHECK-NEXT:      "func.return"() : () -> ()
// CHECK-NEXT:    }) : () -> ()
// CHECK-NEXT:    "func.func"() <{sym_name = "call_void_attributes", function_type = () -> ()}> ({
// CHECK-NEXT:      "func.call"() <{callee = @call_void_attributes}> {hello = "world"} : () -> ()
// CHECK-NEXT:      "func.return"() : () -> ()
// CHECK-NEXT:    }) : () -> ()
// CHECK-NEXT:    "func.func"() <{sym_name = "arg_rec", function_type = (index) -> index}> ({
// CHECK-NEXT:    ^bb0(%0: index):
// CHECK-NEXT:      %1 = "func.call"(%0) <{callee = @arg_rec}> : (index) -> (index)
// CHECK-NEXT:      "func.return"(%1) : (index) -> ()
// CHECK-NEXT:    }) : () -> ()
// CHECK-NEXT:    "func.func"() <{sym_name = "arg_rec_block", function_type = (index) -> index}> ({
// CHECK-NEXT:    ^bb0(%0: index):
// CHECK-NEXT:      %1 = "func.call"(%0) <{callee = @arg_rec_block}> : (index) -> (index)
// CHECK-NEXT:      "func.return"(%1) : (index) -> ()
// CHECK-NEXT:    }) : () -> ()
// CHECK-NEXT:    "func.func"() <{sym_name = "external_fn", function_type = (i32) -> (i32, i32), sym_visibility = "private"}> ({
// CHECK-NEXT:    }) : () -> ()
// CHECK-NEXT:    "func.func"() <{sym_name = "multi_return_body", function_type = (i32) -> (i32, i32)}> ({
// CHECK-NEXT:    ^bb0(%0: i32):
// CHECK-NEXT:      "func.return"(%0, %0) : (i32, i32) -> ()
// CHECK-NEXT:    }) : () -> ()
// CHECK-NEXT:  }