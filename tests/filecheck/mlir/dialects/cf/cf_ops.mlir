// RUN: scair-opt %s | mlir-opt --mlir-print-op-generic | scair-opt | filecheck %s

"func.func"() <{function_type = () -> (), sym_name = "assert"}> ({
^bb0:
  %b = "arith.constant"() <{value = true}> : () -> i1
  "cf.assert"(%b) <{msg = "Expected b to be true"}> : (i1) -> ()
  "cf.assert"(%b) <{msg = "with attrs"}> {foo = 1 : i32} : (i1) -> ()
  "func.return"() : () -> ()
}) : () -> ()

"func.func"() <{function_type = () -> (), sym_name = "br"}> ({
^bb0:
  %a = "arith.constant"() <{value = 1 : i32}> : () -> i32
  %f = "arith.constant"() <{value = 1.0 : f32}> : () -> f32
  "cf.br"()[^bb1] : () -> ()
^bb1:
  "cf.br"(%a, %f)[^bb2] : (i32, f32) -> ()
^bb2(%x: i32, %y: f32):
  "func.return"() : () -> ()
}) : () -> ()

"func.func"() <{function_type = () -> (), sym_name = "cond_br"}> ({
^bb0:
  %c = "arith.constant"() <{value = true}> : () -> i1
  %a = "arith.constant"() <{value = 1 : i32}> : () -> i32
  %b = "arith.constant"() <{value = 2 : i32}> : () -> i32
  "cf.cond_br"(%c)[^bb1, ^bb2] <{operandSegmentSizes = array<i32: 1, 0, 0>}> : (i1) -> ()
^bb1:
  "cf.cond_br"(%c, %a)[^bb3, ^bb2] <{operandSegmentSizes = array<i32: 1, 1, 0>}> : (i1, i32) -> ()
^bb2:
  "cf.cond_br"(%c, %a, %b)[^bb3, ^bb3] <{operandSegmentSizes = array<i32: 1, 1, 1>}> : (i1, i32, i32) -> ()
^bb3(%x: i32):
  "cf.cond_br"(%c)[^bb1, ^bb2] <{branch_weights = array<i32: 1, 2>, operandSegmentSizes = array<i32: 1, 0, 0>}> : (i1) -> ()
}) : () -> ()

"func.func"() <{function_type = () -> (), sym_name = "switch"}> ({
^bb0:
  %flag = "arith.constant"() <{value = 0 : i32}> : () -> i32
  %a = "arith.constant"() <{value = 1 : i32}> : () -> i32
  %b = "arith.constant"() <{value = 2 : i32}> : () -> i32
  "cf.switch"(%flag)[^bb1] <{case_operand_segments = array<i32>, operandSegmentSizes = array<i32: 1, 0, 0>}> : (i32) -> ()
^bb1:
  "cf.switch"(%flag, %a, %b)[^bb2, ^bb2, ^bb3] <{case_operand_segments = array<i32: 1, 0>, case_values = dense<[42, 43]> : vector<2xi32>, operandSegmentSizes = array<i32: 1, 1, 1>}> : (i32, i32, i32) -> ()
^bb2(%x: i32):
  "cf.switch"(%flag, %a, %b, %b, %a)[^bb3, ^bb4, ^bb4] <{case_operand_segments = array<i32: 2, 2>, case_values = dense<[-1, 7]> : vector<2xi32>, operandSegmentSizes = array<i32: 1, 0, 4>}> : (i32, i32, i32, i32, i32) -> ()
^bb3:
  "func.return"() : () -> ()
^bb4(%y: i32, %z: i32):
  "func.return"() : () -> ()
}) : () -> ()

"func.func"() <{function_type = () -> (), sym_name = "switch_i64"}> ({
^bb0:
  %flag = "arith.constant"() <{value = 0 : i64}> : () -> i64
  "cf.switch"(%flag)[^bb1, ^bb1] <{case_operand_segments = array<i32: 0>, case_values = dense<1> : vector<1xi64>, operandSegmentSizes = array<i32: 1, 0, 0>}> : (i64) -> ()
^bb1:
  "func.return"() : () -> ()
}) : () -> ()

// CHECK:       builtin.module {
// CHECK-NEXT:    func.func @assert() {
// CHECK-NEXT:      %0 = "arith.constant"() <{value = true}> : () -> i1
// CHECK-NEXT:      "cf.assert"(%0) <{msg = "Expected b to be true"}> : (i1) -> ()
// CHECK-NEXT:      "cf.assert"(%0) <{msg = "with attrs"}> {foo = 1 : i32} : (i1) -> ()
// CHECK-NEXT:      func.return
// CHECK-NEXT:    }
// CHECK-NEXT:    func.func @br() {
// CHECK-NEXT:      %0 = "arith.constant"() <{value = 1 : i32}> : () -> i32
// CHECK-NEXT:      %1 = "arith.constant"() <{value = 1.0 : f32}> : () -> f32
// CHECK-NEXT:      "cf.br"()[^bb0] : () -> ()
// CHECK-NEXT:    ^bb0:
// CHECK-NEXT:      "cf.br"(%0, %1)[^bb1] : (i32, f32) -> ()
// CHECK-NEXT:    ^bb1(%2: i32, %3: f32):
// CHECK-NEXT:      func.return
// CHECK-NEXT:    }
// CHECK-NEXT:    func.func @cond_br() {
// CHECK-NEXT:      %0 = "arith.constant"() <{value = true}> : () -> i1
// CHECK-NEXT:      %1 = "arith.constant"() <{value = 1 : i32}> : () -> i32
// CHECK-NEXT:      %2 = "arith.constant"() <{value = 2 : i32}> : () -> i32
// CHECK-NEXT:      "cf.cond_br"(%0)[^bb0, ^bb1] <{operandSegmentSizes = array<i32: 1, 0, 0>}> : (i1) -> ()
// CHECK-NEXT:    ^bb0:
// CHECK-NEXT:      "cf.cond_br"(%0, %1)[^bb2, ^bb1] <{operandSegmentSizes = array<i32: 1, 1, 0>}> : (i1, i32) -> ()
// CHECK-NEXT:    ^bb1:
// CHECK-NEXT:      "cf.cond_br"(%0, %1, %2)[^bb2, ^bb2] <{operandSegmentSizes = array<i32: 1, 1, 1>}> : (i1, i32, i32) -> ()
// CHECK-NEXT:    ^bb2(%3: i32):
// CHECK-NEXT:      "cf.cond_br"(%0)[^bb0, ^bb1] <{operandSegmentSizes = array<i32: 1, 0, 0>, branch_weights = array<i32: 1, 2>}> : (i1) -> ()
// CHECK-NEXT:    }
// CHECK-NEXT:    func.func @switch() {
// CHECK-NEXT:      %0 = "arith.constant"() <{value = 0 : i32}> : () -> i32
// CHECK-NEXT:      %1 = "arith.constant"() <{value = 1 : i32}> : () -> i32
// CHECK-NEXT:      %2 = "arith.constant"() <{value = 2 : i32}> : () -> i32
// CHECK-NEXT:      "cf.switch"(%0)[^bb0] <{case_operand_segments = array<i32>, operandSegmentSizes = array<i32: 1, 0, 0>}> : (i32) -> ()
// CHECK-NEXT:    ^bb0:
// CHECK-NEXT:      "cf.switch"(%0, %1, %2)[^bb1, ^bb1, ^bb2] <{case_operand_segments = array<i32: 1, 0>, operandSegmentSizes = array<i32: 1, 1, 1>, case_values = dense<[42, 43]> : vector<2xi32>}> : (i32, i32, i32) -> ()
// CHECK-NEXT:    ^bb1(%3: i32):
// CHECK-NEXT:      "cf.switch"(%0, %1, %2, %2, %1)[^bb2, ^bb3, ^bb3] <{case_operand_segments = array<i32: 2, 2>, operandSegmentSizes = array<i32: 1, 0, 4>, case_values = dense<[-1, 7]> : vector<2xi32>}> : (i32, i32, i32, i32, i32) -> ()
// CHECK-NEXT:    ^bb2:
// CHECK-NEXT:      func.return
// CHECK-NEXT:    ^bb3(%4: i32, %5: i32):
// CHECK-NEXT:      func.return
// CHECK-NEXT:    }
// CHECK-NEXT:    func.func @switch_i64() {
// CHECK-NEXT:      %0 = "arith.constant"() <{value = 0 : i64}> : () -> i64
// CHECK-NEXT:      "cf.switch"(%0)[^bb0, ^bb0] <{case_operand_segments = array<i32: 0>, operandSegmentSizes = array<i32: 1, 0, 0>, case_values = dense<1> : vector<1xi64>}> : (i64) -> ()
// CHECK-NEXT:    ^bb0:
// CHECK-NEXT:      func.return
// CHECK-NEXT:    }
// CHECK-NEXT:  }
