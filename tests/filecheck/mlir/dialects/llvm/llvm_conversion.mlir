// RUN: scair-opt %s -p ConvertFuncToLLVM | mlir-opt --mlir-print-op-generic | scair-opt | filecheck %s

// LLVM functions have one result; multiple func results use a struct.
func.func @pair(%i: i32, %f: f32) -> (i32, f32) {
  func.return %i, %f : i32, f32
}
func.func @caller(%i: i32, %f: f32) -> (i32, f32) {
  %a, %b = "func.call"(%i, %f) <{callee = @pair}> : (i32, f32) -> (i32, f32)
  func.return %a, %b : i32, f32
}

// CHECK: "llvm.func"()
// CHECK-SAME: sym_name = "pair"
// CHECK-SAME: llvm.struct<(i32, f32)>
// CHECK: "llvm.insertvalue"
// CHECK: "llvm.func"()
// CHECK-SAME: sym_name = "caller"
// CHECK: "llvm.call"
// CHECK-SAME: callee = @pair
// CHECK-SAME: -> !llvm.struct<(i32, f32)>
// CHECK: "llvm.extractvalue"
// CHECK: "llvm.extractvalue"
