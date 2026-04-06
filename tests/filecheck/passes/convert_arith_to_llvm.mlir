// RUN: scair-opt %s --split-input-file -p convert-arith-to-llvm | filecheck %s -DFILE=%s

builtin.module {
  func.func @plain(%a: i32, %b: i32, %x: f32, %y: f32) -> (i32, f32) {
    %c2 = "arith.constant"() <{value = 2 : i32}> : () -> i32
    %sum = "arith.addi"(%a, %b) : (i32, i32) -> i32
    %prod = "arith.muli"(%sum, %c2) : (i32, i32) -> i32
    %fsum = "arith.addf"(%x, %y) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
    %fprod = "arith.mulf"(%fsum, %x) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
    func.return %prod, %fprod : i32, f32
  }
}

// CHECK-LABEL: builtin.module {
// CHECK: func.func @plain(
// CHECK: "llvm.mlir.constant"() <{value = 2 : i32}> : () -> i32
// CHECK: "llvm.add"(%{{.*}}, %{{.*}}) : (i32, i32) -> i32
// CHECK: "llvm.mul"(%{{.*}}, %{{.*}}) : (i32, i32) -> i32
// CHECK: "llvm.fadd"(%{{.*}}, %{{.*}}) : (f32, f32) -> f32
// CHECK: "llvm.fmul"(%{{.*}}, %{{.*}}) : (f32, f32) -> f32
// CHECK-NOT: arith.addi
// CHECK-NOT: arith.muli
// CHECK-NOT: arith.addf
// CHECK-NOT: arith.mulf
