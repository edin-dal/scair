// RUN: scair-opt %s | mlir-opt --mlir-print-op-generic | scair-opt | filecheck %s

// `gpu.launch` has custom syntax upstream: it stays generic.

"builtin.module"() ({
  "func.func"() <{function_type = (index, i32, memref<?xf32, 1>) -> (), sym_name = "launch"}> ({
  ^bb0(%arg0: index, %arg1: i32, %arg2: memref<?xf32, 1>):
    "gpu.launch"(%arg0, %arg0, %arg0, %arg0, %arg0, %arg0) <{operandSegmentSizes = array<i32: 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0>}> ({
    ^bb0(%arg104: index, %arg105: index, %arg106: index, %arg107: index, %arg108: index, %arg109: index, %arg110: index, %arg111: index, %arg112: index, %arg113: index, %arg114: index, %arg115: index):
      "gpu.terminator"() : () -> ()
    }) {workgroup_attributions = 0 : i64} : (index, index, index, index, index, index) -> ()
    %0 = "test.op"() : () -> !gpu.async.token
    %1 = "gpu.launch"(%0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0) <{operandSegmentSizes = array<i32: 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0>}> ({
    ^bb0(%arg92: index, %arg93: index, %arg94: index, %arg95: index, %arg96: index, %arg97: index, %arg98: index, %arg99: index, %arg100: index, %arg101: index, %arg102: index, %arg103: index):
      "gpu.terminator"() : () -> ()
    }) {workgroup_attributions = 0 : i64} : (!gpu.async.token, index, index, index, index, index, index) -> !gpu.async.token
    %2 = "gpu.launch"(%arg0, %arg0, %arg0, %arg0, %arg0, %arg0) <{operandSegmentSizes = array<i32: 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0>}> ({
    ^bb0(%arg80: index, %arg81: index, %arg82: index, %arg83: index, %arg84: index, %arg85: index, %arg86: index, %arg87: index, %arg88: index, %arg89: index, %arg90: index, %arg91: index):
      "gpu.terminator"() : () -> ()
    }) {workgroup_attributions = 0 : i64} : (index, index, index, index, index, index) -> !gpu.async.token
    "gpu.launch"(%arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0) <{operandSegmentSizes = array<i32: 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0>}> ({
    ^bb0(%arg62: index, %arg63: index, %arg64: index, %arg65: index, %arg66: index, %arg67: index, %arg68: index, %arg69: index, %arg70: index, %arg71: index, %arg72: index, %arg73: index, %arg74: index, %arg75: index, %arg76: index, %arg77: index, %arg78: index, %arg79: index):
      "gpu.terminator"() : () -> ()
    }) {workgroup_attributions = 0 : i64} : (index, index, index, index, index, index, index, index, index) -> ()
    "gpu.launch"(%arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg1) <{operandSegmentSizes = array<i32: 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1>}> ({
    ^bb0(%arg50: index, %arg51: index, %arg52: index, %arg53: index, %arg54: index, %arg55: index, %arg56: index, %arg57: index, %arg58: index, %arg59: index, %arg60: index, %arg61: index):
      "gpu.terminator"() : () -> ()
    }) {workgroup_attributions = 0 : i64} : (index, index, index, index, index, index, i32) -> ()
    "gpu.launch"(%arg0, %arg0, %arg0, %arg0, %arg0, %arg0) <{function = @test_kernel_func, module = @test_module, operandSegmentSizes = array<i32: 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0>}> ({
    ^bb0(%arg38: index, %arg39: index, %arg40: index, %arg41: index, %arg42: index, %arg43: index, %arg44: index, %arg45: index, %arg46: index, %arg47: index, %arg48: index, %arg49: index):
      "gpu.terminator"() : () -> ()
    }) {workgroup_attributions = 0 : i64} : (index, index, index, index, index, index) -> ()
    "gpu.launch"(%arg0, %arg0, %arg0, %arg0, %arg0, %arg0) <{operandSegmentSizes = array<i32: 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0>}> ({
    ^bb0(%arg23: index, %arg24: index, %arg25: index, %arg26: index, %arg27: index, %arg28: index, %arg29: index, %arg30: index, %arg31: index, %arg32: index, %arg33: index, %arg34: index, %arg35: memref<42xf32, 3>, %arg36: memref<2xf32, 3>, %arg37: memref<2xf32, 5>):
      "test.op"(%arg35, %arg36) : (memref<42xf32, 3>, memref<2xf32, 3>) -> ()
      "gpu.terminator"() : () -> ()
    }) {workgroup_attributions = 2 : i64} : (index, index, index, index, index, index) -> ()
    %3 = "gpu.launch"(%0, %1, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg1) <{function = @test_kernel_func, module = @test_module, operandSegmentSizes = array<i32: 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1>}> ({
    ^bb0(%arg3: index, %arg4: index, %arg5: index, %arg6: index, %arg7: index, %arg8: index, %arg9: index, %arg10: index, %arg11: index, %arg12: index, %arg13: index, %arg14: index, %arg15: index, %arg16: index, %arg17: index, %arg18: index, %arg19: index, %arg20: index, %arg21: memref<42xf32, 3>, %arg22: memref<2xf32, 5>):
      "gpu.terminator"() : () -> ()
    }) {workgroup_attributions = 1 : i64} : (!gpu.async.token, !gpu.async.token, index, index, index, index, index, index, index, index, index, i32) -> !gpu.async.token
    "func.return"() : () -> ()
  }) : () -> ()
}) : () -> ()

// CHECK:       builtin.module {
// CHECK-NEXT:    func.func @launch(%0: index, %1: i32, %2: memref<?xf32, 1 : i64>) {
// CHECK-NEXT:      "gpu.launch"(%0, %0, %0, %0, %0, %0) <{operandSegmentSizes = array<i32: 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0>}> ({
// CHECK-NEXT:      ^bb0(%3: index, %4: index, %5: index, %6: index, %7: index, %8: index, %9: index, %10: index, %11: index, %12: index, %13: index, %14: index):
// CHECK-NEXT:        gpu.terminator
// CHECK-NEXT:      }) {workgroup_attributions = 0 : i64} : (index, index, index, index, index, index) -> ()
// CHECK-NEXT:      %3 = "test.op"() : () -> !gpu.async.token
// CHECK-NEXT:      %4 = "gpu.launch"(%3, %0, %0, %0, %0, %0, %0) <{operandSegmentSizes = array<i32: 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0>}> ({
// CHECK-NEXT:      ^bb0(%5: index, %6: index, %7: index, %8: index, %9: index, %10: index, %11: index, %12: index, %13: index, %14: index, %15: index, %16: index):
// CHECK-NEXT:        gpu.terminator
// CHECK-NEXT:      }) {workgroup_attributions = 0 : i64} : (!gpu.async.token, index, index, index, index, index, index) -> !gpu.async.token
// CHECK-NEXT:      %5 = "gpu.launch"(%0, %0, %0, %0, %0, %0) <{operandSegmentSizes = array<i32: 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0>}> ({
// CHECK-NEXT:      ^bb0(%6: index, %7: index, %8: index, %9: index, %10: index, %11: index, %12: index, %13: index, %14: index, %15: index, %16: index, %17: index):
// CHECK-NEXT:        gpu.terminator
// CHECK-NEXT:      }) {workgroup_attributions = 0 : i64} : (index, index, index, index, index, index) -> !gpu.async.token
// CHECK-NEXT:      "gpu.launch"(%0, %0, %0, %0, %0, %0, %0, %0, %0) <{operandSegmentSizes = array<i32: 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0>}> ({
// CHECK-NEXT:      ^bb0(%6: index, %7: index, %8: index, %9: index, %10: index, %11: index, %12: index, %13: index, %14: index, %15: index, %16: index, %17: index, %18: index, %19: index, %20: index, %21: index, %22: index, %23: index):
// CHECK-NEXT:        gpu.terminator
// CHECK-NEXT:      }) {workgroup_attributions = 0 : i64} : (index, index, index, index, index, index, index, index, index) -> ()
// CHECK-NEXT:      "gpu.launch"(%0, %0, %0, %0, %0, %0, %1) <{operandSegmentSizes = array<i32: 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1>}> ({
// CHECK-NEXT:      ^bb0(%6: index, %7: index, %8: index, %9: index, %10: index, %11: index, %12: index, %13: index, %14: index, %15: index, %16: index, %17: index):
// CHECK-NEXT:        gpu.terminator
// CHECK-NEXT:      }) {workgroup_attributions = 0 : i64} : (index, index, index, index, index, index, i32) -> ()
// CHECK-NEXT:      "gpu.launch"(%0, %0, %0, %0, %0, %0) <{operandSegmentSizes = array<i32: 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0>, module = @test_module, function = @test_kernel_func}> ({
// CHECK-NEXT:      ^bb0(%6: index, %7: index, %8: index, %9: index, %10: index, %11: index, %12: index, %13: index, %14: index, %15: index, %16: index, %17: index):
// CHECK-NEXT:        gpu.terminator
// CHECK-NEXT:      }) {workgroup_attributions = 0 : i64} : (index, index, index, index, index, index) -> ()
// CHECK-NEXT:      "gpu.launch"(%0, %0, %0, %0, %0, %0) <{operandSegmentSizes = array<i32: 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0>}> ({
// CHECK-NEXT:      ^bb0(%6: index, %7: index, %8: index, %9: index, %10: index, %11: index, %12: index, %13: index, %14: index, %15: index, %16: index, %17: index, %18: memref<42xf32, 3 : i64>, %19: memref<2xf32, 3 : i64>, %20: memref<2xf32, 5 : i64>):
// CHECK-NEXT:        "test.op"(%18, %19) : (memref<42xf32, 3 : i64>, memref<2xf32, 3 : i64>) -> ()
// CHECK-NEXT:        gpu.terminator
// CHECK-NEXT:      }) {workgroup_attributions = 2 : i64} : (index, index, index, index, index, index) -> ()
// CHECK-NEXT:      %6 = "gpu.launch"(%3, %4, %0, %0, %0, %0, %0, %0, %0, %0, %0, %1) <{operandSegmentSizes = array<i32: 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1>, module = @test_module, function = @test_kernel_func}> ({
// CHECK-NEXT:      ^bb0(%7: index, %8: index, %9: index, %10: index, %11: index, %12: index, %13: index, %14: index, %15: index, %16: index, %17: index, %18: index, %19: index, %20: index, %21: index, %22: index, %23: index, %24: index, %25: memref<42xf32, 3 : i64>, %26: memref<2xf32, 5 : i64>):
// CHECK-NEXT:        gpu.terminator
// CHECK-NEXT:      }) {workgroup_attributions = 1 : i64} : (!gpu.async.token, !gpu.async.token, index, index, index, index, index, index, index, index, index, i32) -> !gpu.async.token
// CHECK-NEXT:      func.return
// CHECK-NEXT:    }
// CHECK-NEXT:  }
