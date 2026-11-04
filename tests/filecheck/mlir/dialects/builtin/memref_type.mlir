// RUN: scair-opt %s | mlir-opt --allow-unregistered-dialect --mlir-print-op-generic | scair-opt | filecheck %s

"test.op"() {value = memref<f32, 1>} : () -> ()
"test.op"() {value = memref<3x?xf32, 2>} : () -> ()
"test.op"() {value = memref<*xf32>} : () -> ()
"test.op"() {value = memref<*xf32, 3>} : () -> ()
"test.op"() {value = memref<2xi32, "workgroup">} : () -> ()
%buffer = "memref.alloc"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<4xf32, 1>
memref.dealloc %buffer : memref<4xf32, 1>

// CHECK:      builtin.module {
// CHECK-NEXT:   "test.op"() {value = memref<f32, 1 : i64>} : () -> ()
// CHECK-NEXT:   "test.op"() {value = memref<3x?xf32, 2 : i64>} : () -> ()
// CHECK-NEXT:   "test.op"() {value = memref<*xf32>} : () -> ()
// CHECK-NEXT:   "test.op"() {value = memref<*xf32, 3 : i64>} : () -> ()
// CHECK-NEXT:   "test.op"() {value = memref<2xi32, "workgroup">} : () -> ()
// CHECK-NEXT:   %0 = "memref.alloc"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<4xf32, 1 : i64>
// CHECK-NEXT:   memref.dealloc %0 : memref<4xf32, 1 : i64>
// CHECK-NEXT: }
