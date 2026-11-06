// RUN: scair-opt %s | mlir-opt --mlir-print-op-generic | scair-opt | filecheck %s

// `custom<AsyncDependencies>` formats are unsupported: both ops stay generic.
// `symbolOperands` is untested: ScaIR memref types cannot carry a layout map.

%sz = "test.op"() : () -> index
%t0 = "test.op"() : () -> !gpu.async.token

%m0 = "gpu.alloc"() <{operandSegmentSizes = array<i32: 0, 0, 0>}> : () -> memref<13xf32, 1>
%m1, %t1 = "gpu.alloc"(%t0) <{operandSegmentSizes = array<i32: 1, 0, 0>}> : (!gpu.async.token) -> (memref<13xf32, 1>, !gpu.async.token)
%m2 = "gpu.alloc"() <{hostShared, operandSegmentSizes = array<i32: 0, 0, 0>}> : () -> memref<13xf32, 1>
%m3, %t2 = "gpu.alloc"(%t0, %sz) <{hostShared, operandSegmentSizes = array<i32: 1, 1, 0>}> : (!gpu.async.token, index) -> (memref<64x?xf32, 1>, !gpu.async.token)
%m4 = "gpu.alloc"() <{operandSegmentSizes = array<i32: 0, 0, 0>}> : () -> memref<4xf32, #gpu.address_space<workgroup>>

"gpu.dealloc"(%m0) : (memref<13xf32, 1>) -> ()
%t3 = "gpu.dealloc"(%t1, %m1) : (!gpu.async.token, memref<13xf32, 1>) -> !gpu.async.token

// CHECK:       builtin.module {
// CHECK-NEXT:    %0 = "test.op"() : () -> index
// CHECK-NEXT:    %1 = "test.op"() : () -> !gpu.async.token
// CHECK-NEXT:    %2 = "gpu.alloc"() <{operandSegmentSizes = array<i32: 0, 0, 0>}> : () -> memref<13xf32, 1 : i64>
// CHECK-NEXT:    %3, %4 = "gpu.alloc"(%1) <{operandSegmentSizes = array<i32: 1, 0, 0>}> : (!gpu.async.token) -> (memref<13xf32, 1 : i64>, !gpu.async.token)
// CHECK-NEXT:    %5 = "gpu.alloc"() <{operandSegmentSizes = array<i32: 0, 0, 0>, hostShared}> : () -> memref<13xf32, 1 : i64>
// CHECK-NEXT:    %6, %7 = "gpu.alloc"(%1, %0) <{operandSegmentSizes = array<i32: 1, 1, 0>, hostShared}> : (!gpu.async.token, index) -> (memref<64x?xf32, 1 : i64>, !gpu.async.token)
// CHECK-NEXT:    %8 = "gpu.alloc"() <{operandSegmentSizes = array<i32: 0, 0, 0>}> : () -> memref<4xf32, #gpu.address_space<workgroup>>
// CHECK-NEXT:    "gpu.dealloc"(%2) : (memref<13xf32, 1 : i64>) -> ()
// CHECK-NEXT:    %9 = "gpu.dealloc"(%4, %3) : (!gpu.async.token, memref<13xf32, 1 : i64>) -> !gpu.async.token
// CHECK-NEXT:  }
