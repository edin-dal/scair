// RUN: scair-opt %s | mlir-opt --mlir-print-op-generic | scair-opt | filecheck %s
// RUN: scair-opt %s | mlir-opt | filecheck %s --check-prefix=MLIR

func.func @pretty(%m0: memref<4x5xf32>, %m1: memref<2x3x4xf32>, %u0: memref<*xf32>, %s0: memref<?xf32, 5>) {
  %0 = memref.cast %m0 : memref<4x5xf32> to memref<?x?xf32>
  %1 = memref.cast %m0 : memref<4x5xf32> to memref<*xf32>
  %2 = memref.cast %u0 : memref<*xf32> to memref<4x?xf32>
  %3 = memref.memory_space_cast %s0 : memref<?xf32, 5> to memref<?xf32>
  %4 = memref.memory_space_cast %m0 : memref<4x5xf32> to memref<4x5xf32, 3>
  memref.copy %m0, %0 : memref<4x5xf32> to memref<?x?xf32>
  memref.copy %u0, %u0 : memref<*xf32> to memref<*xf32>
  func.return
}

// CHECK:       builtin.module {
// CHECK-NEXT:    func.func @pretty(%0: memref<4x5xf32>, %1: memref<2x3x4xf32>, %2: memref<*xf32>, %3: memref<?xf32, 5 : i64>) {
// CHECK-NEXT:      %4 = memref.cast %0 : memref<4x5xf32> to memref<?x?xf32>
// CHECK-NEXT:      %5 = memref.cast %0 : memref<4x5xf32> to memref<*xf32>
// CHECK-NEXT:      %6 = memref.cast %2 : memref<*xf32> to memref<4x?xf32>
// CHECK-NEXT:      %7 = memref.memory_space_cast %3 : memref<?xf32, 5 : i64> to memref<?xf32>
// CHECK-NEXT:      %8 = memref.memory_space_cast %0 : memref<4x5xf32> to memref<4x5xf32, 3 : i64>
// CHECK-NEXT:      memref.copy %0, %4 : memref<4x5xf32> to memref<?x?xf32>
// CHECK-NEXT:      memref.copy %2, %2 : memref<*xf32> to memref<*xf32>
// CHECK-NEXT:      func.return
// CHECK-NEXT:    }
// CHECK-NEXT:  }

// MLIR:       %cast = memref.cast %arg0 : memref<4x5xf32> to memref<?x?xf32>
// MLIR-NEXT:  %cast_0 = memref.cast %arg0 : memref<4x5xf32> to memref<*xf32>
// MLIR-NEXT:  %cast_1 = memref.cast %arg2 : memref<*xf32> to memref<4x?xf32>
// MLIR-NEXT:  %memspacecast = memref.memory_space_cast %arg3 : memref<?xf32, 5> to memref<?xf32>
// MLIR-NEXT:  %memspacecast_2 = memref.memory_space_cast %arg0 : memref<4x5xf32> to memref<4x5xf32, 3>
// MLIR-NEXT:  memref.copy %arg0, %cast : memref<4x5xf32> to memref<?x?xf32>
// MLIR-NEXT:  memref.copy %arg2, %arg2 : memref<*xf32> to memref<*xf32>
