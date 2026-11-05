// RUN: scair-opt %s | mlir-opt --allow-unregistered-dialect --mlir-print-op-generic | scair-opt | filecheck %s

func.func @allocation(%i: index, %j: index) {

  // alloca: static, dynamic sizes, alignment
  %a0 = "memref.alloca"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<8x64xf32>
  %a1 = "memref.alloca"(%i) <{operandSegmentSizes = array<i32: 1, 0>}> : (index) -> memref<8x?xf32>
  %a2 = "memref.alloca"(%i, %j) <{operandSegmentSizes = array<i32: 2, 0>}> : (index, index) -> memref<?x?xf32>
  %a3 = "memref.alloca"() <{alignment = 32 : i64, operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<4xi32>
  %a4 = "memref.alloca"(%i) <{alignment = 16 : i64, operandSegmentSizes = array<i32: 1, 0>}> : (index) -> memref<?xf64>

  // realloc: without and with dynamic size, alignment
  %r0 = "memref.alloc"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<64xf32>
  %r1 = "memref.realloc"(%r0) : (memref<64xf32>) -> memref<124xf32>
  %r2 = "memref.realloc"(%r1) <{alignment = 8 : i64}> : (memref<124xf32>) -> memref<64xf32>
  %r3 = "memref.realloc"(%r2, %i) : (memref<64xf32>, index) -> memref<?xf32>
  %r4 = "memref.realloc"(%r3, %j) <{alignment = 8 : i64}> : (memref<?xf32>, index) -> memref<?xf32>

  // assume_alignment
  %m = "memref.alloc"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<4x4xf32>
  %al = "memref.assume_alignment"(%m) <{alignment = 16 : i32}> : (memref<4x4xf32>) -> memref<4x4xf32>

  // atomic_rmw
  %v, %w = "test.op"() : () -> (f32, i32)
  %m1 = "memref.alloc"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<10xf32>
  %m2 = "memref.alloc"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<i32>
  %m3 = "memref.alloc"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<4x4xi32>
  %x0 = "memref.atomic_rmw"(%v, %m1, %i) <{kind = 0 : i64}> : (f32, memref<10xf32>, index) -> f32
  %x1 = "memref.atomic_rmw"(%w, %m2) <{kind = 1 : i64}> : (i32, memref<i32>) -> i32
  %x2 = "memref.atomic_rmw"(%w, %m3, %i, %j) <{kind = 13 : i64}> : (i32, memref<4x4xi32>, index, index) -> i32

  // copy
  %c0 = "memref.alloc"(%i) <{operandSegmentSizes = array<i32: 1, 0>}> : (index) -> memref<?xf32>
  "memref.copy"(%m1, %c0) : (memref<10xf32>, memref<?xf32>) -> ()
  %u = "memref.cast"(%m1) : (memref<10xf32>) -> memref<*xf32>
  "memref.copy"(%u, %u) : (memref<*xf32>, memref<*xf32>) -> ()

  // prefetch
  "memref.prefetch"(%m3, %i, %j) <{isDataCache = true, isWrite = false, localityHint = 3 : i32}> : (memref<4x4xi32>, index, index) -> ()
  "memref.prefetch"(%m1, %i) <{isDataCache = false, isWrite = true, localityHint = 0 : i32}> : (memref<10xf32>, index) -> ()
  "memref.prefetch"(%m2) <{isDataCache = true, isWrite = true, localityHint = 2 : i32}> : (memref<i32>) -> ()
  func.return
}

// CHECK:       builtin.module {
// CHECK-NEXT:    func.func @allocation(%0: index, %1: index) {
// CHECK-NEXT:      %2 = "memref.alloca"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<8x64xf32>
// CHECK-NEXT:      %3 = "memref.alloca"(%0) <{operandSegmentSizes = array<i32: 1, 0>}> : (index) -> memref<8x?xf32>
// CHECK-NEXT:      %4 = "memref.alloca"(%0, %1) <{operandSegmentSizes = array<i32: 2, 0>}> : (index, index) -> memref<?x?xf32>
// CHECK-NEXT:      %5 = "memref.alloca"() <{operandSegmentSizes = array<i32: 0, 0>, alignment = 32 : i64}> : () -> memref<4xi32>
// CHECK-NEXT:      %6 = "memref.alloca"(%0) <{operandSegmentSizes = array<i32: 1, 0>, alignment = 16 : i64}> : (index) -> memref<?xf64>
// CHECK-NEXT:      %7 = "memref.alloc"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<64xf32>
// CHECK-NEXT:      %8 = "memref.realloc"(%7) : (memref<64xf32>) -> memref<124xf32>
// CHECK-NEXT:      %9 = "memref.realloc"(%8) <{alignment = 8 : i64}> : (memref<124xf32>) -> memref<64xf32>
// CHECK-NEXT:      %10 = "memref.realloc"(%9, %0) : (memref<64xf32>, index) -> memref<?xf32>
// CHECK-NEXT:      %11 = "memref.realloc"(%10, %1) <{alignment = 8 : i64}> : (memref<?xf32>, index) -> memref<?xf32>
// CHECK-NEXT:      %12 = "memref.alloc"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<4x4xf32>
// CHECK-NEXT:      %13 = "memref.assume_alignment"(%12) <{alignment = 16 : i32}> : (memref<4x4xf32>) -> memref<4x4xf32>
// CHECK-NEXT:      %14, %15 = "test.op"() : () -> (f32, i32)
// CHECK-NEXT:      %16 = "memref.alloc"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<10xf32>
// CHECK-NEXT:      %17 = "memref.alloc"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<i32>
// CHECK-NEXT:      %18 = "memref.alloc"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<4x4xi32>
// CHECK-NEXT:      %19 = "memref.atomic_rmw"(%14, %16, %0) <{kind = 0 : i64}> : (f32, memref<10xf32>, index) -> f32
// CHECK-NEXT:      %20 = "memref.atomic_rmw"(%15, %17) <{kind = 1 : i64}> : (i32, memref<i32>) -> i32
// CHECK-NEXT:      %21 = "memref.atomic_rmw"(%15, %18, %0, %1) <{kind = 13 : i64}> : (i32, memref<4x4xi32>, index, index) -> i32
// CHECK-NEXT:      %22 = "memref.alloc"(%0) <{operandSegmentSizes = array<i32: 1, 0>}> : (index) -> memref<?xf32>
// CHECK-NEXT:      memref.copy %16, %22 : memref<10xf32> to memref<?xf32>
// CHECK-NEXT:      %23 = memref.cast %16 : memref<10xf32> to memref<*xf32>
// CHECK-NEXT:      memref.copy %23, %23 : memref<*xf32> to memref<*xf32>
// CHECK-NEXT:      "memref.prefetch"(%18, %0, %1) <{isWrite = false, localityHint = 3 : i32, isDataCache = true}> : (memref<4x4xi32>, index, index) -> ()
// CHECK-NEXT:      "memref.prefetch"(%16, %0) <{isWrite = true, localityHint = 0 : i32, isDataCache = false}> : (memref<10xf32>, index) -> ()
// CHECK-NEXT:      "memref.prefetch"(%17) <{isWrite = true, localityHint = 2 : i32, isDataCache = true}> : (memref<i32>) -> ()
// CHECK-NEXT:      func.return
// CHECK-NEXT:    }
// CHECK-NEXT:  }
