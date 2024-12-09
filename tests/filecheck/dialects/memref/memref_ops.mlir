// RUN: scair-opt %s | filecheck %s

%0, %1 = "test.op"() : () -> (index, index)
%m0 = "memref.alloc"() : () -> memref<f32>
%m1 = "memref.alloc"() : () -> memref<1xf32>
%m2 = "memref.alloc"(%0) : (index) -> memref<?xf32>
%m3 = "memref.alloc"(%1) : (index) -> memref<3x?x5xf32>
%m4 = "memref.alloc"(%0, %1) : (index, index) -> memref<?x5x?xf32>

%2 = "memref.load"(%m0) : (memref<f32>) -> f32
"memref.store"(%2, %m0) : (f32, memref<f32>) -> ()
%3 = "memref.load"(%m1, %0) : (memref<1xf32>, index) -> f32
"memref.store"(%3, %m1, %1) : (f32, memref<1xf32>, index) -> ()
%4 = "memref.load"(%m2, %0, %1) : (memref<?xf32>, index, index) -> f32
"memref.store"(%4, %m2, %1, %0) : (f32, memref<?xf32>, index, index) -> ()
%5 = "memref.load"(%m3, %0, %1, %0) : (memref<3x?x5xf32>, index, index, index) -> f32
"memref.store"(%5, %m3, %1, %0, %1) : (f32, memref<3x?x5xf32>, index, index, index) -> ()
%6 = "memref.load"(%m4, %0, %1, %0) : (memref<?x5x?xf32>, index, index, index) -> f32
"memref.store"(%6, %m4, %1, %0, %1) : (f32, memref<?x5x?xf32>, index, index, index) -> ()


"memref.dealloc"(%m0) : (memref<f32>) -> ()
"memref.dealloc"(%m1) : (memref<1xf32>) -> ()
"memref.dealloc"(%m2) : (memref<?xf32>) -> ()
"memref.dealloc"(%m3) : (memref<3x?x5xf32>) -> ()
"memref.dealloc"(%m4) : (memref<?x5x?xf32>) -> ()

// CHECK:       builtin.module {
// CHECK-NEXT:  ^bb0():
// CHECK-NEXT:    %0, %1 = "test.op"() : () -> (index, index)
// CHECK-NEXT:    %2 = "memref.alloc"() : () -> (memref<f32>)
// CHECK-NEXT:    %3 = "memref.alloc"() : () -> (memref<1xf32>)
// CHECK-NEXT:    %4 = "memref.alloc"(%0) : (index) -> (memref<?xf32>)
// CHECK-NEXT:    %5 = "memref.alloc"(%1) : (index) -> (memref<3x?x5xf32>)
// CHECK-NEXT:    %6 = "memref.alloc"(%0, %1) : (index, index) -> (memref<?x5x?xf32>)
// CHECK-NEXT:    %7 = "memref.load"(%2) : (memref<f32>) -> (f32)
// CHECK-NEXT:    "memref.store"(%7, %2) : (f32, memref<f32>) -> ()
// CHECK-NEXT:    %8 = "memref.load"(%3, %0) : (memref<1xf32>, index) -> (f32)
// CHECK-NEXT:    "memref.store"(%8, %3, %1) : (f32, memref<1xf32>, index) -> ()
// CHECK-NEXT:    %9 = "memref.load"(%4, %0, %1) : (memref<?xf32>, index, index) -> (f32)
// CHECK-NEXT:    "memref.store"(%9, %4, %1, %0) : (f32, memref<?xf32>, index, index) -> ()
// CHECK-NEXT:    %10 = "memref.load"(%5, %0, %1, %0) : (memref<3x?x5xf32>, index, index, index) -> (f32)
// CHECK-NEXT:    "memref.store"(%10, %5, %1, %0, %1) : (f32, memref<3x?x5xf32>, index, index, index) -> ()
// CHECK-NEXT:    %11 = "memref.load"(%6, %0, %1, %0) : (memref<?x5x?xf32>, index, index, index) -> (f32)
// CHECK-NEXT:    "memref.store"(%11, %6, %1, %0, %1) : (f32, memref<?x5x?xf32>, index, index, index) -> ()
// CHECK-NEXT:    "memref.dealloc"(%2) : (memref<f32>) -> ()
// CHECK-NEXT:    "memref.dealloc"(%3) : (memref<1xf32>) -> ()
// CHECK-NEXT:    "memref.dealloc"(%4) : (memref<?xf32>) -> ()
// CHECK-NEXT:    "memref.dealloc"(%5) : (memref<3x?x5xf32>) -> ()
// CHECK-NEXT:    "memref.dealloc"(%6) : (memref<?x5x?xf32>) -> ()
// CHECK-NEXT:  }