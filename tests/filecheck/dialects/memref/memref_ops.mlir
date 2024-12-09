// RUN: scair-opt %s | filecheck %s

%0, %1 = "test.op"() : () -> (index, index)
%m0 = "memref.alloc"() : () -> memref<f32>
%m1 = "memref.alloc"() : () -> memref<1xf32>
%m2 = "memref.alloc"(%0) : (index) -> memref<?xf32>
%m3 = "memref.alloc"(%1) : (index) -> memref<3x?x5xf32>
%m4 = "memref.alloc"(%0, %1) : (index, index) -> memref<?x5x?xf32>
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
// CHECK-NEXT:    "memref.dealloc"(%2) : (memref<f32>) -> ()
// CHECK-NEXT:    "memref.dealloc"(%3) : (memref<1xf32>) -> ()
// CHECK-NEXT:    "memref.dealloc"(%4) : (memref<?xf32>) -> ()
// CHECK-NEXT:    "memref.dealloc"(%5) : (memref<3x?x5xf32>) -> ()
// CHECK-NEXT:    "memref.dealloc"(%6) : (memref<?x5x?xf32>) -> ()
// CHECK-NEXT:  }