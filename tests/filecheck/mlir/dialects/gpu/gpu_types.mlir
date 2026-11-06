// RUN: scair-opt %s | mlir-opt --mlir-print-op-generic | scair-opt | filecheck %s

"test.op"() {token = !gpu.async.token} : () -> ()
"test.op"() {space = #gpu.address_space<global>} : () -> ()
"test.op"() {space = #gpu.address_space<workgroup>} : () -> ()
"test.op"() {space = #gpu.address_space<private>} : () -> ()
%0 = "test.op"() : () -> memref<4xf32, #gpu.address_space<workgroup>>
%1 = "test.op"() : () -> !gpu.async.token

// CHECK:       builtin.module {
// CHECK-NEXT:    "test.op"() {token = !gpu.async.token} : () -> ()
// CHECK-NEXT:    "test.op"() {space = #gpu.address_space<global>} : () -> ()
// CHECK-NEXT:    "test.op"() {space = #gpu.address_space<workgroup>} : () -> ()
// CHECK-NEXT:    "test.op"() {space = #gpu.address_space<private>} : () -> ()
// CHECK-NEXT:    %0 = "test.op"() : () -> memref<4xf32, #gpu.address_space<workgroup>>
// CHECK-NEXT:    %1 = "test.op"() : () -> !gpu.async.token
// CHECK-NEXT:  }
