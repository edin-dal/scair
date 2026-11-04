// RUN: scair-opt %s | mlir-opt --mlir-print-op-generic --allow-unregistered-dialect | scair-opt | filecheck %s

"test.op"() {space = #nvvm.memory_space<generic>} : () -> ()
"test.op"() {space = #nvvm.memory_space<global>} : () -> ()
"test.op"() {space = #nvvm.memory_space<shared>} : () -> ()
"test.op"() {space = #nvvm.memory_space<constant>} : () -> ()
"test.op"() {space = #nvvm.memory_space<local>} : () -> ()
"test.op"() {space = #nvvm.memory_space<tensor>} : () -> ()
"test.op"() {space = #nvvm.memory_space<shared_cluster>} : () -> ()

%0 = "test.op"() : () -> memref<4xf32, #nvvm.memory_space<global>>
%1 = "test.op"() : () -> memref<4xf32, #nvvm.memory_space<shared>>

// CHECK: "test.op"() {space = #nvvm.memory_space<generic>} : () -> ()
// CHECK: "test.op"() {space = #nvvm.memory_space<global>} : () -> ()
// CHECK: "test.op"() {space = #nvvm.memory_space<shared>} : () -> ()
// CHECK: "test.op"() {space = #nvvm.memory_space<constant>} : () -> ()
// CHECK: "test.op"() {space = #nvvm.memory_space<local>} : () -> ()
// CHECK: "test.op"() {space = #nvvm.memory_space<tensor>} : () -> ()
// CHECK: "test.op"() {space = #nvvm.memory_space<shared_cluster>} : () -> ()
// CHECK: %{{.*}} = "test.op"() : () -> memref<4xf32, #nvvm.memory_space<global>>
// CHECK: %{{.*}} = "test.op"() : () -> memref<4xf32, #nvvm.memory_space<shared>>
