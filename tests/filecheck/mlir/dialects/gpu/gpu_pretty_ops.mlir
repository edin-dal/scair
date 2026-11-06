// RUN: scair-opt %s | mlir-opt --mlir-print-op-generic | scair-opt | filecheck %s
// RUN: scair-opt %s | mlir-opt | sed 's/^module/builtin.module/' | scair-opt | filecheck %s

gpu.barrier
gpu.barrier {foo = 1 : i32}

// CHECK:       builtin.module {
// CHECK-NEXT:    gpu.barrier
// CHECK-NEXT:    gpu.barrier {foo = 1 : i32}
// CHECK-NEXT:  }
