// RUN: scair-opt %s | mlir-opt --mlir-print-op-generic | scair-opt | filecheck %s

// Pretty `gpu.terminator` in a generic `gpu.launch`.

%sz = "test.op"() : () -> index

"gpu.launch"(%sz, %sz, %sz, %sz, %sz, %sz) <{operandSegmentSizes = array<i32: 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0>}> ({
^bb0(%0: index, %1: index, %2: index, %3: index, %4: index, %5: index, %6: index, %7: index, %8: index, %9: index, %10: index, %11: index):
  gpu.terminator
}) {workgroup_attributions = 0 : i64} : (index, index, index, index, index, index) -> ()

"gpu.launch"(%sz, %sz, %sz, %sz, %sz, %sz) <{operandSegmentSizes = array<i32: 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0>}> ({
^bb0(%0: index, %1: index, %2: index, %3: index, %4: index, %5: index, %6: index, %7: index, %8: index, %9: index, %10: index, %11: index):
  gpu.terminator {foo = 1 : i32}
}) {workgroup_attributions = 0 : i64} : (index, index, index, index, index, index) -> ()

// CHECK:       builtin.module {
// CHECK-NEXT:    %0 = "test.op"() : () -> index
// CHECK-NEXT:    "gpu.launch"(%0, %0, %0, %0, %0, %0) <{operandSegmentSizes = array<i32: 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0>}> ({
// CHECK-NEXT:    ^bb0(%1: index, %2: index, %3: index, %4: index, %5: index, %6: index, %7: index, %8: index, %9: index, %10: index, %11: index, %12: index):
// CHECK-NEXT:      gpu.terminator
// CHECK-NEXT:    }) {workgroup_attributions = 0 : i64} : (index, index, index, index, index, index) -> ()
// CHECK-NEXT:    "gpu.launch"(%0, %0, %0, %0, %0, %0) <{operandSegmentSizes = array<i32: 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0>}> ({
// CHECK-NEXT:    ^bb0(%1: index, %2: index, %3: index, %4: index, %5: index, %6: index, %7: index, %8: index, %9: index, %10: index, %11: index, %12: index):
// CHECK-NEXT:      gpu.terminator {foo = 1 : i32}
// CHECK-NEXT:    }) {workgroup_attributions = 0 : i64} : (index, index, index, index, index, index) -> ()
// CHECK-NEXT:  }
