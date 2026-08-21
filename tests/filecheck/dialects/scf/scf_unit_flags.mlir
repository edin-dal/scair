// RUN: scair-opt %s | filecheck %s
// RUN: scair-opt %s | scair-opt | filecheck %s

%lb, %ub, %step = "test.op"() : () -> (index, index, index)

"scf.for"(%lb, %ub, %step) <{"unsignedCmp"}> ({
^bb0(%iv : index):
  "scf.yield"() : () -> ()
}) : (index, index, index) -> ()

"scf.execute_region"() <{"no_inline"}> ({
  "scf.yield"() : () -> ()
}) : () -> ()

// CHECK:       builtin.module {
// CHECK-NEXT:    %0, %1, %2 = "test.op"() : () -> (index, index, index)
// CHECK-NEXT:    "scf.for"(%0, %1, %2) <{unsignedCmp}> ({
// CHECK:         "scf.execute_region"() <{no_inline}> ({
