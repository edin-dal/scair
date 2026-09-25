// RUN: scair-opt %s --verify-diagnostics --split-input-file | filecheck %s

%arg = "test.op"() : () -> index
"scf.index_switch"(%arg) <{cases = array<i64: 0, 1>}> ({
  scf.yield
}, {
  scf.yield
}) : (index) -> ()

// CHECK:       "scf.index_switch"(%0) <{cases = array<i64: 0, 1>}> ({
// CHECK:       }) : (index) -> ()
// CHECK-NEXT:  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
// CHECK-NEXT:  > scf.index_switch: has 1 case regions but 2 case values

// -----

%arg = "test.op"() : () -> index
"scf.index_switch"(%arg) <{cases = array<i64: 0>}> ({
  scf.yield
}, {
  scf.yield
}, {
  scf.yield
}) : (index) -> ()

// CHECK:       "scf.index_switch"(%0) <{cases = array<i64: 0>}> ({
// CHECK:       }) : (index) -> ()
// CHECK-NEXT:  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
// CHECK-NEXT:  > scf.index_switch: has 2 case regions but 1 case values
