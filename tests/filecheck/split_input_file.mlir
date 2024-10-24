// RUN: scair-opt %s | filecheck %s
// RUN: scair-opt --split_input_file %s | filecheck %s --check-prefix=SPLIT

"test.op"() : () -> ()
// -----
"test.op"() : () -> ()

// CHECK:       builtin.module {
// CHECK-NEXT:  ^bb0():
// CHECK-NEXT:    "test.op"() : () -> ()
// CHECK-NEXT:    "test.op"() : () -> ()
// CHECK-NEXT:  }

// SPLIT:       builtin.module {
// SPLIT-NEXT:  ^bb0():
// SPLIT-NEXT:    "test.op"() : () -> ()
// SPLIT-NEXT:  }
// SPLIT-NEXT:  // -----
// SPLIT-NEXT:  builtin.module {
// SPLIT-NEXT:  ^bb0():
// SPLIT-NEXT:    "test.op"() : () -> ()
// SPLIT-NEXT:  }
