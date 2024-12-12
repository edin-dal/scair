// RUN: scair-opt --split_input_file --parsing_diagnostics %s | filecheck %s

notir
// -----
notir

// CHECK:       Parse error at {{.*}}scair/tests/filecheck/tools/parsing_diagnostics.mlir:3:1:
// CHECK-EMPTY:
// CHECK-NEXT:  Expected (OperationPat | Pass | end-of-input):3:1, found "notir"
// CHECK-NEXT:  // -----
// CHECK-NEXT:  Parse error at {{.*}}scair/tests/filecheck/tools/parsing_diagnostics.mlir:1:1:
// CHECK-EMPTY:
// CHECK-NEXT:  Expected (start-of-input | OperationPat | Pass | end-of-input):1:1, found "notir\n\n// "
