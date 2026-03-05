// RUN: scair-opt %s --parsing-diagnostics --split-input-file | filecheck %s

#attralias = 0
"builtin.module"[]

// CHECK:       Parse error at /home/papychacal/scair/tests/filecheck/core/parser/aliases.mlir:4:17:

// CHECK:       "builtin.module"[]
// CHECK-NEXT:                  ^
// CHECK-NEXT:  "("

// -----

!typealias = i42
"builtin.module"[]

// CHECK:  Parse error at /home/papychacal/scair/tests/filecheck/core/parser/aliases.mlir:15:17:

// CHECK:       "builtin.module"[]
// CHECK-NEXT:                  ^
// CHECK-NEXT:  "("