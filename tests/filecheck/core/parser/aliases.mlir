// RUN: scair-opt %s --parsing-diagnostics --split-input-file | filecheck %s -DSOURCE_FILE=%s

#attralias = 0
"builtin.module"[]

// CHECK:       Parse error at [[SOURCE_FILE]]:4:17:

// CHECK:       "builtin.module"[]
// CHECK-NEXT:                  ^
// CHECK-NEXT:  "("

// -----

!typealias = i42
"builtin.module"[]

// CHECK:  Parse error at [[SOURCE_FILE]]:15:17:

// CHECK:       "builtin.module"[]
// CHECK-NEXT:                  ^
// CHECK-NEXT:  "("
