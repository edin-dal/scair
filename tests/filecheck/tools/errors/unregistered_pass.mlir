// RUN: scair-opt -p not-a-pass %s 2>&1 | filecheck %s 

builtin.module{}

// CHECK:       error: 'not-a-pass' does not refer to a registered pass.
// CHECK-NEXT:  Currently registered passes are:
// CHECK-NEXT:    - 
