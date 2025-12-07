// RUN: scair-opt %s --parsing-diagnostics --verify-diagnostics --split-input-file | filecheck %s -DFILE=%s

"test.op"() {"invalid_type" = !cmath.complex<i8>} : () -> ()

// TODO: This should fail because cmath.complex expect f32 | f64. This needs frontend
// support

// CHECK:       builtin.module {
// CHECK-NEXT:    "test.op"() {invalid_type = !cmath.complex<i8>} : () -> ()
// CHECK-NEXT:  }

// -----

"test.op"() {"invalid_type" = !cmath.complex} : () -> ()
// CHECK: Parse error at [[FILE]]:14:45:
// CHECK: "test.op"() {"invalid_type" = !cmath.complex} : () -> ()
// CHECK:                                             ^
// CHECK-NEXT: Number of attributes 1 does not match the number of provided attributes 0

// -----

"test.op"() {"invalid_type" = !cmath.complex<f32, f32>} : () -> ()
// CHECK: Parse error at [[FILE]]:22:55:
// CHECK: "test.op"() {"invalid_type" = !cmath.complex<f32, f32>} : () -> ()
// CHECK:                                                       ^
// CHECK: Number of attributes 1 does not match the number of provided attributes 2

// -----

"cmath.norm"() : () -> ()
// CHECK:  java.lang.Exception: Expected 1 operands, got 0.

// -----

"cmath.mul"() : () -> ()
// CHECK:  java.lang.Exception: Expected 2 operands, got 0.
