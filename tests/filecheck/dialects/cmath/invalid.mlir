// RUN: scair-opt %s --verify_diagnostics --split_input_file | filecheck %s

"test.op"() {"invalid_type" = !cmath.complex<i8>} : () -> ()

// TODO: This should fail because cmath.complex expect f32 | f64. This needs frontend
// support

// CHECK:       builtin.module {
// CHECK-NEXT:  ^bb0():
// CHECK-NEXT:    "test.op"() {invalid_type = !cmath.complex<i8>} : () -> ()
// CHECK-NEXT:  }

// -----

"test.op"() {"invalid_type" = !cmath.complex} : () -> ()
// CHECK: Expected 1 parameters, got 0

// -----

"test.op"() {"invalid_type" = !cmath.complex<f32, f32>} : () -> ()
// CHECK: Expected 1 parameters, got 2

// -----

"cmath.norm"() : () -> ()
// CHECK: Expected 1 operands, got 0

// -----

"cmath.mul"() : () -> ()
// CHECK: Expected 2 operands, got 0
