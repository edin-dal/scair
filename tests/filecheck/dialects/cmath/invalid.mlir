// RUN: scair-opt %s --verify_diagnostics --split_input_file | filecheck %s

"test.op"() {"invalid_type" = !cmath.complex<i8>} : () -> ()

// TODO: This should fail because cmath.complex expect f32 | f64. This needs frontend
// support

// CHECK:       builtin.module {
// CHECK-NEXT:    "test.op"() {invalid_type = !cmath.complex<[i8]>} : () -> ()
// CHECK-NEXT:  }

// -----

"test.op"() {"invalid_type" = !cmath.complex} : () -> ()
// CHECK:  builtin.module {
// CHECK-NEXT:    "test.op"() {invalid_type = !cmath.complex<[]>} : () -> ()
// CHECK-NEXT:  }

// -----

"test.op"() {"invalid_type" = !cmath.complex<f32, f32>} : () -> ()
// CHECK:  builtin.module {
// CHECK-NEXT:    "test.op"() {invalid_type = !cmath.complex<[f32, f32]>} : () -> ()
// CHECK-NEXT:  }

// -----

"cmath.norm"() : () -> ()
// CHECK:  builtin.module {
// CHECK-NEXT:    "cmath.norm"() : () -> ()
// CHECK-NEXT:  }

// -----

"cmath.mul"() : () -> ()
// CHECK:  builtin.module {
// CHECK-NEXT:    "cmath.mul"() : () -> ()
// CHECK-NEXT:  }
