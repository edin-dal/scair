// RUN: scair-opt %s --parsing_diagnostics --verify_diagnostics --split_input_file | filecheck %s

"test.op"() {"invalid_type" = !cmath.complex<i8>} : () -> ()

// TODO: This should fail because cmath.complex expect f32 | f64. This needs frontend
// support

// CHECK: Parse error at /home/maks/phd/scair-main/scair/tests/filecheck/dialects/cmath/invalid.mlir:3:49:
// CHECK: "test.op"() {"invalid_type" = !cmath.complex<i8>} : () -> ()                  
// CHECK:                                                 ^
// CHECK: Expected typ to be of type scair.dialects.builtin.FloatType | scair.dialects.builtin.IndexType, got IntegerType(IntData(8),Signless)

// -----

"test.op"() {"invalid_type" = !cmath.complex} : () -> ()
// CHECK: Parse error at /home/maks/phd/scair-main/scair/tests/filecheck/dialects/cmath/invalid.mlir:2:31:
// CHECK: "test.op"() {"invalid_type" = !cmath.complex} : () -> ()
// CHECK:                               ^
// CHECK: (BuiltIn | DialectType | DialectAttribute)

// -----

"test.op"() {"invalid_type" = !cmath.complex<f32, f32>} : () -> ()
// CHECK: Parse error at /home/maks/phd/scair-main/scair/tests/filecheck/dialects/cmath/invalid.mlir:2:55:
// CHECK: "test.op"() {"invalid_type" = !cmath.complex<f32, f32>} : () -> ()
// CHECK:                                                       ^
// CHECK: Number of attributes 1 does not match the number of provided attributes 2

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
