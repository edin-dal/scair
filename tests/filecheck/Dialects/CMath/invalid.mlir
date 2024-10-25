// RUN: scair-opt %s --verify_diagnostics --split_input_file | filecheck %s

"test.op"() {"invalid_type" = !cmath.complex<i8>} : () -> ()
// CHECK: Complex type must be constructed with either 'f32' or 'f64' attribute.

// -----

"test.op"() {"invalid_type" = !cmath.complex<>} : () -> ()
// CHECK: TupleStream Tuple must contain 1 elements only.

// -----

"test.op"() {"invalid_type" = !cmath.complex<f32, f32>} : () -> ()
// CHECK: TupleStream Tuple must contain 1 elements only.

// -----

"cmath.norm"() : () -> ()
// CHECK: Norm Operation must only contain 1 operand of 'complex' type, and 1 result of 'f32' or 'f64'.

// -----

"cmath.mul"() : () -> ()
// CHECK: Mul Operation must only contain 2 operands and 1 result of 'complex' type.
