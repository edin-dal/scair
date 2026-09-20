// RUN: scair-opt %s --parsing-diagnostics --verify-diagnostics --split-input-file | filecheck %s

// complex.constant's value is an ArrayAttribute[IntegerAttr | FloatAttr]. Erasure
// hides that element type from a plain type test, so the array's contents have to
// be checked when structuring the operation.

builtin.module {
  %0 = complex.constant ["a", "b"] : complex<f32>
}

// CHECK: Type mismatch for property "value": expected scair.dialects.builtin.ArrayAttribute[scair.dialects.builtin.IntegerAttr | scair.dialects.builtin.FloatAttr], but found ["a", "b"]

// -----

// A well-typed value goes through untouched.

builtin.module {
  %0 = complex.constant [1.0 : f32, 0.0 : f32] : complex<f32>
}

// CHECK:       builtin.module {
// CHECK-NEXT:    %0 = complex.constant [1.0 : f32, 0.0 : f32] : complex<f32>
// CHECK-NEXT:  }
