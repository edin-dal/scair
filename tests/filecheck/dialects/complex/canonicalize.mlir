// RUN: scair-opt -p canonicalize %s | filecheck %s

builtin.module {
// CHECK:       builtin.module {

  func.func @create_of_real_and_imag(%0: complex<f32>) -> complex<f32> {
    %1 = "complex.re"(%0) : (complex<f32>) -> (f32)
    %2 = "complex.im"(%0) : (complex<f32>) -> (f32)
    %3 = "complex.create"(%1, %2) : (f32, f32) -> (complex<f32>)
    func.return %3 : complex<f32>
  }
  // CHECK-NEXT:    func.func @create_of_real_and_imag(%0: complex<f32>) -> complex<f32> {
  // CHECK-NEXT:      func.return %0 : complex<f32>
  // CHECK-NEXT:    }

  func.func @create_of_real_and_imag_different_operand(%0: complex<f32>, %1: complex<f32>) -> complex<f32> {
    %2 = "complex.re"(%0) : (complex<f32>) -> (f32)
    %3 = "complex.im"(%1) : (complex<f32>) -> (f32)
    %4 = "complex.create"(%2, %3) : (f32, f32) -> (complex<f32>)
    func.return %4 : complex<f32>
  }
  // CHECK-NEXT:    func.func @create_of_real_and_imag_different_operand(%0: complex<f32>, %1: complex<f32>) -> complex<f32> {
  // CHECK-NEXT:      %2 = "complex.re"(%0) : (complex<f32>) -> (f32)
  // CHECK-NEXT:      %3 = "complex.im"(%1) : (complex<f32>) -> (f32)
  // CHECK-NEXT:      %4 = "complex.create"(%2, %3) : (f32, f32) -> (complex<f32>)
  // CHECK-NEXT:      func.return %4 : complex<f32>
  // CHECK-NEXT:    }

  func.func @real_of_const() -> f32 {
    %0 = complex.constant [1.0 : f32, 0.0 : f32] : complex<f32>
    %1 = "complex.re"(%0) : (complex<f32>) -> (f32)
    func.return %1 : f32
  }
  // CHECK-NEXT:    func.func @real_of_const() -> f32 {
  // CHECK-NEXT:      %0 = "arith.constant"() <{value = 1.0 : f32}> : () -> (f32)
  // CHECK-NEXT:      func.return %0 : f32
  // CHECK-NEXT:    }

  func.func @real_of_create_op() -> f32 {
    %0 = "arith.constant"() <{value = 1.0 : f32}> : () -> (f32)
    %1 = "arith.constant"() <{value = 0.0 : f32}> : () -> (f32)
    %2 = "complex.create"(%0, %1) : (f32, f32) -> (complex<f32>)
    %3 = "complex.re"(%2) : (complex<f32>) -> (f32)
    func.return %3 : f32
  }
  // CHECK-NEXT:    func.func @real_of_create_op() -> f32 {
  // CHECK-NEXT:      %0 = "arith.constant"() <{value = 1.0 : f32}> : () -> (f32)
  // CHECK-NEXT:      func.return %0 : f32
  // CHECK-NEXT:    }

  func.func @imag_of_const() -> f32 {
    %0 = complex.constant [1.0 : f32, 0.0 : f32] : complex<f32>
    %1 = "complex.im"(%0) : (complex<f32>) -> (f32)
    func.return %1 : f32
  }
  // CHECK-NEXT:    func.func @imag_of_const() -> f32 {
  // CHECK-NEXT:      %0 = "arith.constant"() <{value = 0.0 : f32}> : () -> (f32)
  // CHECK-NEXT:      func.return %0 : f32
  // CHECK-NEXT:    }

  func.func @imag_of_create_op() -> f32 {
    %0 = "arith.constant"() <{value = 1.0 : f32}> : () -> (f32)
    %1 = "arith.constant"() <{value = 0.0 : f32}> : () -> (f32)
    %2 = "complex.create"(%0, %1) : (f32, f32) -> (complex<f32>)
    %3 = "complex.im"(%2) : (complex<f32>) -> (f32)
    func.return %3 : f32
  }
  // CHECK-NEXT:    func.func @imag_of_create_op() -> f32 {
  // CHECK-NEXT:      %0 = "arith.constant"() <{value = 0.0 : f32}> : () -> (f32)
  // CHECK-NEXT:      func.return %0 : f32
  // CHECK-NEXT:    }

  func.func @complex_add_sub_lhs() -> complex<f32> {
    %0 = complex.constant [1.0 : f32, 0.0 : f32] : complex<f32>
    %1 = complex.constant [0.0 : f32, 2.0 : f32] : complex<f32>
    %2 = "complex.sub"(%0, %1) <{fastmath = #arith.fastmath<none>}> : (complex<f32>, complex<f32>) -> (complex<f32>)
    %3 = "complex.add"(%2, %1) <{fastmath = #arith.fastmath<none>}> : (complex<f32>, complex<f32>) -> (complex<f32>)
    func.return %3 : complex<f32>
  }
  // CHECK-NEXT:    func.func @complex_add_sub_lhs() -> complex<f32> {
  // CHECK-NEXT:      %0 = complex.constant [1.0 : f32, 0.0 : f32] : complex<f32>
  // CHECK-NEXT:      func.return %0 : complex<f32>
  // CHECK-NEXT:    }

  func.func @complex_add_sub_rhs() -> complex<f32> {
    %0 = complex.constant [1.0 : f32, 0.0 : f32] : complex<f32>
    %1 = complex.constant [0.0 : f32, 2.0 : f32] : complex<f32>
    %2 = "complex.sub"(%0, %1) <{fastmath = #arith.fastmath<none>}> : (complex<f32>, complex<f32>) -> (complex<f32>)
    %3 = "complex.add"(%1, %2) <{fastmath = #arith.fastmath<none>}> : (complex<f32>, complex<f32>) -> (complex<f32>)
    func.return %3 : complex<f32>
  }
  // CHECK-NEXT:    func.func @complex_add_sub_rhs() -> complex<f32> {
  // CHECK-NEXT:      %0 = complex.constant [1.0 : f32, 0.0 : f32] : complex<f32>
  // CHECK-NEXT:      func.return %0 : complex<f32>
  // CHECK-NEXT:    }

  func.func @complex_neg_neg() -> complex<f32> {
    %0 = complex.constant [1.0 : f32, 0.0 : f32] : complex<f32>
    %1 = "complex.neg"(%0) <{fastmath = #arith.fastmath<none>}> : (complex<f32>) -> (complex<f32>)
    %2 = "complex.neg"(%1) <{fastmath = #arith.fastmath<none>}> : (complex<f32>) -> (complex<f32>)
    func.return %2 : complex<f32>
  }
  // CHECK-NEXT:    func.func @complex_neg_neg() -> complex<f32> {
  // CHECK-NEXT:      %0 = complex.constant [1.0 : f32, 0.0 : f32] : complex<f32>
  // CHECK-NEXT:      func.return %0 : complex<f32>
  // CHECK-NEXT:    }

//   func.func @complex_log_exp() -> complex<f32> {
//     %0 = complex.constant [1.0 : f32, 0.0 : f32] : complex<f32>
//     %1 = "complex.exp"(%0) <{fastmath = #arith.fastmath<none>}> : (complex<f32>) -> (complex<f32>)
//     %2 = "complex.log"(%1) <{fastmath = #arith.fastmath<none>}> : (complex<f32>) -> (complex<f32>)
//     func.return %2 : complex<f32>
//   }
//   func.func @complex_exp_log() -> complex<f32> {
//     %0 = complex.constant [1.0 : f32, 0.0 : f32] : complex<f32>
//     %1 = "complex.log"(%0) <{fastmath = #arith.fastmath<none>}> : (complex<f32>) -> (complex<f32>)
//     %2 = "complex.exp"(%1) <{fastmath = #arith.fastmath<none>}> : (complex<f32>) -> (complex<f32>)
//     func.return %2 : complex<f32>
//   }
//   func.func @complex_conj_conj() -> complex<f32> {
//     %0 = complex.constant [1.0 : f32, 0.0 : f32] : complex<f32>
//     %1 = "complex.conj"(%0) <{fastmath = #arith.fastmath<none>}> : (complex<f32>) -> (complex<f32>)
//     %2 = "complex.conj"(%1) <{fastmath = #arith.fastmath<none>}> : (complex<f32>) -> (complex<f32>)
//     func.return %2 : complex<f32>
//   }

  func.func @complex_add_zero() -> complex<f32> {
    %0 = complex.constant [1.0 : f32, 0.0 : f32] : complex<f32>
    %1 = complex.constant [0.0 : f32, 0.0 : f32] : complex<f32>
    %2 = "complex.add"(%0, %1) <{fastmath = #arith.fastmath<none>}> : (complex<f32>, complex<f32>) -> (complex<f32>)
    func.return %2 : complex<f32>
  }
  // CHECK-NEXT:    func.func @complex_add_zero() -> complex<f32> {
  // CHECK-NEXT:      %0 = complex.constant [1.0 : f32, 0.0 : f32] : complex<f32>
  // CHECK-NEXT:      func.return %0 : complex<f32>
  // CHECK-NEXT:    }

  func.func @complex_sub_add_lhs() -> complex<f32> {
    %0 = complex.constant [1.0 : f32, 0.0 : f32] : complex<f32>
    %1 = complex.constant [0.0 : f32, 2.0 : f32] : complex<f32>
    %2 = "complex.add"(%0, %1) <{fastmath = #arith.fastmath<none>}> : (complex<f32>, complex<f32>) -> (complex<f32>)
    %3 = "complex.sub"(%2, %1) <{fastmath = #arith.fastmath<none>}> : (complex<f32>, complex<f32>) -> (complex<f32>)
    func.return %3 : complex<f32>
  }
  // CHECK-NEXT:    func.func @complex_sub_add_lhs() -> complex<f32> {
  // CHECK-NEXT:      %0 = complex.constant [1.0 : f32, 0.0 : f32] : complex<f32>
  // CHECK-NEXT:      func.return %0 : complex<f32>
  // CHECK-NEXT:    }

  func.func @complex_sub_zero() -> complex<f32> {
    %0 = complex.constant [1.0 : f32, 0.0 : f32] : complex<f32>
    %1 = complex.constant [0.0 : f32, 0.0 : f32] : complex<f32>
    %2 = "complex.sub"(%0, %1) <{fastmath = #arith.fastmath<none>}> : (complex<f32>, complex<f32>) -> (complex<f32>)
    func.return %2 : complex<f32>
  }
  // CHECK-NEXT:    func.func @complex_sub_zero() -> complex<f32> {
  // CHECK-NEXT:      %0 = complex.constant [1.0 : f32, 0.0 : f32] : complex<f32>
  // CHECK-NEXT:      func.return %0 : complex<f32>
  // CHECK-NEXT:    }

  func.func @re_neg(%0: f32, %1: f32) -> f32 {
    %2 = "complex.create"(%0, %1) : (f32, f32) -> (complex<f32>)
    %3 = "complex.neg"(%2) <{fastmath = #arith.fastmath<none>}> : (complex<f32>) -> (complex<f32>)
    %4 = "complex.re"(%3) : (complex<f32>) -> (f32)
    func.return %4 : f32
  }
  // CHECK-NEXT:    func.func @re_neg(%0: f32, %1: f32) -> f32 {
  // CHECK-NEXT:      %2 = "arith.negf"(%0) <{fastmath = #arith.fastmath<none>}> : (f32) -> (f32)
  // CHECK-NEXT:      func.return %2 : f32
  // CHECK-NEXT:    }

  func.func @im_neg(%0: f32, %1: f32) -> f32 {
    %2 = "complex.create"(%0, %1) : (f32, f32) -> (complex<f32>)
    %3 = "complex.neg"(%2) <{fastmath = #arith.fastmath<none>}> : (complex<f32>) -> (complex<f32>)
    %4 = "complex.im"(%3) : (complex<f32>) -> (f32)
    func.return %4 : f32
  }
  // CHECK-NEXT:    func.func @im_neg(%0: f32, %1: f32) -> f32 {
  // CHECK-NEXT:      %2 = "arith.negf"(%1) <{fastmath = #arith.fastmath<none>}> : (f32) -> (f32)
  // CHECK-NEXT:      func.return %2 : f32
  // CHECK-NEXT:    }

  func.func @mul_one_f16(%0: f16, %1: f16) -> complex<f16> {
    %2 = "complex.create"(%0, %1) : (f16, f16) -> (complex<f16>)
    %3 = complex.constant [1.0 : f16, 0.0 : f16] : complex<f16>
    %4 = "complex.mul"(%2, %3) <{fastmath = #arith.fastmath<none>}> : (complex<f16>, complex<f16>) -> (complex<f16>)
    func.return %4 : complex<f16>
  }
  // CHECK-NEXT:    func.func @mul_one_f16(%0: f16, %1: f16) -> complex<f16> {
  // CHECK-NEXT:      %2 = "complex.create"(%0, %1) : (f16, f16) -> (complex<f16>)
  // CHECK-NEXT:      func.return %2 : complex<f16>
  // CHECK-NEXT:    }

  func.func @mul_one_f32(%0: f32, %1: f32) -> complex<f32> {
    %2 = "complex.create"(%0, %1) : (f32, f32) -> (complex<f32>)
    %3 = complex.constant [1.0 : f32, 0.0 : f32] : complex<f32>
    %4 = "complex.mul"(%2, %3) <{fastmath = #arith.fastmath<none>}> : (complex<f32>, complex<f32>) -> (complex<f32>)
    func.return %4 : complex<f32>
  }
  // CHECK-NEXT:    func.func @mul_one_f32(%0: f32, %1: f32) -> complex<f32> {
  // CHECK-NEXT:      %2 = "complex.create"(%0, %1) : (f32, f32) -> (complex<f32>)
  // CHECK-NEXT:      func.return %2 : complex<f32>
  // CHECK-NEXT:    }

  func.func @mul_one_f64(%0: f64, %1: f64) -> complex<f64> {
    %2 = "complex.create"(%0, %1) : (f64, f64) -> (complex<f64>)
    %3 = complex.constant [1.0 : f64, 0.0 : f64] : complex<f64>
    %4 = "complex.mul"(%2, %3) <{fastmath = #arith.fastmath<none>}> : (complex<f64>, complex<f64>) -> (complex<f64>)
    func.return %4 : complex<f64>
  }
  // CHECK-NEXT:    func.func @mul_one_f64(%0: f64, %1: f64) -> complex<f64> {
  // CHECK-NEXT:      %2 = "complex.create"(%0, %1) : (f64, f64) -> (complex<f64>)
  // CHECK-NEXT:      func.return %2 : complex<f64>
  // CHECK-NEXT:    }

  func.func @mul_one_f80(%0: f80, %1: f80) -> complex<f80> {
    %2 = "complex.create"(%0, %1) : (f80, f80) -> (complex<f80>)
    %3 = complex.constant [1.0 : f80, 0.0 : f80] : complex<f80>
    %4 = "complex.mul"(%2, %3) <{fastmath = #arith.fastmath<none>}> : (complex<f80>, complex<f80>) -> (complex<f80>)
    func.return %4 : complex<f80>
  }
  // CHECK-NEXT:    func.func @mul_one_f80(%0: f80, %1: f80) -> complex<f80> {
  // CHECK-NEXT:      %2 = "complex.create"(%0, %1) : (f80, f80) -> (complex<f80>)
  // CHECK-NEXT:      func.return %2 : complex<f80>
  // CHECK-NEXT:    }

  func.func @mul_one_f128(%0: f128, %1: f128) -> complex<f128> {
    %2 = "complex.create"(%0, %1) : (f128, f128) -> (complex<f128>)
    %3 = complex.constant [1.0 : f128, 0.0 : f128] : complex<f128>
    %4 = "complex.mul"(%2, %3) <{fastmath = #arith.fastmath<none>}> : (complex<f128>, complex<f128>) -> (complex<f128>)
    func.return %4 : complex<f128>
  }
  // CHECK-NEXT:    func.func @mul_one_f128(%0: f128, %1: f128) -> complex<f128> {
  // CHECK-NEXT:      %2 = "complex.create"(%0, %1) : (f128, f128) -> (complex<f128>)
  // CHECK-NEXT:      func.return %2 : complex<f128>
  // CHECK-NEXT:    }


  func.func @mul_cst() -> complex<f128> {
    %0 = complex.constant [1.0 : f128, 2.0 : f128] : complex<f128>
    %1 = complex.constant [3.0 : f128, 4.0 : f128] : complex<f128>
    %2 = "complex.mul"(%0, %1) <{fastmath = #arith.fastmath<none>}> : (complex<f128>, complex<f128>) -> (complex<f128>)
    func.return %2 : complex<f128>
  }
  // CHECK-NEXT:    func.func @mul_cst() -> complex<f128> {
  // CHECK-NEXT:      %0 = complex.constant [-5.0 : f128, 10.0 : f128] : complex<f128>
  // CHECK-NEXT:      func.return %0 : complex<f128>
  // CHECK-NEXT:    }

//   func.func @fold_between_complex(%0: complex<f32>) -> complex<f32> {
//     %1 = "complex.bitcast"(%0) : (complex<f32>) -> (i64)
//     %2 = "complex.bitcast"(%1) : (i64) -> (complex<f32>)
//     func.return %2 : complex<f32>
//   }
//   func.func @fold_between_i64(%0: i64) -> i64 {
//     %1 = "complex.bitcast"(%0) : (i64) -> (complex<f32>)
//     %2 = "complex.bitcast"(%1) : (complex<f32>) -> (i64)
//     func.return %2 : i64
//   }
//   func.func @canon_arith_bitcast(%0: f64) -> i64 {
//     %1 = "complex.bitcast"(%0) : (f64) -> (complex<f32>)
//     %2 = "complex.bitcast"(%1) : (complex<f32>) -> (i64)
//     func.return %2 : i64
//   }
//   func.func @double_bitcast(%0: f64) -> complex<f32> {
//     %1 = "arith.bitcast"(%0) : (f64) -> (i64)
//     %2 = "complex.bitcast"(%1) : (i64) -> (complex<f32>)
//     func.return %2 : complex<f32>
//   }
//   func.func @double_reverse_bitcast(%0: complex<f32>) -> f64 {
//     %1 = "complex.bitcast"(%0) : (complex<f32>) -> (i64)
//     %2 = "arith.bitcast"(%1) : (i64) -> (f64)
//     func.return %2 : f64
//   }
}
  // CHECK-NEXT:  }
