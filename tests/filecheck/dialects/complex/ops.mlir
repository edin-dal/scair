// RUN: scair-opt %s | filecheck %s

builtin.module {
  func.func @ops(%arg0: f32) {
    %0 = "complex.constant"() <{value = [1.000000e-01, -1.000000e+00]}> : () -> complex<f64>
    %1 = "complex.constant"() <{value = [1.000000e-01 : f32, -1.000000e+00 : f32]}> : () -> complex<f32>
    %2 = "complex.create"(%arg0, %arg0) : (f32, f32) -> complex<f32>
    %3 = "complex.re"(%2) <{fastmath = #arith.fastmath<none>}> : (complex<f32>) -> f32
    %4 = "complex.im"(%2) <{fastmath = #arith.fastmath<none>}> : (complex<f32>) -> f32
    %5 = "complex.abs"(%2) <{fastmath = #arith.fastmath<none>}> : (complex<f32>) -> f32
    %6 = "complex.add"(%2, %2) <{fastmath = #arith.fastmath<none>}> : (complex<f32>, complex<f32>) -> complex<f32>
    // %7 = "complex.cos"(%2) <{fastmath = #arith.fastmath<none>}> : (complex<f32>) -> complex<f32>
    %8 = "complex.div"(%2, %2) <{fastmath = #arith.fastmath<none>}> : (complex<f32>, complex<f32>) -> complex<f32>
    // %9 = "complex.eq"(%2, %2) : (complex<f32>, complex<f32>) -> i1
    // %10 = "complex.exp"(%2) <{fastmath = #arith.fastmath<none>}> : (complex<f32>) -> complex<f32>
    // %11 = "complex.expm1"(%2) <{fastmath = #arith.fastmath<none>}> : (complex<f32>) -> complex<f32>
    // %12 = "complex.log"(%2) <{fastmath = #arith.fastmath<none>}> : (complex<f32>) -> complex<f32>
    // %13 = "complex.log1p"(%2) <{fastmath = #arith.fastmath<none>}> : (complex<f32>) -> complex<f32>
    %14 = "complex.mul"(%2, %2) <{fastmath = #arith.fastmath<none>}> : (complex<f32>, complex<f32>) -> complex<f32>
    %15 = "complex.neg"(%2) <{fastmath = #arith.fastmath<none>}> : (complex<f32>) -> complex<f32>
    // %16 = "complex.neq"(%2, %2) : (complex<f32>, complex<f32>) -> i1
    // %17 = "complex.sign"(%2) <{fastmath = #arith.fastmath<none>}> : (complex<f32>) -> complex<f32>
    // %18 = "complex.sin"(%2) <{fastmath = #arith.fastmath<none>}> : (complex<f32>) -> complex<f32>
    %19 = "complex.sub"(%2, %2) <{fastmath = #arith.fastmath<none>}> : (complex<f32>, complex<f32>) -> complex<f32>
    // %20 = "complex.tanh"(%2) <{fastmath = #arith.fastmath<none>}> : (complex<f32>) -> complex<f32>
    // %21 = "complex.pow"(%2, %2) <{fastmath = #arith.fastmath<none>}> : (complex<f32>, complex<f32>) -> complex<f32>
    // %22 = "complex.sqrt"(%2) <{fastmath = #arith.fastmath<none>}> : (complex<f32>) -> complex<f32>
    // %23 = "complex.rsqrt"(%2) <{fastmath = #arith.fastmath<none>}> : (complex<f32>) -> complex<f32>
    // %24 = "complex.atan2"(%2, %2) <{fastmath = #arith.fastmath<none>}> : (complex<f32>, complex<f32>) -> complex<f32>
    // %25 = "complex.tan"(%2) <{fastmath = #arith.fastmath<none>}> : (complex<f32>) -> complex<f32>
    // %26 = "complex.bitcast"(%2) : (complex<f32>) -> i64
    func.return
  }
}

// CHECK:       builtin.module {
// CHECK-NEXT:    func.func @ops(%0: f32) {
// CHECK-NEXT:      %1 = "complex.constant"() <{value = [0.1 : f64, -1.0 : f64]}> : () -> (complex<f64>)
// CHECK-NEXT:      %2 = "complex.constant"() <{value = [0.1 : f32, -1.0 : f32]}> : () -> (complex<f32>)
// CHECK-NEXT:      %3 = "complex.create"(%0, %0) : (f32, f32) -> (complex<f32>)
// CHECK-NEXT:      %4 = "complex.re"(%3) : (complex<f32>) -> (f32)
// CHECK-NEXT:      %5 = "complex.im"(%3) : (complex<f32>) -> (f32)
// CHECK-NEXT:      %6 = "complex.abs"(%3) : (complex<f32>) -> (f32)
// CHECK-NEXT:      %7 = "complex.add"(%3, %3) <{fastmath = #arith.fastmath<none>}> : (complex<f32>, complex<f32>) -> (complex<f32>)
// CHECK-NEXT:      %8 = "complex.div"(%3, %3) <{fastmath = #arith.fastmath<none>}> : (complex<f32>, complex<f32>) -> (complex<f32>)
// CHECK-NEXT:      %9 = "complex.mul"(%3, %3) <{fastmath = #arith.fastmath<none>}> : (complex<f32>, complex<f32>) -> (complex<f32>)
// CHECK-NEXT:      %10 = "complex.neg"(%3) <{fastmath = #arith.fastmath<none>}> : (complex<f32>) -> (complex<f32>)
// CHECK-NEXT:      %11 = "complex.sub"(%3, %3) <{fastmath = #arith.fastmath<none>}> : (complex<f32>, complex<f32>) -> (complex<f32>)
// CHECK-NEXT:      func.return
// CHECK-NEXT:    }
// CHECK-NEXT:  }
