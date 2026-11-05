// RUN: scair-opt %s | mlir-opt --mlir-print-op-generic | scair-opt | filecheck %s

%f, %g, %h = "test.op"() : () -> (f32, f32, f32)
// CHECK:    %0, %1, %2 = "test.op"() : () -> (f32, f32, f32)
%v, %w, %x = "test.op"() : () -> (vector<4xf32>, vector<4xf32>, vector<4xf32>)
// CHECK-NEXT:    %3, %4, %5 = "test.op"() : () -> (vector<4xf32>, vector<4xf32>, vector<4xf32>)
%t, %u = "test.op"() : () -> (tensor<2x3xf64>, tensor<2x3xf64>)
// CHECK-NEXT:    %6, %7 = "test.op"() : () -> (tensor<2x3xf64>, tensor<2x3xf64>)
%i, %j = "test.op"() : () -> (i32, i32)
// CHECK-NEXT:    %8, %9 = "test.op"() : () -> (i32, i32)
%iv, %iw = "test.op"() : () -> (vector<4xi64>, vector<4xi64>)
// CHECK-NEXT:    %10, %11 = "test.op"() : () -> (vector<4xi64>, vector<4xi64>)
%ix, %iy = "test.op"() : () -> (index, index)
// CHECK-NEXT:    %12, %13 = "test.op"() : () -> (index, index)

%r1 = "math.acos"(%f) : (f32) -> f32
// CHECK-NEXT:    %14 = "math.acos"(%0) <{fastmath = #arith.fastmath<none>}> : (f32) -> f32
%r2 = "math.acos"(%v) <{fastmath = #arith.fastmath<fast>}> : (vector<4xf32>) -> vector<4xf32>
// CHECK-NEXT:    %15 = "math.acos"(%3) <{fastmath = #arith.fastmath<fast>}> : (vector<4xf32>) -> vector<4xf32>
%r3 = "math.acos"(%t) : (tensor<2x3xf64>) -> tensor<2x3xf64>
// CHECK-NEXT:    %16 = "math.acos"(%6) <{fastmath = #arith.fastmath<none>}> : (tensor<2x3xf64>) -> tensor<2x3xf64>
%r4 = "math.acosh"(%f) : (f32) -> f32
// CHECK-NEXT:    %17 = "math.acosh"(%0) <{fastmath = #arith.fastmath<none>}> : (f32) -> f32
%r5 = "math.acosh"(%v) <{fastmath = #arith.fastmath<nnan,ninf>}> : (vector<4xf32>) -> vector<4xf32>
// CHECK-NEXT:    %18 = "math.acosh"(%3) <{fastmath = #arith.fastmath<nnan,ninf>}> : (vector<4xf32>) -> vector<4xf32>
%r6 = "math.asin"(%f) : (f32) -> f32
// CHECK-NEXT:    %19 = "math.asin"(%0) <{fastmath = #arith.fastmath<none>}> : (f32) -> f32
%r7 = "math.asin"(%v) <{fastmath = #arith.fastmath<reassoc,contract>}> : (vector<4xf32>) -> vector<4xf32>
// CHECK-NEXT:    %20 = "math.asin"(%3) <{fastmath = #arith.fastmath<reassoc,contract>}> : (vector<4xf32>) -> vector<4xf32>
%r8 = "math.asinh"(%f) : (f32) -> f32
// CHECK-NEXT:    %21 = "math.asinh"(%0) <{fastmath = #arith.fastmath<none>}> : (f32) -> f32
%r9 = "math.asinh"(%v) <{fastmath = #arith.fastmath<fast>}> : (vector<4xf32>) -> vector<4xf32>
// CHECK-NEXT:    %22 = "math.asinh"(%3) <{fastmath = #arith.fastmath<fast>}> : (vector<4xf32>) -> vector<4xf32>
%r10 = "math.atan"(%f) : (f32) -> f32
// CHECK-NEXT:    %23 = "math.atan"(%0) <{fastmath = #arith.fastmath<none>}> : (f32) -> f32
%r11 = "math.atan"(%v) <{fastmath = #arith.fastmath<nnan,ninf>}> : (vector<4xf32>) -> vector<4xf32>
// CHECK-NEXT:    %24 = "math.atan"(%3) <{fastmath = #arith.fastmath<nnan,ninf>}> : (vector<4xf32>) -> vector<4xf32>
%r12 = "math.atanh"(%f) : (f32) -> f32
// CHECK-NEXT:    %25 = "math.atanh"(%0) <{fastmath = #arith.fastmath<none>}> : (f32) -> f32
%r13 = "math.atanh"(%v) <{fastmath = #arith.fastmath<reassoc,contract>}> : (vector<4xf32>) -> vector<4xf32>
// CHECK-NEXT:    %26 = "math.atanh"(%3) <{fastmath = #arith.fastmath<reassoc,contract>}> : (vector<4xf32>) -> vector<4xf32>
%r14 = "math.cbrt"(%f) : (f32) -> f32
// CHECK-NEXT:    %27 = "math.cbrt"(%0) <{fastmath = #arith.fastmath<none>}> : (f32) -> f32
%r15 = "math.cbrt"(%v) <{fastmath = #arith.fastmath<fast>}> : (vector<4xf32>) -> vector<4xf32>
// CHECK-NEXT:    %28 = "math.cbrt"(%3) <{fastmath = #arith.fastmath<fast>}> : (vector<4xf32>) -> vector<4xf32>
%r16 = "math.ceil"(%f) : (f32) -> f32
// CHECK-NEXT:    %29 = "math.ceil"(%0) <{fastmath = #arith.fastmath<none>}> : (f32) -> f32
%r17 = "math.ceil"(%v) <{fastmath = #arith.fastmath<nnan,ninf>}> : (vector<4xf32>) -> vector<4xf32>
// CHECK-NEXT:    %30 = "math.ceil"(%3) <{fastmath = #arith.fastmath<nnan,ninf>}> : (vector<4xf32>) -> vector<4xf32>
%r18 = "math.ceil"(%t) <{fastmath = #arith.fastmath<reassoc,contract>}> : (tensor<2x3xf64>) -> tensor<2x3xf64>
// CHECK-NEXT:    %31 = "math.ceil"(%6) <{fastmath = #arith.fastmath<reassoc,contract>}> : (tensor<2x3xf64>) -> tensor<2x3xf64>
%r19 = "math.cos"(%f) : (f32) -> f32
// CHECK-NEXT:    %32 = "math.cos"(%0) <{fastmath = #arith.fastmath<none>}> : (f32) -> f32
%r20 = "math.cos"(%v) <{fastmath = #arith.fastmath<reassoc,contract>}> : (vector<4xf32>) -> vector<4xf32>
// CHECK-NEXT:    %33 = "math.cos"(%3) <{fastmath = #arith.fastmath<reassoc,contract>}> : (vector<4xf32>) -> vector<4xf32>
%r21 = "math.cosh"(%f) : (f32) -> f32
// CHECK-NEXT:    %34 = "math.cosh"(%0) <{fastmath = #arith.fastmath<none>}> : (f32) -> f32
%r22 = "math.cosh"(%v) <{fastmath = #arith.fastmath<fast>}> : (vector<4xf32>) -> vector<4xf32>
// CHECK-NEXT:    %35 = "math.cosh"(%3) <{fastmath = #arith.fastmath<fast>}> : (vector<4xf32>) -> vector<4xf32>
%r23 = "math.erf"(%f) : (f32) -> f32
// CHECK-NEXT:    %36 = "math.erf"(%0) <{fastmath = #arith.fastmath<none>}> : (f32) -> f32
%r24 = "math.erf"(%v) <{fastmath = #arith.fastmath<nnan,ninf>}> : (vector<4xf32>) -> vector<4xf32>
// CHECK-NEXT:    %37 = "math.erf"(%3) <{fastmath = #arith.fastmath<nnan,ninf>}> : (vector<4xf32>) -> vector<4xf32>
%r25 = "math.erfc"(%f) : (f32) -> f32
// CHECK-NEXT:    %38 = "math.erfc"(%0) <{fastmath = #arith.fastmath<none>}> : (f32) -> f32
%r26 = "math.erfc"(%v) <{fastmath = #arith.fastmath<reassoc,contract>}> : (vector<4xf32>) -> vector<4xf32>
// CHECK-NEXT:    %39 = "math.erfc"(%3) <{fastmath = #arith.fastmath<reassoc,contract>}> : (vector<4xf32>) -> vector<4xf32>
%r27 = "math.exp"(%f) : (f32) -> f32
// CHECK-NEXT:    %40 = "math.exp"(%0) <{fastmath = #arith.fastmath<none>}> : (f32) -> f32
%r28 = "math.exp"(%v) <{fastmath = #arith.fastmath<fast>}> : (vector<4xf32>) -> vector<4xf32>
// CHECK-NEXT:    %41 = "math.exp"(%3) <{fastmath = #arith.fastmath<fast>}> : (vector<4xf32>) -> vector<4xf32>
%r29 = "math.exp2"(%f) : (f32) -> f32
// CHECK-NEXT:    %42 = "math.exp2"(%0) <{fastmath = #arith.fastmath<none>}> : (f32) -> f32
%r30 = "math.exp2"(%v) <{fastmath = #arith.fastmath<nnan,ninf>}> : (vector<4xf32>) -> vector<4xf32>
// CHECK-NEXT:    %43 = "math.exp2"(%3) <{fastmath = #arith.fastmath<nnan,ninf>}> : (vector<4xf32>) -> vector<4xf32>
%r31 = "math.expm1"(%f) : (f32) -> f32
// CHECK-NEXT:    %44 = "math.expm1"(%0) <{fastmath = #arith.fastmath<none>}> : (f32) -> f32
%r32 = "math.expm1"(%v) <{fastmath = #arith.fastmath<reassoc,contract>}> : (vector<4xf32>) -> vector<4xf32>
// CHECK-NEXT:    %45 = "math.expm1"(%3) <{fastmath = #arith.fastmath<reassoc,contract>}> : (vector<4xf32>) -> vector<4xf32>
%r33 = "math.expm1"(%t) <{fastmath = #arith.fastmath<nnan,ninf>}> : (tensor<2x3xf64>) -> tensor<2x3xf64>
// CHECK-NEXT:    %46 = "math.expm1"(%6) <{fastmath = #arith.fastmath<nnan,ninf>}> : (tensor<2x3xf64>) -> tensor<2x3xf64>
%r34 = "math.floor"(%f) : (f32) -> f32
// CHECK-NEXT:    %47 = "math.floor"(%0) <{fastmath = #arith.fastmath<none>}> : (f32) -> f32
%r35 = "math.floor"(%v) <{fastmath = #arith.fastmath<fast>}> : (vector<4xf32>) -> vector<4xf32>
// CHECK-NEXT:    %48 = "math.floor"(%3) <{fastmath = #arith.fastmath<fast>}> : (vector<4xf32>) -> vector<4xf32>
%r36 = "math.log"(%f) : (f32) -> f32
// CHECK-NEXT:    %49 = "math.log"(%0) <{fastmath = #arith.fastmath<none>}> : (f32) -> f32
%r37 = "math.log"(%v) <{fastmath = #arith.fastmath<nnan,ninf>}> : (vector<4xf32>) -> vector<4xf32>
// CHECK-NEXT:    %50 = "math.log"(%3) <{fastmath = #arith.fastmath<nnan,ninf>}> : (vector<4xf32>) -> vector<4xf32>
%r38 = "math.log10"(%f) : (f32) -> f32
// CHECK-NEXT:    %51 = "math.log10"(%0) <{fastmath = #arith.fastmath<none>}> : (f32) -> f32
%r39 = "math.log10"(%v) <{fastmath = #arith.fastmath<reassoc,contract>}> : (vector<4xf32>) -> vector<4xf32>
// CHECK-NEXT:    %52 = "math.log10"(%3) <{fastmath = #arith.fastmath<reassoc,contract>}> : (vector<4xf32>) -> vector<4xf32>
%r40 = "math.log1p"(%f) : (f32) -> f32
// CHECK-NEXT:    %53 = "math.log1p"(%0) <{fastmath = #arith.fastmath<none>}> : (f32) -> f32
%r41 = "math.log1p"(%v) <{fastmath = #arith.fastmath<fast>}> : (vector<4xf32>) -> vector<4xf32>
// CHECK-NEXT:    %54 = "math.log1p"(%3) <{fastmath = #arith.fastmath<fast>}> : (vector<4xf32>) -> vector<4xf32>
%r42 = "math.log2"(%f) : (f32) -> f32
// CHECK-NEXT:    %55 = "math.log2"(%0) <{fastmath = #arith.fastmath<none>}> : (f32) -> f32
%r43 = "math.log2"(%v) <{fastmath = #arith.fastmath<nnan,ninf>}> : (vector<4xf32>) -> vector<4xf32>
// CHECK-NEXT:    %56 = "math.log2"(%3) <{fastmath = #arith.fastmath<nnan,ninf>}> : (vector<4xf32>) -> vector<4xf32>
%r44 = "math.round"(%f) : (f32) -> f32
// CHECK-NEXT:    %57 = "math.round"(%0) <{fastmath = #arith.fastmath<none>}> : (f32) -> f32
%r45 = "math.round"(%v) <{fastmath = #arith.fastmath<reassoc,contract>}> : (vector<4xf32>) -> vector<4xf32>
// CHECK-NEXT:    %58 = "math.round"(%3) <{fastmath = #arith.fastmath<reassoc,contract>}> : (vector<4xf32>) -> vector<4xf32>
%r46 = "math.roundeven"(%f) : (f32) -> f32
// CHECK-NEXT:    %59 = "math.roundeven"(%0) <{fastmath = #arith.fastmath<none>}> : (f32) -> f32
%r47 = "math.roundeven"(%v) <{fastmath = #arith.fastmath<fast>}> : (vector<4xf32>) -> vector<4xf32>
// CHECK-NEXT:    %60 = "math.roundeven"(%3) <{fastmath = #arith.fastmath<fast>}> : (vector<4xf32>) -> vector<4xf32>
%r48 = "math.roundeven"(%t) <{fastmath = #arith.fastmath<fast>}> : (tensor<2x3xf64>) -> tensor<2x3xf64>
// CHECK-NEXT:    %61 = "math.roundeven"(%6) <{fastmath = #arith.fastmath<fast>}> : (tensor<2x3xf64>) -> tensor<2x3xf64>
%r49 = "math.rsqrt"(%f) : (f32) -> f32
// CHECK-NEXT:    %62 = "math.rsqrt"(%0) <{fastmath = #arith.fastmath<none>}> : (f32) -> f32
%r50 = "math.rsqrt"(%v) <{fastmath = #arith.fastmath<nnan,ninf>}> : (vector<4xf32>) -> vector<4xf32>
// CHECK-NEXT:    %63 = "math.rsqrt"(%3) <{fastmath = #arith.fastmath<nnan,ninf>}> : (vector<4xf32>) -> vector<4xf32>
%r51 = "math.sin"(%f) : (f32) -> f32
// CHECK-NEXT:    %64 = "math.sin"(%0) <{fastmath = #arith.fastmath<none>}> : (f32) -> f32
%r52 = "math.sin"(%v) <{fastmath = #arith.fastmath<reassoc,contract>}> : (vector<4xf32>) -> vector<4xf32>
// CHECK-NEXT:    %65 = "math.sin"(%3) <{fastmath = #arith.fastmath<reassoc,contract>}> : (vector<4xf32>) -> vector<4xf32>
%r53 = "math.sinh"(%f) : (f32) -> f32
// CHECK-NEXT:    %66 = "math.sinh"(%0) <{fastmath = #arith.fastmath<none>}> : (f32) -> f32
%r54 = "math.sinh"(%v) <{fastmath = #arith.fastmath<fast>}> : (vector<4xf32>) -> vector<4xf32>
// CHECK-NEXT:    %67 = "math.sinh"(%3) <{fastmath = #arith.fastmath<fast>}> : (vector<4xf32>) -> vector<4xf32>
%r55 = "math.sqrt"(%f) : (f32) -> f32
// CHECK-NEXT:    %68 = "math.sqrt"(%0) <{fastmath = #arith.fastmath<none>}> : (f32) -> f32
%r56 = "math.sqrt"(%v) <{fastmath = #arith.fastmath<nnan,ninf>}> : (vector<4xf32>) -> vector<4xf32>
// CHECK-NEXT:    %69 = "math.sqrt"(%3) <{fastmath = #arith.fastmath<nnan,ninf>}> : (vector<4xf32>) -> vector<4xf32>
%r57 = "math.tan"(%f) : (f32) -> f32
// CHECK-NEXT:    %70 = "math.tan"(%0) <{fastmath = #arith.fastmath<none>}> : (f32) -> f32
%r58 = "math.tan"(%v) <{fastmath = #arith.fastmath<reassoc,contract>}> : (vector<4xf32>) -> vector<4xf32>
// CHECK-NEXT:    %71 = "math.tan"(%3) <{fastmath = #arith.fastmath<reassoc,contract>}> : (vector<4xf32>) -> vector<4xf32>
%r59 = "math.tanh"(%f) : (f32) -> f32
// CHECK-NEXT:    %72 = "math.tanh"(%0) <{fastmath = #arith.fastmath<none>}> : (f32) -> f32
%r60 = "math.tanh"(%v) <{fastmath = #arith.fastmath<fast>}> : (vector<4xf32>) -> vector<4xf32>
// CHECK-NEXT:    %73 = "math.tanh"(%3) <{fastmath = #arith.fastmath<fast>}> : (vector<4xf32>) -> vector<4xf32>
%r61 = "math.trunc"(%f) : (f32) -> f32
// CHECK-NEXT:    %74 = "math.trunc"(%0) <{fastmath = #arith.fastmath<none>}> : (f32) -> f32
%r62 = "math.trunc"(%v) <{fastmath = #arith.fastmath<nnan,ninf>}> : (vector<4xf32>) -> vector<4xf32>
// CHECK-NEXT:    %75 = "math.trunc"(%3) <{fastmath = #arith.fastmath<nnan,ninf>}> : (vector<4xf32>) -> vector<4xf32>
%r63 = "math.trunc"(%t) : (tensor<2x3xf64>) -> tensor<2x3xf64>
// CHECK-NEXT:    %76 = "math.trunc"(%6) <{fastmath = #arith.fastmath<none>}> : (tensor<2x3xf64>) -> tensor<2x3xf64>
%r64 = "math.atan2"(%f, %g) : (f32, f32) -> f32
// CHECK-NEXT:    %77 = "math.atan2"(%0, %1) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
%r65 = "math.atan2"(%v, %w) <{fastmath = #arith.fastmath<fast>}> : (vector<4xf32>, vector<4xf32>) -> vector<4xf32>
// CHECK-NEXT:    %78 = "math.atan2"(%3, %4) <{fastmath = #arith.fastmath<fast>}> : (vector<4xf32>, vector<4xf32>) -> vector<4xf32>
%r66 = "math.atan2"(%t, %u) : (tensor<2x3xf64>, tensor<2x3xf64>) -> tensor<2x3xf64>
// CHECK-NEXT:    %79 = "math.atan2"(%6, %7) <{fastmath = #arith.fastmath<none>}> : (tensor<2x3xf64>, tensor<2x3xf64>) -> tensor<2x3xf64>
%r67 = "math.copysign"(%f, %g) : (f32, f32) -> f32
// CHECK-NEXT:    %80 = "math.copysign"(%0, %1) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
%r68 = "math.copysign"(%v, %w) <{fastmath = #arith.fastmath<nnan,ninf>}> : (vector<4xf32>, vector<4xf32>) -> vector<4xf32>
// CHECK-NEXT:    %81 = "math.copysign"(%3, %4) <{fastmath = #arith.fastmath<nnan,ninf>}> : (vector<4xf32>, vector<4xf32>) -> vector<4xf32>
%r69 = "math.copysign"(%t, %u) : (tensor<2x3xf64>, tensor<2x3xf64>) -> tensor<2x3xf64>
// CHECK-NEXT:    %82 = "math.copysign"(%6, %7) <{fastmath = #arith.fastmath<none>}> : (tensor<2x3xf64>, tensor<2x3xf64>) -> tensor<2x3xf64>
%r70 = "math.powf"(%f, %g) : (f32, f32) -> f32
// CHECK-NEXT:    %83 = "math.powf"(%0, %1) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
%r71 = "math.powf"(%v, %w) <{fastmath = #arith.fastmath<reassoc,contract>}> : (vector<4xf32>, vector<4xf32>) -> vector<4xf32>
// CHECK-NEXT:    %84 = "math.powf"(%3, %4) <{fastmath = #arith.fastmath<reassoc,contract>}> : (vector<4xf32>, vector<4xf32>) -> vector<4xf32>
%r72 = "math.powf"(%t, %u) : (tensor<2x3xf64>, tensor<2x3xf64>) -> tensor<2x3xf64>
// CHECK-NEXT:    %85 = "math.powf"(%6, %7) <{fastmath = #arith.fastmath<none>}> : (tensor<2x3xf64>, tensor<2x3xf64>) -> tensor<2x3xf64>
%r73 = "math.fma"(%f, %g, %h) : (f32, f32, f32) -> f32
// CHECK-NEXT:    %86 = "math.fma"(%0, %1, %2) <{fastmath = #arith.fastmath<none>}> : (f32, f32, f32) -> f32
%r74 = "math.fma"(%v, %w, %x) <{fastmath = #arith.fastmath<contract>}> : (vector<4xf32>, vector<4xf32>, vector<4xf32>) -> vector<4xf32>
// CHECK-NEXT:    %87 = "math.fma"(%3, %4, %5) <{fastmath = #arith.fastmath<contract>}> : (vector<4xf32>, vector<4xf32>, vector<4xf32>) -> vector<4xf32>
%r75 = "math.clampf"(%f, %g, %h) : (f32, f32, f32) -> f32
// CHECK-NEXT:    %88 = "math.clampf"(%0, %1, %2) <{fastmath = #arith.fastmath<none>}> : (f32, f32, f32) -> f32
%r76 = "math.clampf"(%v, %w, %x) <{fastmath = #arith.fastmath<nnan>}> : (vector<4xf32>, vector<4xf32>, vector<4xf32>) -> vector<4xf32>
// CHECK-NEXT:    %89 = "math.clampf"(%3, %4, %5) <{fastmath = #arith.fastmath<nnan>}> : (vector<4xf32>, vector<4xf32>, vector<4xf32>) -> vector<4xf32>
%r77 = "math.absi"(%i) : (i32) -> i32
// CHECK-NEXT:    %90 = "math.absi"(%8) : (i32) -> i32
%r78 = "math.absi"(%iv) : (vector<4xi64>) -> vector<4xi64>
// CHECK-NEXT:    %91 = "math.absi"(%10) : (vector<4xi64>) -> vector<4xi64>
%r79 = "math.absi"(%ix) : (index) -> index
// CHECK-NEXT:    %92 = "math.absi"(%12) : (index) -> index
%r80 = "math.ctlz"(%i) : (i32) -> i32
// CHECK-NEXT:    %93 = "math.ctlz"(%8) : (i32) -> i32
%r81 = "math.ctlz"(%iv) : (vector<4xi64>) -> vector<4xi64>
// CHECK-NEXT:    %94 = "math.ctlz"(%10) : (vector<4xi64>) -> vector<4xi64>
%r82 = "math.ctlz"(%ix) : (index) -> index
// CHECK-NEXT:    %95 = "math.ctlz"(%12) : (index) -> index
%r83 = "math.cttz"(%i) : (i32) -> i32
// CHECK-NEXT:    %96 = "math.cttz"(%8) : (i32) -> i32
%r84 = "math.cttz"(%iv) : (vector<4xi64>) -> vector<4xi64>
// CHECK-NEXT:    %97 = "math.cttz"(%10) : (vector<4xi64>) -> vector<4xi64>
%r85 = "math.cttz"(%ix) : (index) -> index
// CHECK-NEXT:    %98 = "math.cttz"(%12) : (index) -> index
%r86 = "math.ctpop"(%i) : (i32) -> i32
// CHECK-NEXT:    %99 = "math.ctpop"(%8) : (i32) -> i32
%r87 = "math.ctpop"(%iv) : (vector<4xi64>) -> vector<4xi64>
// CHECK-NEXT:    %100 = "math.ctpop"(%10) : (vector<4xi64>) -> vector<4xi64>
%r88 = "math.ctpop"(%ix) : (index) -> index
// CHECK-NEXT:    %101 = "math.ctpop"(%12) : (index) -> index
%r89 = "math.ipowi"(%i, %j) : (i32, i32) -> i32
// CHECK-NEXT:    %102 = "math.ipowi"(%8, %9) : (i32, i32) -> i32
%r90 = "math.ipowi"(%iv, %iw) : (vector<4xi64>, vector<4xi64>) -> vector<4xi64>
// CHECK-NEXT:    %103 = "math.ipowi"(%10, %11) : (vector<4xi64>, vector<4xi64>) -> vector<4xi64>
%r91 = "math.ipowi"(%ix, %iy) : (index, index) -> index
// CHECK-NEXT:    %104 = "math.ipowi"(%12, %13) : (index, index) -> index
%r92 = "math.isfinite"(%f) : (f32) -> i1
// CHECK-NEXT:    %105 = "math.isfinite"(%0) <{fastmath = #arith.fastmath<none>}> : (f32) -> i1
%r93 = "math.isfinite"(%v) <{fastmath = #arith.fastmath<ninf>}> : (vector<4xf32>) -> vector<4xi1>
// CHECK-NEXT:    %106 = "math.isfinite"(%3) <{fastmath = #arith.fastmath<ninf>}> : (vector<4xf32>) -> vector<4xi1>
%r94 = "math.isfinite"(%t) : (tensor<2x3xf64>) -> tensor<2x3xi1>
// CHECK-NEXT:    %107 = "math.isfinite"(%6) <{fastmath = #arith.fastmath<none>}> : (tensor<2x3xf64>) -> tensor<2x3xi1>
%r95 = "math.isinf"(%f) : (f32) -> i1
// CHECK-NEXT:    %108 = "math.isinf"(%0) <{fastmath = #arith.fastmath<none>}> : (f32) -> i1
%r96 = "math.isinf"(%v) <{fastmath = #arith.fastmath<ninf>}> : (vector<4xf32>) -> vector<4xi1>
// CHECK-NEXT:    %109 = "math.isinf"(%3) <{fastmath = #arith.fastmath<ninf>}> : (vector<4xf32>) -> vector<4xi1>
%r97 = "math.isinf"(%t) : (tensor<2x3xf64>) -> tensor<2x3xi1>
// CHECK-NEXT:    %110 = "math.isinf"(%6) <{fastmath = #arith.fastmath<none>}> : (tensor<2x3xf64>) -> tensor<2x3xi1>
%r98 = "math.isnan"(%f) : (f32) -> i1
// CHECK-NEXT:    %111 = "math.isnan"(%0) <{fastmath = #arith.fastmath<none>}> : (f32) -> i1
%r99 = "math.isnan"(%v) <{fastmath = #arith.fastmath<ninf>}> : (vector<4xf32>) -> vector<4xi1>
// CHECK-NEXT:    %112 = "math.isnan"(%3) <{fastmath = #arith.fastmath<ninf>}> : (vector<4xf32>) -> vector<4xi1>
%r100 = "math.isnan"(%t) : (tensor<2x3xf64>) -> tensor<2x3xi1>
// CHECK-NEXT:    %113 = "math.isnan"(%6) <{fastmath = #arith.fastmath<none>}> : (tensor<2x3xf64>) -> tensor<2x3xi1>
%r101 = "math.isnormal"(%f) : (f32) -> i1
// CHECK-NEXT:    %114 = "math.isnormal"(%0) <{fastmath = #arith.fastmath<none>}> : (f32) -> i1
%r102 = "math.isnormal"(%v) <{fastmath = #arith.fastmath<ninf>}> : (vector<4xf32>) -> vector<4xi1>
// CHECK-NEXT:    %115 = "math.isnormal"(%3) <{fastmath = #arith.fastmath<ninf>}> : (vector<4xf32>) -> vector<4xi1>
%r103 = "math.isnormal"(%t) : (tensor<2x3xf64>) -> tensor<2x3xi1>
// CHECK-NEXT:    %116 = "math.isnormal"(%6) <{fastmath = #arith.fastmath<none>}> : (tensor<2x3xf64>) -> tensor<2x3xi1>
%s, %c = "math.sincos"(%f) : (f32) -> (f32, f32)
// CHECK-NEXT:    %117, %118 = "math.sincos"(%0) <{fastmath = #arith.fastmath<none>}> : (f32) -> (f32, f32)
%sv, %cv = "math.sincos"(%v) <{fastmath = #arith.fastmath<afn>}> : (vector<4xf32>) -> (vector<4xf32>, vector<4xf32>)
// CHECK-NEXT:    %119, %120 = "math.sincos"(%3) <{fastmath = #arith.fastmath<afn>}> : (vector<4xf32>) -> (vector<4xf32>, vector<4xf32>)
%st, %ct = "math.sincos"(%t) : (tensor<2x3xf64>) -> (tensor<2x3xf64>, tensor<2x3xf64>)
// CHECK-NEXT:    %121, %122 = "math.sincos"(%6) <{fastmath = #arith.fastmath<none>}> : (tensor<2x3xf64>) -> (tensor<2x3xf64>, tensor<2x3xf64>)
