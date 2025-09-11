"test.op"() ({
^bb0(%1 : !cmath.complex<f32>, %3 : !cmath.complex<f64>, %4 : !cmath.complex<f32>):
  "test.op"() : () -> ()
  %0 = "cmath.norm"(%1) : (!cmath.complex<f32>) -> (f32)
  %2 = "cmath.mul"(%3, %4) : (!cmath.complex<f64>, !cmath.complex<f32>) -> (!cmath.complex<f32>)
}) : () -> ()