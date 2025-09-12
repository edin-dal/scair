func.func @sample() -> i32 {
  %0 = "samplemath.constant"() <{in = 5 : i32}> : () -> (i32)
  %1 = "samplemath.constant"() <{in = 5 : i32}> : () -> (i32)
  %2 = "samplemath.mul"(%0, %1) : (i32, i32) -> (i32)
  func.return %2 : i32
}