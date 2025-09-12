%0 = "samplemath.constant"() <{in = 5 : i32}> : () -> (i32)
%1 = "samplemath.constant"() <{in = 5 : i32}> : () -> (i32)
%2 = "samplemath.mul"(%0, %1) : (i32, i32) -> (i32)
"test.op"(%2) : (i32) -> () 