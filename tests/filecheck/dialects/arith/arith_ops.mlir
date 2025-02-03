// RUN: scair-opt %s 

%lhsi32, %rhsi32, %pred32 = "test.op"() : () -> (i32, i32, i32)


%remsi2 = arith.remsi %lhsi32 %rhsi32 i32 i32 i32
// CHECK-NEXT:    %16 = "arith.remsi"(%2, %3) : (i32, i32) -> (i32)
