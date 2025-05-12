// RUN: scair-opt %s 

builtin.module {
    %0 = "test.op"() : () -> !stensor.stensor<f32, [-1, -1], #isl.map<{ [i, j] -> [0] : i < j }>, #isl.map<{ [i, j] -> [i', j'] : i' = i and j' = j and i = j}>>
}