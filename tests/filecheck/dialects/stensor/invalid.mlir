// RUN: scair-opt %s --verify_diagnostics --split_input_file | filecheck %s

builtin.module {
    %0 = "test.op"() : () -> !stensor.stensor<f32, [-1, -1], #isl.map<{ [i] -> [0] : i < 10 }>, #isl.map<{ [i, j] -> [i', j'] : i' = i and j' = j and i = j}>>
}
// CHECK: The known map domain must match dimensionality of the tensor, got 1 instead of 2

// -----

builtin.module {
    %0 = "test.op"() : () -> !stensor.stensor<f32, [-1, -1], #isl.map<{ [i, j] -> [0] : i < j }>, #isl.map<{ [i] -> [i', j'] : i' = i and i' = j'}>>
}
// CHECK: The redundant map domain must match dimensionality of the tensor, got 1 instead of 2

// -----

builtin.module {
    %0 = "test.op"() : () -> !stensor.stensor<f32, [-1, -1], #isl.map<{ [i, j] -> [0] : i < j }>, #isl.map<{ [i, j] -> [j'] : j' = j and i = j}>>
}
// CHECK: The redundant map must match dimensionality of the tensor, got 1 instead of 2

// -----

builtin.module {
    %0 = "test.op"() : () -> !stensor.stensor<f32, [-1, -1], #isl.map<{ [i, j] -> [j] : i < j }>, #isl.map<{ [i, j] -> [i', j'] : i' = i and j' = j and i = j}>>
}
// CHECK: The known map must map to single values, got the range [-infty, infty]

// -----

builtin.module {
    %0 = "test.op"() : () -> !stensor.stensor<f32, [-1, -1], #isl.map<{ [i, j] -> [0, 2] : i < j }>, #isl.map<{ [i, j] -> [i', j'] : i' = i and j' = j and i = j}>>
}
// CHECK: The known map must be of dimension 1, got 2

