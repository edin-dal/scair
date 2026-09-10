// RUN: scair-opt %s --parsing-diagnostics --split-input-file | filecheck %s

"test.op"() {value = dense<[1, 2, 3, 4]> : tensor<2x2xi32>} : () -> ()

// CHECK: inferred shape of elements literal (List(4)) does not match type (List(2, 2))

// -----

"test.op"() {value = dense<[[1, 2], [3]]> : tensor<2x2xi32>} : () -> ()

// CHECK: tensor literal ranks are not consistent between elements

// -----

"test.op"() {value = dense<[1, 2.0]> : tensor<2xi32>} : () -> ()

// CHECK: tensor literal mixes integer and floating-point elements

// -----

"test.op"() {value = dense<[1, 2]> : tensor<?xi32>} : () -> ()

// CHECK: dense elements attribute requires a statically shaped type

// -----

"test.op"() {value = dense<> : tensor<2xi32>} : () -> ()

// CHECK: Dense elements attribute has 0 values, but type tensor<2xi32> has 2 elements
