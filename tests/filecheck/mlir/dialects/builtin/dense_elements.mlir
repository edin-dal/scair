// RUN: scair-opt %s | mlir-opt --allow-unregistered-dialect --mlir-print-op-generic | scair-opt | filecheck %s

"test.op"() {value = dense<7> : tensor<4xi8>} : () -> ()
"test.op"() {value = dense<[1, 2]> : tensor<2xi64>} : () -> ()
"test.op"() {value = dense<[true, false]> : tensor<2xi1>} : () -> ()
"test.op"() {value = dense<[3, 4]> : tensor<2xindex>} : () -> ()
"test.op"() {value = dense<[1.0, 2.0]> : tensor<2xf16>} : () -> ()
"test.op"() {value = dense<[1.5, 2.25]> : tensor<2xf64>} : () -> ()
"test.op"() {value = dense<[1.0, 2.0]> : tensor<2xf128>} : () -> ()
"test.op"() {value = dense<[[1, 2], [3, 4]]> : tensor<2x2xi16>} : () -> ()
"test.op"() {value = dense<[[1.0, 2.0], [3.0, 4.0]]> : vector<2x2xf32>} : () -> ()
"test.op"() {value = dense<> : tensor<0xi32>} : () -> ()
"test.op"() {value = dense<9> : tensor<i32>} : () -> ()

// CHECK:      builtin.module {
// CHECK-NEXT:   "test.op"() {value = dense<7> : tensor<4xi8>} : () -> ()
// CHECK-NEXT:   "test.op"() {value = dense<[1, 2]> : tensor<2xi64>} : () -> ()
// CHECK-NEXT:   "test.op"() {value = dense<[true, false]> : tensor<2xi1>} : () -> ()
// CHECK-NEXT:   "test.op"() {value = dense<[3, 4]> : tensor<2xindex>} : () -> ()
// CHECK-NEXT:   "test.op"() {value = dense<[1.0, 2.0]> : tensor<2xf16>} : () -> ()
// CHECK-NEXT:   "test.op"() {value = dense<[1.5, 2.25]> : tensor<2xf64>} : () -> ()
// CHECK-NEXT:   "test.op"() {value = dense<[1.0, 2.0]> : tensor<2xf128>} : () -> ()
// CHECK-NEXT:   "test.op"() {value = dense<[[1, 2], [3, 4]]> : tensor<2x2xi16>} : () -> ()
// CHECK-NEXT:   "test.op"() {value = dense<[[1.0, 2.0], [3.0, 4.0]]> : vector<2x2xf32>} : () -> ()
// CHECK-NEXT:   "test.op"() {value = dense<> : tensor<0xi32>} : () -> ()
// CHECK-NEXT:   "test.op"() {value = dense<9> : tensor<i32>} : () -> ()
// CHECK-NEXT: }
