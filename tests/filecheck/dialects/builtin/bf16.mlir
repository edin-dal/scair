// RUN: scair-opt %s | filecheck %s

%0 = "test.op"() {value = 1.5 : bf16} : () -> bf16
%1 = "test.op"(%0) : (bf16) -> tensor<2xbf16>
%2 = "test.op"() : () -> complex<bf16>
"test.op"() {value = dense<[1.0, 2.0]> : tensor<2xbf16>} : () -> ()
"test.op"() {type = memref<4xbf16>} : () -> ()

// CHECK:      builtin.module {
// CHECK-NEXT:   %0 = "test.op"() {value = 1.5 : bf16} : () -> bf16
// CHECK-NEXT:   %1 = "test.op"(%0) : (bf16) -> tensor<2xbf16>
// CHECK-NEXT:   %2 = "test.op"() : () -> complex<bf16>
// CHECK-NEXT:   "test.op"() {value = dense<[1.0, 2.0]> : tensor<2xbf16>} : () -> ()
// CHECK-NEXT:   "test.op"() {type = memref<4xbf16>} : () -> ()
// CHECK-NEXT: }
