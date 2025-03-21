// RUN: scair-opt %s | filecheck %s

"builtin.module"() ({
    %0 = "test.op"() {hello = "world", "quoted" = i3298} : () -> (!cmathv2.complex<f32>)
    %1 = "cmathv2.normv2"(%0) : (!cmathv2.complex<f32>) -> (i32)
}) : () -> ()

// CHECK: builtin.module {
// CHECK:   %0 = "test.op"() {hello = "world", quoted = i3298} : () -> (!cmathV2.complex<[f32]>)
// CHECK:   %1 = "cmathv2.normv2"(%0) : (!cmathV2.complex<[f32]>) -> (i32)
// CHECK: }