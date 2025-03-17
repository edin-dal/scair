// RUN: scair-opt %s | filecheck %s

"builtin.module"() ({
    %0 = "test.op"() {hello = "world", "quoted" = i3298} : () -> (i32)
    %1 = "cmathV2.normV2"(%0) : (i32) -> (i32)
}) : () -> ()

// CHECK: builtin.module {
// CHECK:   %0 = "test.op"() {hello = "world", quoted = i3298} : () -> (i32)
// CHECK:   %1 = "cmathV2.normV2"(%0) : (i32) -> (i32)
// CHECK: }