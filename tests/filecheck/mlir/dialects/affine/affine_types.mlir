// RUN: scair-opt %s | mlir-opt --mlir-print-op-generic --mlir-print-local-scope | scair-opt | filecheck %s

"builtin.module"() ({
    %0 = "test.op"() {hello = "world", "quoted" = i3298} : () -> (i32)
    "test.op"() {hello = affine_map<(d0, d1)[s0] -> (d0 + d1 + s0, d1 + s0)>} : () -> ()
    "test.op"() {hello = affine_map<(d0, d1, d2)[s0, s1] -> (d0 + d1 + s0, d1 + s0 + 4, 2 * d2)>} : () -> ()
    "test.op"() {hello = affine_set<(d0, d1)[s0] : (d0 + d1 + s0 >= d1 + s0)>} : () -> ()
    "test.op"() {hello = affine_set<(d0, d1, d2)[s0, s1, s2] : (d0 == s0, d1 <= s1, d2 >= s2)>} : () -> ()
    "test.op"(%0) : (i32) -> ()
}) : () -> ()

// CHECK:       builtin.module {
// CHECK-NEXT:    %0 = "test.op"() {hello = "world", quoted = i3298} : () -> (i32)
// CHECK-NEXT:    "test.op"() {hello = affine_map<(d0, d1)[s0] -> (d0 + d1 + s0, d1 + s0)>} : () -> ()
// CHECK-NEXT:    "test.op"() {hello = affine_map<(d0, d1, d2)[s0, s1] -> (d0 + d1 + s0, d1 + s0 + 4, d2 * 2)>} : () -> ()
// CHECK-NEXT:    "test.op"() {hello = affine_set<(d0, d1)[s0]: (d0 + d1 + s0 - d1 + s0 >= 0)>} : () -> ()
// CHECK-NEXT:    "test.op"() {hello = affine_set<(d0, d1, d2)[s0, s1, s2]: (d0 - s0 == 0, d1 + s1 >= 0, d2 - s2 >= 0)>} : () -> ()
// CHECK-NEXT:    "test.op"(%0) : (i32) -> ()
// CHECK-NEXT:  