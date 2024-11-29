// RUN: scair-opt %s | filecheck %s

"builtin.module"() ({
    %0 = "test.op1"() {hello = "world", "quoted" = i3298} : () -> (i32)
    "test.op1"() {hello = affine_map<(d0, d1)[s0] -> (d0 + d1 + s0, d1 + s0)>} : () -> ()
    "test.op2"() {hello = affine_map<(d0, d1, d2)[s0, s1] -> (d0 + d1 + s0, d1 + s0 + 4, 2 * d2)>} : () -> ()
    "test.op3"() {hello = affine_set<(d0, d1)[s0] : (d0 + d1 + s0 >= d1 + s0)>} : () -> ()
    "test.op4"() {hello = affine_set<(d0, d1, d2)[s0, s1, s2] : (d0 == s0, d1 <= s1, d2 >= s2)>} : () -> ()
    "test.op7"(%0) : (i32) -> ()
}) : () -> ()

// CHECK: builtin.module {
// CHECK: ^bb0():
// CHECK:   %0 = "test.op1"() {hello = "world", quoted = i3298} : () -> (i32)
// CHECK:   "test.op1"() {hello = affine_map<(d0, d1)[s0] -> (d0 + d1 + s0, d1 + s0)>} : () -> ()
// CHECK:   "test.op2"() {hello = affine_map<(d0, d1, d2)[s0, s1] -> (d0 + d1 + s0, d1 + s0 + 4, 2 * d2)>} : () -> ()
// CHECK:   "test.op3"() {hello = affine_set<(d0, d1)[s0]: (d0 + d1 + s0 >= d1 + s0)>} : () -> ()
// CHECK:   "test.op4"() {hello = affine_set<(d0, d1, d2)[s0, s1, s2]: (d0 == s0, d1 <= s1, d2 >= s2)>} : () -> ()
// CHECK:   "test.op7"(%0) : (i32) -> ()
// CHECK: }