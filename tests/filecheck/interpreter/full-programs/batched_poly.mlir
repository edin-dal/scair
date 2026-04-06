// RUN: scair-run %s | filecheck %s

builtin.module {
  func.func @main() -> (i64) {

    %x0 = "arith.constant"() <{value = 4 : i64}>  : () -> i64
    %x1 = "arith.constant"() <{value = 1 : i64}>  : () -> i64
    %x2 = "arith.constant"() <{value = 2 : i64}>  : () -> i64
    %x3 = "arith.constant"() <{value = 3 : i64}>  : () -> i64

    %c3 = "arith.constant"() <{value = 3 : i64}>  : () -> i64
    %c2 = "arith.constant"() <{value = 2 : i64}>  : () -> i64
    %c5 = "arith.constant"() <{value = 5 : i64}>  : () -> i64
    %c7 = "arith.constant"() <{value = 7 : i64}>  : () -> i64

    %k10   = "arith.constant"() <{value = 10 : i64}>   : () -> i64
    %k100  = "arith.constant"() <{value = 100 : i64}>  : () -> i64
    %k1000 = "arith.constant"() <{value = 1000 : i64}> : () -> i64

    // ---- p(x0) ----
    %x0_sq   = "arith.muli"(%x0, %x0) <{overflowFlags = #arith.overflow<none>}> : (i64, i64) -> i64
    %x0_cube = "arith.muli"(%x0_sq, %x0) <{overflowFlags = #arith.overflow<none>}> : (i64, i64) -> i64
    %t0_1    = "arith.muli"(%c3, %x0_cube) <{overflowFlags = #arith.overflow<none>}> : (i64, i64) -> i64
    %t0_2    = "arith.muli"(%c2, %x0_sq)   <{overflowFlags = #arith.overflow<none>}> : (i64, i64) -> i64
    %t0_3    = "arith.muli"(%c5, %x0)      <{overflowFlags = #arith.overflow<none>}> : (i64, i64) -> i64
    %t0_s1   = "arith.addi"(%t0_1, %t0_2)  <{overflowFlags = #arith.overflow<none>}> : (i64, i64) -> i64
    %t0_s2   = "arith.subi"(%t0_s1, %t0_3) <{overflowFlags = #arith.overflow<none>}> : (i64, i64) -> i64
    %p0      = "arith.addi"(%t0_s2, %c7)   <{overflowFlags = #arith.overflow<none>}> : (i64, i64) -> i64

    // ---- p(x1) ----
    %x1_sq   = "arith.muli"(%x1, %x1) <{overflowFlags = #arith.overflow<none>}> : (i64, i64) -> i64
    %x1_cube = "arith.muli"(%x1_sq, %x1) <{overflowFlags = #arith.overflow<none>}> : (i64, i64) -> i64
    %t1_1    = "arith.muli"(%c3, %x1_cube) <{overflowFlags = #arith.overflow<none>}> : (i64, i64) -> i64
    %t1_2    = "arith.muli"(%c2, %x1_sq)   <{overflowFlags = #arith.overflow<none>}> : (i64, i64) -> i64
    %t1_3    = "arith.muli"(%c5, %x1)      <{overflowFlags = #arith.overflow<none>}> : (i64, i64) -> i64
    %t1_s1   = "arith.addi"(%t1_1, %t1_2)  <{overflowFlags = #arith.overflow<none>}> : (i64, i64) -> i64
    %t1_s2   = "arith.subi"(%t1_s1, %t1_3) <{overflowFlags = #arith.overflow<none>}> : (i64, i64) -> i64
    %p1      = "arith.addi"(%t1_s2, %c7)   <{overflowFlags = #arith.overflow<none>}> : (i64, i64) -> i64

    // ---- p(x2) ----
    %x2_sq   = "arith.muli"(%x2, %x2) <{overflowFlags = #arith.overflow<none>}> : (i64, i64) -> i64
    %x2_cube = "arith.muli"(%x2_sq, %x2) <{overflowFlags = #arith.overflow<none>}> : (i64, i64) -> i64
    %t2_1    = "arith.muli"(%c3, %x2_cube) <{overflowFlags = #arith.overflow<none>}> : (i64, i64) -> i64
    %t2_2    = "arith.muli"(%c2, %x2_sq)   <{overflowFlags = #arith.overflow<none>}> : (i64, i64) -> i64
    %t2_3    = "arith.muli"(%c5, %x2)      <{overflowFlags = #arith.overflow<none>}> : (i64, i64) -> i64
    %t2_s1   = "arith.addi"(%t2_1, %t2_2)  <{overflowFlags = #arith.overflow<none>}> : (i64, i64) -> i64
    %t2_s2   = "arith.subi"(%t2_s1, %t2_3) <{overflowFlags = #arith.overflow<none>}> : (i64, i64) -> i64
    %p2      = "arith.addi"(%t2_s2, %c7)   <{overflowFlags = #arith.overflow<none>}> : (i64, i64) -> i64

    // ---- p(x3) ----
    %x3_sq   = "arith.muli"(%x3, %x3) <{overflowFlags = #arith.overflow<none>}> : (i64, i64) -> i64
    %x3_cube = "arith.muli"(%x3_sq, %x3) <{overflowFlags = #arith.overflow<none>}> : (i64, i64) -> i64
    %t3_1    = "arith.muli"(%c3, %x3_cube) <{overflowFlags = #arith.overflow<none>}> : (i64, i64) -> i64
    %t3_2    = "arith.muli"(%c2, %x3_sq)   <{overflowFlags = #arith.overflow<none>}> : (i64, i64) -> i64
    %t3_3    = "arith.muli"(%c5, %x3)      <{overflowFlags = #arith.overflow<none>}> : (i64, i64) -> i64
    %t3_s1   = "arith.addi"(%t3_1, %t3_2)  <{overflowFlags = #arith.overflow<none>}> : (i64, i64) -> i64
    %t3_s2   = "arith.subi"(%t3_s1, %t3_3) <{overflowFlags = #arith.overflow<none>}> : (i64, i64) -> i64
    %p3      = "arith.addi"(%t3_s2, %c7)   <{overflowFlags = #arith.overflow<none>}> : (i64, i64) -> i64

    // Pack results into one return value.
    %p1s   = "arith.muli"(%p1, %k10)   <{overflowFlags = #arith.overflow<none>}> : (i64, i64) -> i64
    %p2s   = "arith.muli"(%p2, %k100)  <{overflowFlags = #arith.overflow<none>}> : (i64, i64) -> i64
    %p3s   = "arith.muli"(%p3, %k1000) <{overflowFlags = #arith.overflow<none>}> : (i64, i64) -> i64
    %s01   = "arith.addi"(%p0, %p1s)   <{overflowFlags = #arith.overflow<none>}> : (i64, i64) -> i64
    %s012  = "arith.addi"(%s01, %p2s)  <{overflowFlags = #arith.overflow<none>}> : (i64, i64) -> i64
    %batch = "arith.addi"(%s012, %p3s) <{overflowFlags = #arith.overflow<none>}> : (i64, i64) -> i64

    func.return %batch : i64
  }
}

// Expected values (for reference):
// p(4)=211, p(1)=11, p(2)=33, p(3)=85
// batch = 211 + 10*11 + 100*33 + 1000*85 = 88621

// CHECK: Result: 88621