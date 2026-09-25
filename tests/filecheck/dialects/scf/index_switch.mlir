// RUN: scair-opt %s | scair-opt -g | scair-opt | filecheck %s

%arg = "test.op"() : () -> index

// Default region only
"scf.index_switch"(%arg) <{cases = array<i64>}> ({
  scf.yield
}) : (index) -> ()

// CHECK:       builtin.module {
// CHECK-NEXT:    %0 = "test.op"() : () -> index
// CHECK-NEXT:    "scf.index_switch"(%0) <{cases = array<i64>}> ({
// CHECK-NEXT:      scf.yield
// CHECK-NEXT:    }) : (index) -> ()

// Default region and several case regions, with results
%res:2 = "scf.index_switch"(%arg) <{cases = array<i64: 2, 5>}> ({
  %d0, %d1 = "test.op"() : () -> (i32, f32)
  scf.yield %d0, %d1 : i32, f32
}, {
  %a0, %a1 = "test.op"() : () -> (i32, f32)
  scf.yield %a0, %a1 : i32, f32
}, {
  %b0, %b1 = "test.op"() : () -> (i32, f32)
  scf.yield %b0, %b1 : i32, f32
}) : (index) -> (i32, f32)

// CHECK-NEXT:    %1, %2 = "scf.index_switch"(%0) <{cases = array<i64: 2, 5>}> ({
// CHECK-NEXT:      %3, %4 = "test.op"() : () -> (i32, f32)
// CHECK-NEXT:      scf.yield %3, %4 : i32, f32
// CHECK-NEXT:    }, {
// CHECK-NEXT:      %3, %4 = "test.op"() : () -> (i32, f32)
// CHECK-NEXT:      scf.yield %3, %4 : i32, f32
// CHECK-NEXT:    }, {
// CHECK-NEXT:      %3, %4 = "test.op"() : () -> (i32, f32)
// CHECK-NEXT:      scf.yield %3, %4 : i32, f32
// CHECK-NEXT:    }) : (index) -> (i32, f32)
// CHECK-NEXT:  }
