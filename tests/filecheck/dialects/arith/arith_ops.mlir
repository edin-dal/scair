// RUN: scair-opt %s | filecheck %s

%lhsi1, %rhsi1 = "test.op"() : () -> (i1, i1)
%lhsi32, %rhsi32 = "test.op"() : () -> (i32, i32)
%lhsi64, %rhsi64 = "test.op"() : () -> (i64, i64)
%lhsindex, %rhsindex = "test.op"() : () -> (index, index)
%lhsf32, %rhsf32 = "test.op"() : () -> (f32, f32)
%lhsf64, %rhsf64 = "test.op"() : () -> (f64, f64)

%divsi = "arith.divsi"(%lhsi32, %rhsi32) : (i32, i32) -> i32
// CHECK:         %12 = "arith.divsi"(%2, %3) : (i32, i32) -> i32
%divsi_index = "arith.divsi"(%lhsindex, %rhsindex) : (index, index) -> index
// CHECK-NEXT:    %13 = "arith.divsi"(%6, %7) : (index, index) -> index
%divui = "arith.divui"(%lhsi32, %rhsi32) : (i32, i32) -> i32
// CHECK-NEXT:    %14 = "arith.divui"(%2, %3) : (i32, i32) -> i32
%divui_index = "arith.divui"(%lhsindex, %rhsindex) : (index, index) -> index
// CHECK-NEXT:    %15 = "arith.divui"(%6, %7) : (index, index) -> index
%remsi = "arith.remsi"(%lhsi32, %rhsi32) : (i32, i32) -> i32
// CHECK-NEXT:    %16 = "arith.remsi"(%2, %3) : (i32, i32) -> i32
%remui = "arith.remui"(%lhsi32, %rhsi32) : (i32, i32) -> i32
// CHECK-NEXT:    %17 = "arith.remui"(%2, %3) : (i32, i32) -> i32
%remui_index = "arith.remui"(%lhsindex, %rhsindex) : (index, index) -> index
// CHECK-NEXT:    %18 = "arith.remui"(%6, %7) : (index, index) -> index
%cmpi = "arith.cmpi"(%lhsi32, %rhsi32) <{"predicate" = 2 : i64}> : (i32, i32) -> i1
// CHECK-NEXT:    %19 = "arith.cmpi"(%2, %3) <{predicate = 2}> : (i32, i32) -> i1
%cmpi_index = "arith.cmpi"(%lhsindex, %rhsindex) <{"predicate" = 2 : i64}> : (index, index) -> i1
// CHECK-NEXT:    %20 = "arith.cmpi"(%6, %7) <{predicate = 2}> : (index, index) -> i1
%addi = "arith.addi"(%lhsi32, %rhsi32) {"hello" = "world"} : (i32, i32) -> i32
// CHECK-NEXT:    %21 = "arith.addi"(%2, %3) {hello = "world"} : (i32, i32) -> i32
%addf = "arith.addf"(%lhsf64, %rhsf64) <{"fastmath" = #arith.fastmath<none>}> : (f64, f64) -> f64
// CHECK-NEXT:    %22 = "arith.addf"(%10, %11) <{fastmath = #arith.fastmath<none>}> : (f64, f64) -> f64

%addf_vector = "arith.addf"(%lhsf32, %rhsf32) <{"fastmath" = #arith.fastmath<none>}> : (f32, f32) -> f32
// CHECK-NEXT:    %23 = "arith.addf"(%8, %9) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
%mulf = "arith.mulf"(%lhsf32, %rhsf32) <{"fastmath" = #arith.fastmath<none>}> : (f32, f32) -> f32
// CHECK-NEXT:    %24 = "arith.mulf"(%8, %9) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
%mulf_vector = "arith.mulf"(%lhsf32, %rhsf32) <{"fastmath" = #arith.fastmath<none>}> : (f32, f32) -> f32
// CHECK-NEXT:    %25 = "arith.mulf"(%8, %9) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
%divf = "arith.divf"(%lhsf32, %rhsf32) <{"fastmath" = #arith.fastmath<none>}> : (f32, f32) -> f32
// CHECK-NEXT:    %26 = "arith.divf"(%8, %9) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
%divf_vector = "arith.divf"(%lhsf32, %rhsf32) <{"fastmath" = #arith.fastmath<none>}> : (f32, f32) -> f32
// CHECK-NEXT:    %27 = "arith.divf"(%8, %9) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
%faddf = "arith.addf"(%lhsf32, %rhsf32) <{"fastmath" = #arith.fastmath<fast>}> : (f32, f32) -> f32
// CHECK-NEXT:    %28 = "arith.addf"(%8, %9) <{fastmath = #arith.fastmath<fast>}> : (f32, f32) -> f32
%faddf_vector = "arith.addf"(%lhsf32, %rhsf32) <{"fastmath" = #arith.fastmath<fast>}> : (f32, f32) -> f32
// CHECK-NEXT:    %29 = "arith.addf"(%8, %9) <{fastmath = #arith.fastmath<fast>}> : (f32, f32) -> f32
%index = "arith.index_cast"(%lhsi32) : (i32) -> index
// CHECK-NEXT:    %30 = "arith.index_cast"(%2) : (i32) -> index
%andi = "arith.andi"(%lhsi32, %rhsi32) : (i32, i32) -> i32
// CHECK-NEXT:    %31 = "arith.andi"(%2, %3) : (i32, i32) -> i32
