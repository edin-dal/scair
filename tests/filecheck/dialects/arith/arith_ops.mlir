// RUN: scair-opt %s | filecheck %s

%lhsi1, %rhsi1 = "test.op"() : () -> (i1, i1)
%lhsi32, %rhsi32 = "test.op"() : () -> (i32, i32)
%lhsi64, %rhsi64 = "test.op"() : () -> (i64, i64)
%lhsindex, %rhsindex = "test.op"() : () -> (index, index)
%lhsf32, %rhsf32 = "test.op"() : () -> (f32, f32)
%lhsf64, %rhsf64 = "test.op"() : () -> (f64, f64)

%divsi = "arith.divsi"(%lhsi32, %rhsi32) : (i32, i32) -> i32
// CHECK:    %12 = "arith.divsi"(%2, %3) : (i32, i32) -> (i32)
%divsi_index = "arith.divsi"(%lhsindex, %rhsindex) : (index, index) -> index
// CHECK:    %13 = "arith.divsi"(%6, %7) : (index, index) -> (index)
%divui = "arith.divui"(%lhsi32, %rhsi32) : (i32, i32) -> i32
// CHECK:    %14 = "arith.divui"(%2, %3) : (i32, i32) -> (i32)
%divui_index = "arith.divui"(%lhsindex, %rhsindex) : (index, index) -> index
// CHECK:    %15 = "arith.divui"(%6, %7) : (index, index) -> (index)
%ceildivsi = "arith.ceildivsi"(%lhsi32, %rhsi32) : (i32, i32) -> i32
// CHECK:    %16 = "arith.ceildivsi"(%2, %3) : (i32, i32) -> (i32)
%ceildivsi_index = "arith.ceildivsi"(%lhsindex, %rhsindex) : (index, index) -> index
// CHECK:    %17 = "arith.ceildivsi"(%6, %7) : (index, index) -> (index)
%floordivsi = "arith.floordivsi"(%lhsi32, %rhsi32) : (i32, i32) -> i32
// CHECK:    %18 = "arith.floordivsi"(%2, %3) : (i32, i32) -> (i32)
%floordivsi_index = "arith.floordivsi"(%lhsindex, %rhsindex) : (index, index) -> index
// CHECK:    %19 = "arith.floordivsi"(%6, %7) : (index, index) -> (index)
%ceildivui = "arith.ceildivui"(%lhsi32, %rhsi32) : (i32, i32) -> i32
// CHECK:    %20 = "arith.ceildivui"(%2, %3) : (i32, i32) -> (i32)
%ceildivui_index = "arith.ceildivui"(%lhsindex, %rhsindex) : (index, index) -> index
// CHECK:    %21 = "arith.ceildivui"(%6, %7) : (index, index) -> (index)
%remsi = "arith.remsi"(%lhsi32, %rhsi32) : (i32, i32) -> i32
// CHECK:    %22 = "arith.remsi"(%2, %3) : (i32, i32) -> (i32)
%remui = "arith.remui"(%lhsi32, %rhsi32) : (i32, i32) -> i32
// CHECK:    %23 = "arith.remui"(%2, %3) : (i32, i32) -> (i32)
%remui_index = "arith.remui"(%lhsindex, %rhsindex) : (index, index) -> index
// CHECK:    %24 = "arith.remui"(%6, %7) : (index, index) -> (index)
%cmpi = "arith.cmpi"(%lhsi32, %rhsi32) <{"predicate" = 2 : i64}> : (i32, i32) -> i1
// CHECK:    %25 = "arith.cmpi"(%2, %3) <{predicate = 2}> : (i32, i32) -> (i1)
%cmpi_index = "arith.cmpi"(%lhsindex, %rhsindex) <{"predicate" = 2 : i64}> : (index, index) -> i1
// CHECK:    %26 = "arith.cmpi"(%6, %7) <{predicate = 2}> : (index, index) -> (i1)
%addi = "arith.addi"(%lhsi32, %rhsi32) {"hello" = "world"} : (i32, i32) -> i32
// CHECK:    %27 = "arith.addi"(%2, %3) {hello = "world"} : (i32, i32) -> (i32)
%addf = "arith.addf"(%lhsf32, %rhsf32) <{"fastmath" = #arith.fastmath<none>}> : (f32, f32) -> f32
// CHECK:    %28 = "arith.addf"(%8, %9) <{fastmath = arith.fastmath<none>}> : (f32, f32) -> (f32)

%addf_vector = "arith.addf"(%lhsf32, %rhsf32) <{"fastmath" = #arith.fastmath<none>}> : (f32, f32) -> f32
// CHECK:    %29 = "arith.addf"(%8, %9) <{fastmath = arith.fastmath<none>}> : (f32, f32) -> (f32)
%mulf = "arith.mulf"(%lhsf32, %rhsf32) <{"fastmath" = #arith.fastmath<none>}> : (f32, f32) -> f32
// CHECK:    %30 = "arith.mulf"(%8, %9) <{fastmath = arith.fastmath<none>}> : (f32, f32) -> (f32)
%mulf_vector = "arith.mulf"(%lhsf32, %rhsf32) <{"fastmath" = #arith.fastmath<none>}> : (f32, f32) -> f32
// CHECK:    %31 = "arith.mulf"(%8, %9) <{fastmath = arith.fastmath<none>}> : (f32, f32) -> (f32)
%divf = "arith.divf"(%lhsf32, %rhsf32) <{"fastmath" = #arith.fastmath<none>}> : (f32, f32) -> f32
// CHECK:    %32 = "arith.divf"(%8, %9) <{fastmath = arith.fastmath<none>}> : (f32, f32) -> (f32)
%divf_vector = "arith.divf"(%lhsf32, %rhsf32) <{"fastmath" = #arith.fastmath<none>}> : (f32, f32) -> f32
// CHECK:    %33 = "arith.divf"(%8, %9) <{fastmath = arith.fastmath<none>}> : (f32, f32) -> (f32)
%faddf = "arith.addf"(%lhsf32, %rhsf32) <{"fastmath" = #arith.fastmath<fast>}> : (f32, f32) -> f32
// CHECK:    %34 = "arith.addf"(%8, %9) <{fastmath = arith.fastmath<fast>}> : (f32, f32) -> (f32)
%faddf_vector = "arith.addf"(%lhsf32, %rhsf32) <{"fastmath" = #arith.fastmath<fast>}> : (f32, f32) -> f32
// CHECK:    %35 = "arith.addf"(%8, %9) <{fastmath = arith.fastmath<fast>}> : (f32, f32) -> (f32)
%index = "arith.index_cast"(%lhsi32) : (i32) -> index
// CHECK:    %36 = "arith.index_cast"(%2) : (i32) -> (index)
