// RUN: scair-opt %s | filecheck %s

%lhsi1, %rhsi1 = "test.op"() : () -> (i1, i1)
%lhsi32, %rhsi32 = "test.op"() : () -> (i32, i32)
%lhsi64, %rhsi64 = "test.op"() : () -> (i64, i64)
%lhsindex, %rhsindex = "test.op"() : () -> (index, index)
%lhsf32, %rhsf32 = "test.op"() : () -> (f32, f32)
%lhsf64, %rhsf64 = "test.op"() : () -> (f64, f64)

%divsi = "arith.divsi"(%lhsi32, %rhsi32) : (i32, i32) -> i32
%divsi_index = "arith.divsi"(%lhsindex, %rhsindex) : (index, index) -> index
%divui = "arith.divui"(%lhsi32, %rhsi32) : (i32, i32) -> i32
%divui_index = "arith.divui"(%lhsindex, %rhsindex) : (index, index) -> index
%ceildivsi = "arith.ceildivsi"(%lhsi32, %rhsi32) : (i32, i32) -> i32
%ceildivsi_index = "arith.ceildivsi"(%lhsindex, %rhsindex) : (index, index) -> index
%floordivsi = "arith.floordivsi"(%lhsi32, %rhsi32) : (i32, i32) -> i32
%floordivsi_index = "arith.floordivsi"(%lhsindex, %rhsindex) : (index, index) -> index
%ceildivui = "arith.ceildivui"(%lhsi32, %rhsi32) : (i32, i32) -> i32
%ceildivui_index = "arith.ceildivui"(%lhsindex, %rhsindex) : (index, index) -> index
%remsi = "arith.remsi"(%lhsi32, %rhsi32) : (i32, i32) -> i32
%remui = "arith.remui"(%lhsi32, %rhsi32) : (i32, i32) -> i32
%remui_index = "arith.remui"(%lhsindex, %rhsindex) : (index, index) -> index
%cmpi = "arith.cmpi"(%lhsi32, %rhsi32) <{"predicate" = 2 : i64}> : (i32, i32) -> i1
%cmpi_index = "arith.cmpi"(%lhsindex, %rhsindex) <{"predicate" = 2 : i64}> : (index, index) -> i1
%addi = "arith.addi"(%lhsi32, %rhsi32) {"hello" = "world"} : (i32, i32) -> i32
%addf = "arith.addf"(%lhsf32, %rhsf32) <{"fastmath" = #arith.fastmath<none>}> : (f32, f32) -> f32
%addf_vector = "arith.addf"(%lhsvec, %rhsvec) <{"fastmath" = #arith.fastmath<none>}> : (vector<4xf32>, vector<4xf32>) -> vector<4xf32>
%mulf = "arith.mulf"(%lhsf32, %rhsf32) <{"fastmath" = #arith.fastmath<none>}> : (f32, f32) -> f32
%mulf_vector = "arith.mulf"(%lhsvec, %rhsvec) <{"fastmath" = #arith.fastmath<none>}> : (vector<4xf32>, vector<4xf32>) -> vector<4xf32>
%divf = "arith.divf"(%lhsf32, %rhsf32) <{"fastmath" = #arith.fastmath<none>}> : (f32, f32) -> f32
%divf_vector = "arith.divf"(%lhsvec, %rhsvec) <{"fastmath" = #arith.fastmath<none>}> : (vector<4xf32>, vector<4xf32>) -> vector<4xf32>
%faddf = "arith.addf"(%lhsf32, %rhsf32) <{"fastmath" = #arith.fastmath<fast>}> : (f32, f32) -> f32
%faddf_vector = "arith.addf"(%lhsvec, %rhsvec) <{"fastmath" = #arith.fastmath<fast>}> : (vector<4xf32>, vector<4xf32>) -> vector<4xf32>
%index = "arith.index_cast"(%lhsi32) : (i32) -> index