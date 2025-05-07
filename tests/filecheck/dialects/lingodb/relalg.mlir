// RUN: scair-opt %s -s | filecheck %s

%0 = relalg.basetable  {rows = 0x4156E48FC0000000 : i64, table_identifier = "lineitem"} columns: {l_comment => @lineitem::@l_comment({type = !db.string}), l_commitdate => @lineitem::@l_commitdate({type = !db.date<day>}), l_discount => @lineitem::@l_discount({type = !db.decimal<12, 2>}), l_extendedprice => @lineitem::@l_extendedprice({type = !db.decimal<12, 2>}), l_linenumber => @lineitem::@l_linenumber({type = i32}), l_linestatus => @lineitem::@l_linestatus({type = !db.char<1>}), l_orderkey => @lineitem::@l_orderkey({type = i32}), l_partkey => @lineitem::@l_partkey({type = i32}), l_quantity => @lineitem::@l_quantity({type = !db.decimal<12, 2>}), l_receiptdate => @lineitem::@l_receiptdate({type = !db.date<day>}), l_returnflag => @lineitem::@l_returnflag({type = !db.char<1>}), l_shipdate => @lineitem::@l_shipdate({type = !db.date<day>}), l_shipinstruct => @lineitem::@l_shipinstruct({type = !db.string}), l_shipmode => @lineitem::@l_shipmode({type = !db.string}), l_suppkey => @lineitem::@l_suppkey({type = i32}), l_tax => @lineitem::@l_tax({type = !db.decimal<12, 2>}), primaryKeyHashValue => @lineitem::@primaryKeyHashValue({type = index})}
%1 = relalg.selection %0 (%arg0: !tuples.tuple){
  %10 = db.constant("1998-09-02") : !db.date<day>
  %11 = tuples.getcol %arg0 @lineitem::@l_shipdate : !db.date<day>
  %12 = db.compare lte %11 : !db.date<day>, %10 : !db.date<day>
  tuples.return %12 : i1
} attributes {cost = 5930888.26171875 : f64, evaluationCost = 2.000000e+00 : f64, rows = 5930888.26171875 : f64, selectivity = 0.98828125 : f64}
%2 = relalg.map %1 computes : [@map0::@tmp_attr5({type = !db.decimal<36, 6>})] (%arg0: !tuples.tuple){
  %10 = db.constant(1 : i32) : !db.decimal<12, 2>
  %11 = tuples.getcol %arg0 @lineitem::@l_extendedprice : !db.decimal<12, 2>
  %12 = tuples.getcol %arg0 @lineitem::@l_discount : !db.decimal<12, 2>
  %13 = db.sub %10 : !db.decimal<12, 2>, %12 : !db.decimal<12, 2>
  %14 = db.mul %11 : !db.decimal<12, 2>, %13 : !db.decimal<12, 2>
  %15 = tuples.getcol %arg0 @lineitem::@l_tax : !db.decimal<12, 2>
  %16 = db.add %10 : !db.decimal<12, 2>, %15 : !db.decimal<12, 2>
  %17 = db.mul %14 : !db.decimal<24, 4>, %16 : !db.decimal<12, 2>
  tuples.return %17 : !db.decimal<36, 6>
} attributes {rows = 5930888.26171875 : f64}
%3 = relalg.map %2 computes : [@map0::@tmp_attr3({type = !db.decimal<24, 4>})] (%arg0: !tuples.tuple){
  %10 = db.constant(1 : i32) : !db.decimal<12, 2>
  %11 = tuples.getcol %arg0 @lineitem::@l_extendedprice : !db.decimal<12, 2>
  %12 = tuples.getcol %arg0 @lineitem::@l_discount : !db.decimal<12, 2>
  %13 = db.sub %10 : !db.decimal<12, 2>, %12 : !db.decimal<12, 2>
  %14 = db.mul %11 : !db.decimal<12, 2>, %13 : !db.decimal<12, 2>
  tuples.return %14 : !db.decimal<24, 4>
} attributes {rows = 5930888.26171875 : f64}
%4 = relalg.aggregation %3 [@lineitem::@l_returnflag,@lineitem::@l_linestatus] computes : [@aggr0::@tmp_attr9({type = i64}),@aggr0::@tmp_attr4({type = !db.decimal<36, 6>}),@aggr0::@tmp_attr2({type = !db.decimal<24, 4>}),@aggr0::@tmp_attr1({type = !db.decimal<12, 2>}),@aggr0::@tmp_attr0({type = !db.decimal<12, 2>}),@aggr_rw::@rw0({type = !db.decimal<12, 2>}),@aggr_rw::@rw1({type = i64}),@aggr_rw::@rw2({type = !db.decimal<12, 2>}),@aggr_rw::@rw3({type = i64}),@aggr_rw::@rw4({type = !db.decimal<12, 2>}),@aggr_rw::@rw5({type = i64})] (%arg0: !tuples.tuplestream,%arg1: !tuples.tuple){
  %10 = relalg.count %arg0
  %11 = relalg.aggrfn sum @lineitem::@l_discount %arg0 : !db.decimal<12, 2>
  %12 = relalg.aggrfn count @lineitem::@l_discount %arg0 : i64
  %13 = relalg.aggrfn sum @lineitem::@l_quantity %arg0 : !db.decimal<12, 2>
  %14 = relalg.aggrfn count @lineitem::@l_quantity %arg0 : i64
  %15 = relalg.aggrfn sum @map0::@tmp_attr5 %arg0 : !db.decimal<36, 6>
  %16 = relalg.aggrfn sum @map0::@tmp_attr3 %arg0 : !db.decimal<24, 4>
  %17 = relalg.aggrfn sum @lineitem::@l_extendedprice %arg0 : !db.decimal<12, 2>
  %18 = relalg.aggrfn count @lineitem::@l_extendedprice %arg0 : i64
  tuples.return %10, %15, %16, %17, %13, %11, %12, %13, %14, %17, %18 : i64, !db.decimal<36, 6>, !db.decimal<24, 4>, !db.decimal<12, 2>, !db.decimal<12, 2>, !db.decimal<12, 2>, i64, !db.decimal<12, 2>, i64, !db.decimal<12, 2>, i64
} attributes {rows = 5930888.26171875 : f64}
%5 = relalg.map %4 computes : [@aggr0::@tmp_attr8({type = !db.decimal<31, 21>})] (%arg0: !tuples.tuple){
  %10 = tuples.getcol %arg0 @aggr_rw::@rw0 : !db.decimal<12, 2>
  %11 = tuples.getcol %arg0 @aggr_rw::@rw1 : i64
  %12 = db.cast %11 : i64 -> !db.decimal<19, 0>
  %13 = db.div %10 : !db.decimal<12, 2>, %12 : !db.decimal<19, 0>
  tuples.return %13 : !db.decimal<31, 21>
} attributes {rows = 5930888.26171875 : f64}
%6 = relalg.map %5 computes : [@aggr0::@tmp_attr6({type = !db.decimal<31, 21>})] (%arg0: !tuples.tuple){
  %10 = tuples.getcol %arg0 @aggr_rw::@rw2 : !db.decimal<12, 2>
  %11 = tuples.getcol %arg0 @aggr_rw::@rw3 : i64
  %12 = db.cast %11 : i64 -> !db.decimal<19, 0>
  %13 = db.div %10 : !db.decimal<12, 2>, %12 : !db.decimal<19, 0>
  tuples.return %13 : !db.decimal<31, 21>
} attributes {rows = 5930888.26171875 : f64}
%7 = relalg.map %6 computes : [@aggr0::@tmp_attr7({type = !db.decimal<31, 21>})] (%arg0: !tuples.tuple){
  %10 = tuples.getcol %arg0 @aggr_rw::@rw4 : !db.decimal<12, 2>
  %11 = tuples.getcol %arg0 @aggr_rw::@rw5 : i64
  %12 = db.cast %11 : i64 -> !db.decimal<19, 0>
  %13 = db.div %10 : !db.decimal<12, 2>, %12 : !db.decimal<19, 0>
  tuples.return %13 : !db.decimal<31, 21>
} attributes {rows = 5930888.26171875 : f64}
%8 = relalg.sort %7 [(@lineitem::@l_returnflag,asc),(@lineitem::@l_linestatus,asc)] {rows = 5930888.26171875 : f64}
%9 = relalg.materialize %8 [@lineitem::@l_returnflag,@lineitem::@l_linestatus,@aggr0::@tmp_attr0,@aggr0::@tmp_attr1,@aggr0::@tmp_attr2,@aggr0::@tmp_attr4,@aggr0::@tmp_attr6,@aggr0::@tmp_attr7,@aggr0::@tmp_attr8,@aggr0::@tmp_attr9] => ["l_returnflag", "l_linestatus", "sum_qty", "sum_base_price", "sum_disc_price", "sum_charge", "avg_qty", "avg_price", "avg_disc", "count_order"] : !subop.result_table<[l_returnflag$0 : !db.char<1>, l_linestatus$0 : !db.char<1>, sum_qty$0 : !db.decimal<12, 2>, sum_base_price$0 : !db.decimal<12, 2>, sum_disc_price$0 : !db.decimal<24, 4>, sum_charge$0 : !db.decimal<36, 6>, avg_qty$0 : !db.decimal<31, 21>, avg_price$0 : !db.decimal<31, 21>, avg_disc$0 : !db.decimal<31, 21>, count_order$0 : i64]>
subop.set_result 0 %9 : !subop.result_table<[l_returnflag$0 : !db.char<1>, l_linestatus$0 : !db.char<1>, sum_qty$0 : !db.decimal<12, 2>, sum_base_price$0 : !db.decimal<12, 2>, sum_disc_price$0 : !db.decimal<24, 4>, sum_charge$0 : !db.decimal<36, 6>, avg_qty$0 : !db.decimal<31, 21>, avg_price$0 : !db.decimal<31, 21>, avg_disc$0 : !db.decimal<31, 21>, count_order$0 : i64]>
tuples.return

// CHECK:       builtin.module {
// CHECK-NEXT:    %0 = "relalg.basetable"() <{l_returnflag = @lineitem::@l_returnflag({type = !db.char<1>}), l_shipmode = @lineitem::@l_shipmode({type = !db.string}), l_shipinstruct = @lineitem::@l_shipinstruct({type = !db.string}), l_quantity = @lineitem::@l_quantity({type = !db.decimal<12, 2>}), l_tax = @lineitem::@l_tax({type = !db.decimal<12, 2>}), l_shipdate = @lineitem::@l_shipdate({type = !db.date<day>}), l_suppkey = @lineitem::@l_suppkey({type = i32}), l_orderkey = @lineitem::@l_orderkey({type = i32}), l_comment = @lineitem::@l_comment({type = !db.string}), l_linestatus = @lineitem::@l_linestatus({type = !db.char<1>}), primaryKeyHashValue = @lineitem::@primaryKeyHashValue({type = index}), l_receiptdate = @lineitem::@l_receiptdate({type = !db.date<day>}), l_linenumber = @lineitem::@l_linenumber({type = i32}), l_extendedprice = @lineitem::@l_extendedprice({type = !db.decimal<12, 2>}), l_partkey = @lineitem::@l_partkey({type = i32}), l_discount = @lineitem::@l_discount({type = !db.decimal<12, 2>}), l_commitdate = @lineitem::@l_commitdate({type = !db.date<day>})}> {rows = 4708201766515113984, table_identifier = "lineitem"} : () -> (!tuples.tuplestream)
// CHECK-NEXT:    %1 = "relalg.selection"(%0) ({
// CHECK-NEXT:    ^bb0(%2: !tuples.tuple):
// CHECK-NEXT:      %3 = db.constant("1998-09-02") : !db.date<day>
// CHECK-NEXT:      %4 = "tuples.getcol"(%2) {attr = @lineitem::@l_shipdate} : (!tuples.tuple) -> (!db.date<day>)
// CHECK-NEXT:      %5 = db.compare lte %4 : !db.date<day>, %3 : !db.date<day> : i1
// CHECK-NEXT:      "tuples.return"(%5) : (i1) -> ()
// CHECK-NEXT:    }) {cost = 5930888.5, evaluationCost = 2.0, rows = 5930888.5, selectivity = 0.98828125} : (!tuples.tuplestream) -> (!tuples.tuplestream)
// CHECK-NEXT:    %2 = "relalg.map"(%1) ({
// CHECK-NEXT:    ^bb0(%3: !tuples.tuple):
// CHECK-NEXT:      %4 = db.constant(1 : i32) : !db.decimal<12, 2>
// CHECK-NEXT:      %5 = "tuples.getcol"(%3) {attr = @lineitem::@l_extendedprice} : (!tuples.tuple) -> (!db.decimal<12, 2>)
// CHECK-NEXT:      %6 = "tuples.getcol"(%3) {attr = @lineitem::@l_discount} : (!tuples.tuple) -> (!db.decimal<12, 2>)
// CHECK-NEXT:      %7 = db.sub %4 : !db.decimal<12, 2>, %6 : !db.decimal<12, 2> : !db.decimal<12, 2>
// CHECK-NEXT:      %8 = db.mul %5 : !db.decimal<12, 2>, %7 : !db.decimal<12, 2> : !db.decimal<24, 4>
// CHECK-NEXT:      %9 = "tuples.getcol"(%3) {attr = @lineitem::@l_tax} : (!tuples.tuple) -> (!db.decimal<12, 2>)
// CHECK-NEXT:      %10 = db.add %4 : !db.decimal<12, 2>, %9 : !db.decimal<12, 2> : !db.decimal<12, 2>
// CHECK-NEXT:      %11 = db.mul %8 : !db.decimal<24, 4>, %10 : !db.decimal<12, 2> : !db.decimal<36, 6>
// CHECK-NEXT:      "tuples.return"(%11) : (!db.decimal<36, 6>) -> ()
// CHECK-NEXT:    }) {rows = 5930888.5, computed_cols = [@map0::@tmp_attr5({type = !db.decimal<36, 6>})]} : (!tuples.tuplestream) -> (!tuples.tuplestream)
// CHECK-NEXT:    %3 = "relalg.map"(%2) ({
// CHECK-NEXT:    ^bb0(%4: !tuples.tuple):
// CHECK-NEXT:      %5 = db.constant(1 : i32) : !db.decimal<12, 2>
// CHECK-NEXT:      %6 = "tuples.getcol"(%4) {attr = @lineitem::@l_extendedprice} : (!tuples.tuple) -> (!db.decimal<12, 2>)
// CHECK-NEXT:      %7 = "tuples.getcol"(%4) {attr = @lineitem::@l_discount} : (!tuples.tuple) -> (!db.decimal<12, 2>)
// CHECK-NEXT:      %8 = db.sub %5 : !db.decimal<12, 2>, %7 : !db.decimal<12, 2> : !db.decimal<12, 2>
// CHECK-NEXT:      %9 = db.mul %6 : !db.decimal<12, 2>, %8 : !db.decimal<12, 2> : !db.decimal<24, 4>
// CHECK-NEXT:      "tuples.return"(%9) : (!db.decimal<24, 4>) -> ()
// CHECK-NEXT:    }) {rows = 5930888.5, computed_cols = [@map0::@tmp_attr3({type = !db.decimal<24, 4>})]} : (!tuples.tuplestream) -> (!tuples.tuplestream)
// CHECK-NEXT:    %4 = "relalg.aggregation"(%3) ({
// CHECK-NEXT:    ^bb0(%5: !tuples.tuplestream, %6: !tuples.tuple):
// CHECK-NEXT:      %7 = "relalg.count"(%5) : (!tuples.tuplestream) -> (i64)
// CHECK-NEXT:      %8 = "relalg.aggrfn"(%5) {fn = sum, attr = @lineitem::@l_discount} : (!tuples.tuplestream) -> (!db.decimal<12, 2>)
// CHECK-NEXT:      %9 = "relalg.aggrfn"(%5) {fn = count, attr = @lineitem::@l_discount} : (!tuples.tuplestream) -> (i64)
// CHECK-NEXT:      %10 = "relalg.aggrfn"(%5) {fn = sum, attr = @lineitem::@l_quantity} : (!tuples.tuplestream) -> (!db.decimal<12, 2>)
// CHECK-NEXT:      %11 = "relalg.aggrfn"(%5) {fn = count, attr = @lineitem::@l_quantity} : (!tuples.tuplestream) -> (i64)
// CHECK-NEXT:      %12 = "relalg.aggrfn"(%5) {fn = sum, attr = @map0::@tmp_attr5} : (!tuples.tuplestream) -> (!db.decimal<36, 6>)
// CHECK-NEXT:      %13 = "relalg.aggrfn"(%5) {fn = sum, attr = @map0::@tmp_attr3} : (!tuples.tuplestream) -> (!db.decimal<24, 4>)
// CHECK-NEXT:      %14 = "relalg.aggrfn"(%5) {fn = sum, attr = @lineitem::@l_extendedprice} : (!tuples.tuplestream) -> (!db.decimal<12, 2>)
// CHECK-NEXT:      %15 = "relalg.aggrfn"(%5) {fn = count, attr = @lineitem::@l_extendedprice} : (!tuples.tuplestream) -> (i64)
// CHECK-NEXT:      "tuples.return"(%7, %12, %13, %14, %10, %8, %9, %10, %11, %14, %15) : (i64, !db.decimal<36, 6>, !db.decimal<24, 4>, !db.decimal<12, 2>, !db.decimal<12, 2>, !db.decimal<12, 2>, i64, !db.decimal<12, 2>, i64, !db.decimal<12, 2>, i64) -> ()
// CHECK-NEXT:    }) {rows = 5930888.5, group_by_cols = [@lineitem::@l_returnflag, @lineitem::@l_linestatus], computed_cols = [@aggr0::@tmp_attr9({type = i64}), @aggr0::@tmp_attr4({type = !db.decimal<36, 6>}), @aggr0::@tmp_attr2({type = !db.decimal<24, 4>}), @aggr0::@tmp_attr1({type = !db.decimal<12, 2>}), @aggr0::@tmp_attr0({type = !db.decimal<12, 2>}), @aggr_rw::@rw0({type = !db.decimal<12, 2>}), @aggr_rw::@rw1({type = i64}), @aggr_rw::@rw2({type = !db.decimal<12, 2>}), @aggr_rw::@rw3({type = i64}), @aggr_rw::@rw4({type = !db.decimal<12, 2>}), @aggr_rw::@rw5({type = i64})]} : (!tuples.tuplestream) -> (!tuples.tuplestream)
// CHECK-NEXT:    %5 = "relalg.map"(%4) ({
// CHECK-NEXT:    ^bb0(%6: !tuples.tuple):
// CHECK-NEXT:      %7 = "tuples.getcol"(%6) {attr = @aggr_rw::@rw0} : (!tuples.tuple) -> (!db.decimal<12, 2>)
// CHECK-NEXT:      %8 = "tuples.getcol"(%6) {attr = @aggr_rw::@rw1} : (!tuples.tuple) -> (i64)
// CHECK-NEXT:      %9 = db.cast %8 : i64 -> !db.decimal<19, 0>
// CHECK-NEXT:      %10 = db.div %7 : !db.decimal<12, 2>, %9 : !db.decimal<19, 0> : !db.decimal<31, 21>
// CHECK-NEXT:      "tuples.return"(%10) : (!db.decimal<31, 21>) -> ()
// CHECK-NEXT:    }) {rows = 5930888.5, computed_cols = [@aggr0::@tmp_attr8({type = !db.decimal<31, 21>})]} : (!tuples.tuplestream) -> (!tuples.tuplestream)
// CHECK-NEXT:    %6 = "relalg.map"(%5) ({
// CHECK-NEXT:    ^bb0(%7: !tuples.tuple):
// CHECK-NEXT:      %8 = "tuples.getcol"(%7) {attr = @aggr_rw::@rw2} : (!tuples.tuple) -> (!db.decimal<12, 2>)
// CHECK-NEXT:      %9 = "tuples.getcol"(%7) {attr = @aggr_rw::@rw3} : (!tuples.tuple) -> (i64)
// CHECK-NEXT:      %10 = db.cast %9 : i64 -> !db.decimal<19, 0>
// CHECK-NEXT:      %11 = db.div %8 : !db.decimal<12, 2>, %10 : !db.decimal<19, 0> : !db.decimal<31, 21>
// CHECK-NEXT:      "tuples.return"(%11) : (!db.decimal<31, 21>) -> ()
// CHECK-NEXT:    }) {rows = 5930888.5, computed_cols = [@aggr0::@tmp_attr6({type = !db.decimal<31, 21>})]} : (!tuples.tuplestream) -> (!tuples.tuplestream)
// CHECK-NEXT:    %7 = "relalg.map"(%6) ({
// CHECK-NEXT:    ^bb0(%8: !tuples.tuple):
// CHECK-NEXT:      %9 = "tuples.getcol"(%8) {attr = @aggr_rw::@rw4} : (!tuples.tuple) -> (!db.decimal<12, 2>)
// CHECK-NEXT:      %10 = "tuples.getcol"(%8) {attr = @aggr_rw::@rw5} : (!tuples.tuple) -> (i64)
// CHECK-NEXT:      %11 = db.cast %10 : i64 -> !db.decimal<19, 0>
// CHECK-NEXT:      %12 = db.div %9 : !db.decimal<12, 2>, %11 : !db.decimal<19, 0> : !db.decimal<31, 21>
// CHECK-NEXT:      "tuples.return"(%12) : (!db.decimal<31, 21>) -> ()
// CHECK-NEXT:    }) {rows = 5930888.5, computed_cols = [@aggr0::@tmp_attr7({type = !db.decimal<31, 21>})]} : (!tuples.tuplestream) -> (!tuples.tuplestream)
// CHECK-NEXT:    %8 = "relalg.sort"(%7) {rows = 5930888.5, sortspecs = [(@lineitem::@l_returnflag,asc), (@lineitem::@l_linestatus,asc)]} : (!tuples.tuplestream) -> (!tuples.tuplestream)
// CHECK-NEXT:    %9 = "relalg.materialize"(%8) {cols = [@lineitem::@l_returnflag, @lineitem::@l_linestatus, @aggr0::@tmp_attr0, @aggr0::@tmp_attr1, @aggr0::@tmp_attr2, @aggr0::@tmp_attr4, @aggr0::@tmp_attr6, @aggr0::@tmp_attr7, @aggr0::@tmp_attr8, @aggr0::@tmp_attr9], columns = ["l_returnflag", "l_linestatus", "sum_qty", "sum_base_price", "sum_disc_price", "sum_charge", "avg_qty", "avg_price", "avg_disc", "count_order"]} : (!tuples.tuplestream) -> (!subop.result_table<[l_returnflag$0 : !db.char<1>, l_linestatus$0 : !db.char<1>, sum_qty$0 : !db.decimal<12, 2>, sum_base_price$0 : !db.decimal<12, 2>, sum_disc_price$0 : !db.decimal<24, 4>, sum_charge$0 : !db.decimal<36, 6>, avg_qty$0 : !db.decimal<31, 21>, avg_price$0 : !db.decimal<31, 21>, avg_disc$0 : !db.decimal<31, 21>, count_order$0 : i64]>)
// CHECK-NEXT:    "subop.set_result"(%9) {result_id = 0} : (!subop.result_table<[l_returnflag$0 : !db.char<1>, l_linestatus$0 : !db.char<1>, sum_qty$0 : !db.decimal<12, 2>, sum_base_price$0 : !db.decimal<12, 2>, sum_disc_price$0 : !db.decimal<24, 4>, sum_charge$0 : !db.decimal<36, 6>, avg_qty$0 : !db.decimal<31, 21>, avg_price$0 : !db.decimal<31, 21>, avg_disc$0 : !db.decimal<31, 21>, count_order$0 : i64]>) -> ()
// CHECK-NEXT:    "tuples.return"() : () -> ()
// CHECK-NEXT:  }