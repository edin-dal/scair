// RUN: scair-opt %s | filecheck %s

"op1"() ({
^bb0():
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

// CHECK: "op1"() ({
// CHECK: ^bb0():
// CHECK:   %0 = "test.op"() : () -> (!tuples.tuple<f32, f32>)
// CHECK:   %1 = "test.op"() : () -> (!tuples.tuplestream<f32, f32>)
// CHECK:   %2 = "test.op"() : () -> (i32)
// CHECK:   "tuples.return"(%2) : (i32) -> ()
// CHECK:   %3 = "tuples.getcol"(%0) {attr = @example::@one} : (!tuples.tuple<f32, f32>) -> (i32)

}) : () -> ()
