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
// CHECK:   %0 = "relalg.basetable"() <{l_returnflag = @lineitem::@l_returnflag({type = !db.char<1>}), l_shipmode = @lineitem::@l_shipmode({type = !db.string<>}), l_shipinstruct = @lineitem::@l_shipinstruct({type = !db.string<>}), l_quantity = @lineitem::@l_quantity({type = !db.decimal<12, 2>}), l_tax = @lineitem::@l_tax({type = !db.decimal<12, 2>}), l_shipdate = @lineitem::@l_shipdate({type = !db.date<day>}), l_suppkey = @lineitem::@l_suppkey({type = i32}), l_orderkey = @lineitem::@l_orderkey({type = i32}), l_comment = @lineitem::@l_comment({type = !db.string<>}), l_linestatus = @lineitem::@l_linestatus({type = !db.char<1>}), primaryKeyHashValue = @lineitem::@primaryKeyHashValue({type = index}), l_receiptdate = @lineitem::@l_receiptdate({type = !db.date<day>}), l_linenumber = @lineitem::@l_linenumber({type = i32}), l_extendedprice = @lineitem::@l_extendedprice({type = !db.decimal<12, 2>}), l_partkey = @lineitem::@l_partkey({type = i32}), l_discount = @lineitem::@l_discount({type = !db.decimal<12, 2>}), l_commitdate = @lineitem::@l_commitdate({type = !db.date<day>})}> {rows = 4708201766515113984, table_identifier = "lineitem"} : () -> (!tuples.tuplestream<>)
// CHECK:   %1 = "relalg.selection"(%0) ({
// CHECK:   ^bb1(%2: !tuples.tuple<>):
// CHECK:     %3 = "db.constant"() {value = "1998-09-02"} : () -> (!db.date<day>)
// CHECK:     %4 = "tuples.getcol"(%2) {attr = @lineitem::@l_shipdate} : (!tuples.tuple<>) -> (!db.date<day>)
// CHECK:     %5 = "db.compare"(%4, %3) {predicate = lte} : (!db.date<day>, !db.date<day>) -> (i1)
// CHECK:     "tuples.return"(%5) : (i1) -> ()
// CHECK:   }) {cost = 5930888.5, evaluationCost = 2.0, rows = 5930888.5, selectivity = 0.98828125} : (!tuples.tuplestream<>) -> (!tuples.tuplestream<>)
// CHECK:   %6 = "relalg.map"(%1) ({
// CHECK:   ^bb2(%7: !tuples.tuple<>):
// CHECK:     %8 = "db.constant"() {value = 1 : i32} : () -> (!db.decimal<12, 2>)
// CHECK:     %9 = "tuples.getcol"(%7) {attr = @lineitem::@l_extendedprice} : (!tuples.tuple<>) -> (!db.decimal<12, 2>)
// CHECK:     %10 = "tuples.getcol"(%7) {attr = @lineitem::@l_discount} : (!tuples.tuple<>) -> (!db.decimal<12, 2>)
// CHECK:     %11 = "db.sub"(%8, %10) : (!db.decimal<12, 2>, !db.decimal<12, 2>) -> (!db.decimal<12, 2>)
// CHECK:     %12 = "db.mul"(%9, %11) : (!db.decimal<12, 2>, !db.decimal<12, 2>) -> (!db.decimal<24, 4>)
// CHECK:     %13 = "tuples.getcol"(%7) {attr = @lineitem::@l_tax} : (!tuples.tuple<>) -> (!db.decimal<12, 2>)
// CHECK:     %14 = "db.add"(%8, %13) : (!db.decimal<12, 2>, !db.decimal<12, 2>) -> (!db.decimal<12, 2>)
// CHECK:     %15 = "db.mul"(%12, %14) : (!db.decimal<24, 4>, !db.decimal<12, 2>) -> (!db.decimal<36, 6>)
// CHECK:     "tuples.return"(%15) : (!db.decimal<36, 6>) -> ()
// CHECK:   }) {rows = 5930888.5, computed_cols = [@map0::@tmp_attr5({type = !db.decimal<36, 6>})]} : (!tuples.tuplestream<>) -> (!tuples.tuplestream<>)
// CHECK:   %16 = "relalg.map"(%6) ({
// CHECK:   ^bb3(%17: !tuples.tuple<>):
// CHECK:     %18 = "db.constant"() {value = 1 : i32} : () -> (!db.decimal<12, 2>)
// CHECK:     %19 = "tuples.getcol"(%17) {attr = @lineitem::@l_extendedprice} : (!tuples.tuple<>) -> (!db.decimal<12, 2>)
// CHECK:     %20 = "tuples.getcol"(%17) {attr = @lineitem::@l_discount} : (!tuples.tuple<>) -> (!db.decimal<12, 2>)
// CHECK:     %21 = "db.sub"(%18, %20) : (!db.decimal<12, 2>, !db.decimal<12, 2>) -> (!db.decimal<12, 2>)
// CHECK:     %22 = "db.mul"(%19, %21) : (!db.decimal<12, 2>, !db.decimal<12, 2>) -> (!db.decimal<24, 4>)
// CHECK:     "tuples.return"(%22) : (!db.decimal<24, 4>) -> ()
// CHECK:   }) {rows = 5930888.5, computed_cols = [@map0::@tmp_attr3({type = !db.decimal<24, 4>})]} : (!tuples.tuplestream<>) -> (!tuples.tuplestream<>)
// CHECK:   %23 = "relalg.aggregation"(%16) ({
// CHECK:   ^bb4(%24: !tuples.tuplestream<>, %25: !tuples.tuple<>):
// CHECK:     %26 = "relalg.count"(%24) : (!tuples.tuplestream<>) -> (i64)
// CHECK:     %27 = "relalg.aggrfn"(%24) {fn = sum, attr = @lineitem::@l_discount} : (!tuples.tuplestream<>) -> (!db.decimal<12, 2>)
// CHECK:     %28 = "relalg.aggrfn"(%24) {fn = count, attr = @lineitem::@l_discount} : (!tuples.tuplestream<>) -> (i64)
// CHECK:     %29 = "relalg.aggrfn"(%24) {fn = sum, attr = @lineitem::@l_quantity} : (!tuples.tuplestream<>) -> (!db.decimal<12, 2>)
// CHECK:     %30 = "relalg.aggrfn"(%24) {fn = count, attr = @lineitem::@l_quantity} : (!tuples.tuplestream<>) -> (i64)
// CHECK:     %31 = "relalg.aggrfn"(%24) {fn = sum, attr = @map0::@tmp_attr5} : (!tuples.tuplestream<>) -> (!db.decimal<36, 6>)
// CHECK:     %32 = "relalg.aggrfn"(%24) {fn = sum, attr = @map0::@tmp_attr3} : (!tuples.tuplestream<>) -> (!db.decimal<24, 4>)
// CHECK:     %33 = "relalg.aggrfn"(%24) {fn = sum, attr = @lineitem::@l_extendedprice} : (!tuples.tuplestream<>) -> (!db.decimal<12, 2>)
// CHECK:     %34 = "relalg.aggrfn"(%24) {fn = count, attr = @lineitem::@l_extendedprice} : (!tuples.tuplestream<>) -> (i64)
// CHECK:     "tuples.return"(%26, %31, %32, %33, %29, %27, %28, %29, %30, %33, %34) : (i64, !db.decimal<36, 6>, !db.decimal<24, 4>, !db.decimal<12, 2>, !db.decimal<12, 2>, !db.decimal<12, 2>, i64, !db.decimal<12, 2>, i64, !db.decimal<12, 2>, i64) -> ()
// CHECK:   }) {rows = 5930888.5, group_by_cols = [@lineitem::@l_returnflag, @lineitem::@l_linestatus], computed_cols = [@aggr0::@tmp_attr9({type = i64}), @aggr0::@tmp_attr4({type = !db.decimal<36, 6>}), @aggr0::@tmp_attr2({type = !db.decimal<24, 4>}), @aggr0::@tmp_attr1({type = !db.decimal<12, 2>}), @aggr0::@tmp_attr0({type = !db.decimal<12, 2>}), @aggr_rw::@rw0({type = !db.decimal<12, 2>}), @aggr_rw::@rw1({type = i64}), @aggr_rw::@rw2({type = !db.decimal<12, 2>}), @aggr_rw::@rw3({type = i64}), @aggr_rw::@rw4({type = !db.decimal<12, 2>}), @aggr_rw::@rw5({type = i64})]} : (!tuples.tuplestream<>) -> (!tuples.tuplestream<>)
// CHECK:   %35 = "relalg.map"(%23) ({
// CHECK:   ^bb5(%36: !tuples.tuple<>):
// CHECK:     %37 = "tuples.getcol"(%36) {attr = @aggr_rw::@rw0} : (!tuples.tuple<>) -> (!db.decimal<12, 2>)
// CHECK:     %38 = "tuples.getcol"(%36) {attr = @aggr_rw::@rw1} : (!tuples.tuple<>) -> (i64)
// CHECK:     %39 = "db.cast"(%38) : (i64) -> (!db.decimal<19, 0>)
// CHECK:     %40 = "db.div"(%37, %39) : (!db.decimal<12, 2>, !db.decimal<19, 0>) -> (!db.decimal<31, 21>)
// CHECK:     "tuples.return"(%40) : (!db.decimal<31, 21>) -> ()
// CHECK:   }) {rows = 5930888.5, computed_cols = [@aggr0::@tmp_attr8({type = !db.decimal<31, 21>})]} : (!tuples.tuplestream<>) -> (!tuples.tuplestream<>)
// CHECK:   %41 = "relalg.map"(%35) ({
// CHECK:   ^bb6(%42: !tuples.tuple<>):
// CHECK:     %43 = "tuples.getcol"(%42) {attr = @aggr_rw::@rw2} : (!tuples.tuple<>) -> (!db.decimal<12, 2>)
// CHECK:     %44 = "tuples.getcol"(%42) {attr = @aggr_rw::@rw3} : (!tuples.tuple<>) -> (i64)
// CHECK:     %45 = "db.cast"(%44) : (i64) -> (!db.decimal<19, 0>)
// CHECK:     %46 = "db.div"(%43, %45) : (!db.decimal<12, 2>, !db.decimal<19, 0>) -> (!db.decimal<31, 21>)
// CHECK:     "tuples.return"(%46) : (!db.decimal<31, 21>) -> ()
// CHECK:   }) {rows = 5930888.5, computed_cols = [@aggr0::@tmp_attr6({type = !db.decimal<31, 21>})]} : (!tuples.tuplestream<>) -> (!tuples.tuplestream<>)
// CHECK:   %47 = "relalg.map"(%41) ({
// CHECK:   ^bb7(%48: !tuples.tuple<>):
// CHECK:     %49 = "tuples.getcol"(%48) {attr = @aggr_rw::@rw4} : (!tuples.tuple<>) -> (!db.decimal<12, 2>)
// CHECK:     %50 = "tuples.getcol"(%48) {attr = @aggr_rw::@rw5} : (!tuples.tuple<>) -> (i64)
// CHECK:     %51 = "db.cast"(%50) : (i64) -> (!db.decimal<19, 0>)
// CHECK:     %52 = "db.div"(%49, %51) : (!db.decimal<12, 2>, !db.decimal<19, 0>) -> (!db.decimal<31, 21>)
// CHECK:     "tuples.return"(%52) : (!db.decimal<31, 21>) -> ()
// CHECK:   }) {rows = 5930888.5, computed_cols = [@aggr0::@tmp_attr7({type = !db.decimal<31, 21>})]} : (!tuples.tuplestream<>) -> (!tuples.tuplestream<>)
// CHECK:   %53 = "relalg.sort"(%47) {rows = 5930888.5, sortspecs = [(@lineitem::@l_returnflag,asc), (@lineitem::@l_linestatus,asc)]} : (!tuples.tuplestream<>) -> (!tuples.tuplestream<>)
// CHECK:   %54 = "relalg.materialize"(%53) {cols = [@lineitem::@l_returnflag, @lineitem::@l_linestatus, @aggr0::@tmp_attr0, @aggr0::@tmp_attr1, @aggr0::@tmp_attr2, @aggr0::@tmp_attr4, @aggr0::@tmp_attr6, @aggr0::@tmp_attr7, @aggr0::@tmp_attr8, @aggr0::@tmp_attr9], columns = ["l_returnflag", "l_linestatus", "sum_qty", "sum_base_price", "sum_disc_price", "sum_charge", "avg_qty", "avg_price", "avg_disc", "count_order"]} : (!tuples.tuplestream<>) -> (!subop.result_table<[l_returnflag$0 : !db.char<1>, l_linestatus$0 : !db.char<1>, sum_qty$0 : !db.decimal<12, 2>, sum_base_price$0 : !db.decimal<12, 2>, sum_disc_price$0 : !db.decimal<24, 4>, sum_charge$0 : !db.decimal<36, 6>, avg_qty$0 : !db.decimal<31, 21>, avg_price$0 : !db.decimal<31, 21>, avg_disc$0 : !db.decimal<31, 21>, count_order$0 : i64]>)
// CHECK:   "subop.set_result"(%54) {result_id = 0} : (!subop.result_table<[l_returnflag$0 : !db.char<1>, l_linestatus$0 : !db.char<1>, sum_qty$0 : !db.decimal<12, 2>, sum_base_price$0 : !db.decimal<12, 2>, sum_disc_price$0 : !db.decimal<24, 4>, sum_charge$0 : !db.decimal<36, 6>, avg_qty$0 : !db.decimal<31, 21>, avg_price$0 : !db.decimal<31, 21>, avg_disc$0 : !db.decimal<31, 21>, count_order$0 : i64]>) -> ()
// CHECK:   "tuples.return"() : () -> ()
// CHECK: }) : () -> ()

}) : () -> ()
