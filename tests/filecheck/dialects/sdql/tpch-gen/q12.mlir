func.func @range_builtin(%n: i32) -> dictionary<i32, i32> {
  %zero = "arith.constant"() <{value = 0}> : () -> i32
  %is_terminal = "arith.cmpi"(%n, %zero) <{"predicate" = 3}> : (i32, i32) -> i1
  %res = "scf.if"(%is_terminal) ({
    %empty = sdql.empty_dictionary : dictionary<i32, i32>
    "scf.yield"(%empty) : (dictionary<i32, i32>) -> ()
  }, {
    %one = "arith.constant"() <{value = 1}> : () -> i32
    %smaller_n = "arith.subi"(%n, %one) : (i32, i32) -> i32
    %prev_range = "func.call"(%smaller_n) <{callee = @range}> : (i32) -> dictionary<i32, i32>
    %true = "arith.constant"() <{value = 1}> : () -> i32
    %extension = sdql.create_dictionary %smaller_n, %true : i32, i32 -> dictionary<i32, i32>

    // return %extension + %prev_range
    %added = sdql.dictionary_add %extension %prev_range : dictionary<i32, i32>, dictionary<i32, i32> -> dictionary<i32, i32>
    "scf.yield"(%added) : (dictionary<i32, i32>) -> ()
  }) : (i1) -> dictionary<i32, i32>
  func.return %res : dictionary<i32, i32>
}
%orders = sdql.load "datasets/tpch/orders.tbl" : record<"o_orderkey" : dictionary<i32, i32>, "o_custkey" : dictionary<i32, i32>, "o_orderstatus" : dictionary<i32, memref<1xi8>>, "o_totalprice" : dictionary<i32, f64>, "o_orderdate" : dictionary<i32, i32>, "o_orderpriority" : dictionary<i32, memref<15xi8>>, "o_clerk" : dictionary<i32, memref<15xi8>>, "o_shippriority" : dictionary<i32, i32>, "o_comment" : dictionary<i32, memref<79xi8>>, "size" : i32>
%lineitem = sdql.load "datasets/tpch/lineitem.tbl" : record<"l_orderkey" : dictionary<i32, i32>, "l_partkey" : dictionary<i32, i32>, "l_suppkey" : dictionary<i32, i32>, "l_linenumber" : dictionary<i32, i32>, "l_quantity" : dictionary<i32, f64>, "l_extendedprice" : dictionary<i32, f64>, "l_discount" : dictionary<i32, f64>, "l_tax" : dictionary<i32, f64>, "l_returnflag" : dictionary<i32, memref<1xi8>>, "l_linestatus" : dictionary<i32, memref<1xi8>>, "l_shipdate" : dictionary<i32, i32>, "l_commitdate" : dictionary<i32, i32>, "l_receiptdate" : dictionary<i32, i32>, "l_shipinstruct" : dictionary<i32, memref<25xi8>>, "l_shipmode" : dictionary<i32, memref<10xi8>>, "l_comment" : dictionary<i32, memref<44xi8>>, "size" : i32>
// MAIL
%constds_0 = "arith.constant"() <{value = dense<[77, 65, 73, 76]> : memref<?xi8>}> : () -> memref<?xi8>
%consti_1 = "arith.constant"() <{value = 5 : i32}> : () -> i32
%mail = sdql.external "ConstantString", %constds_0, %consti_1 : memref<?xi8>, i32 -> memref<4xi8>
// SHIP
%constds_2 = "arith.constant"() <{value = dense<[83, 72, 73, 80]> : memref<?xi8>}> : () -> memref<?xi8>
%consti_3 = "arith.constant"() <{value = 5 : i32}> : () -> i32
%ship = sdql.external "ConstantString", %constds_2, %consti_3 : memref<?xi8>, i32 -> memref<4xi8>
// 1-URGENT
%constds_4 = "arith.constant"() <{value = dense<[49, 45, 85, 82, 71, 69, 78, 84]> : memref<?xi8>}> : () -> memref<?xi8>
%consti_5 = "arith.constant"() <{value = 9 : i32}> : () -> i32
%urgent = sdql.external "ConstantString", %constds_4, %consti_5 : memref<?xi8>, i32 -> memref<8xi8>
// 2-HIGH
%constds_6 = "arith.constant"() <{value = dense<[50, 45, 72, 73, 71, 72]> : memref<?xi8>}> : () -> memref<?xi8>
%consti_7 = "arith.constant"() <{value = 7 : i32}> : () -> i32
%high = sdql.external "ConstantString", %constds_6, %consti_7 : memref<?xi8>, i32 -> memref<6xi8>
%fieldnode_8 = sdql.access_record %lineitem "size" : record<"l_orderkey": dictionary<i32, i32>, "l_partkey": dictionary<i32, i32>, "l_suppkey": dictionary<i32, i32>, "l_linenumber": dictionary<i32, i32>, "l_quantity": dictionary<i32, f64>, "l_extendedprice": dictionary<i32, f64>, "l_discount": dictionary<i32, f64>, "l_tax": dictionary<i32, f64>, "l_returnflag": dictionary<i32, memref<1xi8>>, "l_linestatus": dictionary<i32, memref<1xi8>>, "l_shipdate": dictionary<i32, i32>, "l_commitdate": dictionary<i32, i32>, "l_receiptdate": dictionary<i32, i32>, "l_shipinstruct": dictionary<i32, memref<25xi8>>, "l_shipmode": dictionary<i32, memref<10xi8>>, "l_comment": dictionary<i32, memref<44xi8>>, "size": i32> -> i32
%rangenode_9 = "func.call"(%fieldnode_8) <{callee = @range_builtin}> : (i32) -> dictionary<i32, i32>
%l_h = sdql.sum %rangenode_9 : dictionary<i32, i32> -> dictionary<i32, dictionary<memref<10xi8>, i32>> {
^bb_10(%i: i32, %_: i32):
  %fieldnode_12 = sdql.access_record %lineitem "l_orderkey" : record<"l_orderkey": dictionary<i32, i32>, "l_partkey": dictionary<i32, i32>, "l_suppkey": dictionary<i32, i32>, "l_linenumber": dictionary<i32, i32>, "l_quantity": dictionary<i32, f64>, "l_extendedprice": dictionary<i32, f64>, "l_discount": dictionary<i32, f64>, "l_tax": dictionary<i32, f64>, "l_returnflag": dictionary<i32, memref<1xi8>>, "l_linestatus": dictionary<i32, memref<1xi8>>, "l_shipdate": dictionary<i32, i32>, "l_commitdate": dictionary<i32, i32>, "l_receiptdate": dictionary<i32, i32>, "l_shipinstruct": dictionary<i32, memref<25xi8>>, "l_shipmode": dictionary<i32, memref<10xi8>>, "l_comment": dictionary<i32, memref<44xi8>>, "size": i32> -> dictionary<i32, i32>
  %get_11 = sdql.lookup_dictionary %fieldnode_12 [%i : i32] : dictionary<i32, i32> -> i32
  %fieldnode_14 = sdql.access_record %lineitem "l_partkey" : record<"l_orderkey": dictionary<i32, i32>, "l_partkey": dictionary<i32, i32>, "l_suppkey": dictionary<i32, i32>, "l_linenumber": dictionary<i32, i32>, "l_quantity": dictionary<i32, f64>, "l_extendedprice": dictionary<i32, f64>, "l_discount": dictionary<i32, f64>, "l_tax": dictionary<i32, f64>, "l_returnflag": dictionary<i32, memref<1xi8>>, "l_linestatus": dictionary<i32, memref<1xi8>>, "l_shipdate": dictionary<i32, i32>, "l_commitdate": dictionary<i32, i32>, "l_receiptdate": dictionary<i32, i32>, "l_shipinstruct": dictionary<i32, memref<25xi8>>, "l_shipmode": dictionary<i32, memref<10xi8>>, "l_comment": dictionary<i32, memref<44xi8>>, "size": i32> -> dictionary<i32, i32>
  %get_13 = sdql.lookup_dictionary %fieldnode_14 [%i : i32] : dictionary<i32, i32> -> i32
  %fieldnode_16 = sdql.access_record %lineitem "l_suppkey" : record<"l_orderkey": dictionary<i32, i32>, "l_partkey": dictionary<i32, i32>, "l_suppkey": dictionary<i32, i32>, "l_linenumber": dictionary<i32, i32>, "l_quantity": dictionary<i32, f64>, "l_extendedprice": dictionary<i32, f64>, "l_discount": dictionary<i32, f64>, "l_tax": dictionary<i32, f64>, "l_returnflag": dictionary<i32, memref<1xi8>>, "l_linestatus": dictionary<i32, memref<1xi8>>, "l_shipdate": dictionary<i32, i32>, "l_commitdate": dictionary<i32, i32>, "l_receiptdate": dictionary<i32, i32>, "l_shipinstruct": dictionary<i32, memref<25xi8>>, "l_shipmode": dictionary<i32, memref<10xi8>>, "l_comment": dictionary<i32, memref<44xi8>>, "size": i32> -> dictionary<i32, i32>
  %get_15 = sdql.lookup_dictionary %fieldnode_16 [%i : i32] : dictionary<i32, i32> -> i32
  %fieldnode_18 = sdql.access_record %lineitem "l_linenumber" : record<"l_orderkey": dictionary<i32, i32>, "l_partkey": dictionary<i32, i32>, "l_suppkey": dictionary<i32, i32>, "l_linenumber": dictionary<i32, i32>, "l_quantity": dictionary<i32, f64>, "l_extendedprice": dictionary<i32, f64>, "l_discount": dictionary<i32, f64>, "l_tax": dictionary<i32, f64>, "l_returnflag": dictionary<i32, memref<1xi8>>, "l_linestatus": dictionary<i32, memref<1xi8>>, "l_shipdate": dictionary<i32, i32>, "l_commitdate": dictionary<i32, i32>, "l_receiptdate": dictionary<i32, i32>, "l_shipinstruct": dictionary<i32, memref<25xi8>>, "l_shipmode": dictionary<i32, memref<10xi8>>, "l_comment": dictionary<i32, memref<44xi8>>, "size": i32> -> dictionary<i32, i32>
  %get_17 = sdql.lookup_dictionary %fieldnode_18 [%i : i32] : dictionary<i32, i32> -> i32
  %fieldnode_20 = sdql.access_record %lineitem "l_quantity" : record<"l_orderkey": dictionary<i32, i32>, "l_partkey": dictionary<i32, i32>, "l_suppkey": dictionary<i32, i32>, "l_linenumber": dictionary<i32, i32>, "l_quantity": dictionary<i32, f64>, "l_extendedprice": dictionary<i32, f64>, "l_discount": dictionary<i32, f64>, "l_tax": dictionary<i32, f64>, "l_returnflag": dictionary<i32, memref<1xi8>>, "l_linestatus": dictionary<i32, memref<1xi8>>, "l_shipdate": dictionary<i32, i32>, "l_commitdate": dictionary<i32, i32>, "l_receiptdate": dictionary<i32, i32>, "l_shipinstruct": dictionary<i32, memref<25xi8>>, "l_shipmode": dictionary<i32, memref<10xi8>>, "l_comment": dictionary<i32, memref<44xi8>>, "size": i32> -> dictionary<i32, f64>
  %get_19 = sdql.lookup_dictionary %fieldnode_20 [%i : i32] : dictionary<i32, f64> -> f64
  %fieldnode_22 = sdql.access_record %lineitem "l_extendedprice" : record<"l_orderkey": dictionary<i32, i32>, "l_partkey": dictionary<i32, i32>, "l_suppkey": dictionary<i32, i32>, "l_linenumber": dictionary<i32, i32>, "l_quantity": dictionary<i32, f64>, "l_extendedprice": dictionary<i32, f64>, "l_discount": dictionary<i32, f64>, "l_tax": dictionary<i32, f64>, "l_returnflag": dictionary<i32, memref<1xi8>>, "l_linestatus": dictionary<i32, memref<1xi8>>, "l_shipdate": dictionary<i32, i32>, "l_commitdate": dictionary<i32, i32>, "l_receiptdate": dictionary<i32, i32>, "l_shipinstruct": dictionary<i32, memref<25xi8>>, "l_shipmode": dictionary<i32, memref<10xi8>>, "l_comment": dictionary<i32, memref<44xi8>>, "size": i32> -> dictionary<i32, f64>
  %get_21 = sdql.lookup_dictionary %fieldnode_22 [%i : i32] : dictionary<i32, f64> -> f64
  %fieldnode_24 = sdql.access_record %lineitem "l_discount" : record<"l_orderkey": dictionary<i32, i32>, "l_partkey": dictionary<i32, i32>, "l_suppkey": dictionary<i32, i32>, "l_linenumber": dictionary<i32, i32>, "l_quantity": dictionary<i32, f64>, "l_extendedprice": dictionary<i32, f64>, "l_discount": dictionary<i32, f64>, "l_tax": dictionary<i32, f64>, "l_returnflag": dictionary<i32, memref<1xi8>>, "l_linestatus": dictionary<i32, memref<1xi8>>, "l_shipdate": dictionary<i32, i32>, "l_commitdate": dictionary<i32, i32>, "l_receiptdate": dictionary<i32, i32>, "l_shipinstruct": dictionary<i32, memref<25xi8>>, "l_shipmode": dictionary<i32, memref<10xi8>>, "l_comment": dictionary<i32, memref<44xi8>>, "size": i32> -> dictionary<i32, f64>
  %get_23 = sdql.lookup_dictionary %fieldnode_24 [%i : i32] : dictionary<i32, f64> -> f64
  %fieldnode_26 = sdql.access_record %lineitem "l_tax" : record<"l_orderkey": dictionary<i32, i32>, "l_partkey": dictionary<i32, i32>, "l_suppkey": dictionary<i32, i32>, "l_linenumber": dictionary<i32, i32>, "l_quantity": dictionary<i32, f64>, "l_extendedprice": dictionary<i32, f64>, "l_discount": dictionary<i32, f64>, "l_tax": dictionary<i32, f64>, "l_returnflag": dictionary<i32, memref<1xi8>>, "l_linestatus": dictionary<i32, memref<1xi8>>, "l_shipdate": dictionary<i32, i32>, "l_commitdate": dictionary<i32, i32>, "l_receiptdate": dictionary<i32, i32>, "l_shipinstruct": dictionary<i32, memref<25xi8>>, "l_shipmode": dictionary<i32, memref<10xi8>>, "l_comment": dictionary<i32, memref<44xi8>>, "size": i32> -> dictionary<i32, f64>
  %get_25 = sdql.lookup_dictionary %fieldnode_26 [%i : i32] : dictionary<i32, f64> -> f64
  %fieldnode_28 = sdql.access_record %lineitem "l_returnflag" : record<"l_orderkey": dictionary<i32, i32>, "l_partkey": dictionary<i32, i32>, "l_suppkey": dictionary<i32, i32>, "l_linenumber": dictionary<i32, i32>, "l_quantity": dictionary<i32, f64>, "l_extendedprice": dictionary<i32, f64>, "l_discount": dictionary<i32, f64>, "l_tax": dictionary<i32, f64>, "l_returnflag": dictionary<i32, memref<1xi8>>, "l_linestatus": dictionary<i32, memref<1xi8>>, "l_shipdate": dictionary<i32, i32>, "l_commitdate": dictionary<i32, i32>, "l_receiptdate": dictionary<i32, i32>, "l_shipinstruct": dictionary<i32, memref<25xi8>>, "l_shipmode": dictionary<i32, memref<10xi8>>, "l_comment": dictionary<i32, memref<44xi8>>, "size": i32> -> dictionary<i32, memref<1xi8>>
  %get_27 = sdql.lookup_dictionary %fieldnode_28 [%i : i32] : dictionary<i32, memref<1xi8>> -> memref<1xi8>
  %fieldnode_30 = sdql.access_record %lineitem "l_linestatus" : record<"l_orderkey": dictionary<i32, i32>, "l_partkey": dictionary<i32, i32>, "l_suppkey": dictionary<i32, i32>, "l_linenumber": dictionary<i32, i32>, "l_quantity": dictionary<i32, f64>, "l_extendedprice": dictionary<i32, f64>, "l_discount": dictionary<i32, f64>, "l_tax": dictionary<i32, f64>, "l_returnflag": dictionary<i32, memref<1xi8>>, "l_linestatus": dictionary<i32, memref<1xi8>>, "l_shipdate": dictionary<i32, i32>, "l_commitdate": dictionary<i32, i32>, "l_receiptdate": dictionary<i32, i32>, "l_shipinstruct": dictionary<i32, memref<25xi8>>, "l_shipmode": dictionary<i32, memref<10xi8>>, "l_comment": dictionary<i32, memref<44xi8>>, "size": i32> -> dictionary<i32, memref<1xi8>>
  %get_29 = sdql.lookup_dictionary %fieldnode_30 [%i : i32] : dictionary<i32, memref<1xi8>> -> memref<1xi8>
  %fieldnode_32 = sdql.access_record %lineitem "l_shipdate" : record<"l_orderkey": dictionary<i32, i32>, "l_partkey": dictionary<i32, i32>, "l_suppkey": dictionary<i32, i32>, "l_linenumber": dictionary<i32, i32>, "l_quantity": dictionary<i32, f64>, "l_extendedprice": dictionary<i32, f64>, "l_discount": dictionary<i32, f64>, "l_tax": dictionary<i32, f64>, "l_returnflag": dictionary<i32, memref<1xi8>>, "l_linestatus": dictionary<i32, memref<1xi8>>, "l_shipdate": dictionary<i32, i32>, "l_commitdate": dictionary<i32, i32>, "l_receiptdate": dictionary<i32, i32>, "l_shipinstruct": dictionary<i32, memref<25xi8>>, "l_shipmode": dictionary<i32, memref<10xi8>>, "l_comment": dictionary<i32, memref<44xi8>>, "size": i32> -> dictionary<i32, i32>
  %get_31 = sdql.lookup_dictionary %fieldnode_32 [%i : i32] : dictionary<i32, i32> -> i32
  %fieldnode_34 = sdql.access_record %lineitem "l_commitdate" : record<"l_orderkey": dictionary<i32, i32>, "l_partkey": dictionary<i32, i32>, "l_suppkey": dictionary<i32, i32>, "l_linenumber": dictionary<i32, i32>, "l_quantity": dictionary<i32, f64>, "l_extendedprice": dictionary<i32, f64>, "l_discount": dictionary<i32, f64>, "l_tax": dictionary<i32, f64>, "l_returnflag": dictionary<i32, memref<1xi8>>, "l_linestatus": dictionary<i32, memref<1xi8>>, "l_shipdate": dictionary<i32, i32>, "l_commitdate": dictionary<i32, i32>, "l_receiptdate": dictionary<i32, i32>, "l_shipinstruct": dictionary<i32, memref<25xi8>>, "l_shipmode": dictionary<i32, memref<10xi8>>, "l_comment": dictionary<i32, memref<44xi8>>, "size": i32> -> dictionary<i32, i32>
  %get_33 = sdql.lookup_dictionary %fieldnode_34 [%i : i32] : dictionary<i32, i32> -> i32
  %fieldnode_36 = sdql.access_record %lineitem "l_receiptdate" : record<"l_orderkey": dictionary<i32, i32>, "l_partkey": dictionary<i32, i32>, "l_suppkey": dictionary<i32, i32>, "l_linenumber": dictionary<i32, i32>, "l_quantity": dictionary<i32, f64>, "l_extendedprice": dictionary<i32, f64>, "l_discount": dictionary<i32, f64>, "l_tax": dictionary<i32, f64>, "l_returnflag": dictionary<i32, memref<1xi8>>, "l_linestatus": dictionary<i32, memref<1xi8>>, "l_shipdate": dictionary<i32, i32>, "l_commitdate": dictionary<i32, i32>, "l_receiptdate": dictionary<i32, i32>, "l_shipinstruct": dictionary<i32, memref<25xi8>>, "l_shipmode": dictionary<i32, memref<10xi8>>, "l_comment": dictionary<i32, memref<44xi8>>, "size": i32> -> dictionary<i32, i32>
  %get_35 = sdql.lookup_dictionary %fieldnode_36 [%i : i32] : dictionary<i32, i32> -> i32
  %fieldnode_38 = sdql.access_record %lineitem "l_shipinstruct" : record<"l_orderkey": dictionary<i32, i32>, "l_partkey": dictionary<i32, i32>, "l_suppkey": dictionary<i32, i32>, "l_linenumber": dictionary<i32, i32>, "l_quantity": dictionary<i32, f64>, "l_extendedprice": dictionary<i32, f64>, "l_discount": dictionary<i32, f64>, "l_tax": dictionary<i32, f64>, "l_returnflag": dictionary<i32, memref<1xi8>>, "l_linestatus": dictionary<i32, memref<1xi8>>, "l_shipdate": dictionary<i32, i32>, "l_commitdate": dictionary<i32, i32>, "l_receiptdate": dictionary<i32, i32>, "l_shipinstruct": dictionary<i32, memref<25xi8>>, "l_shipmode": dictionary<i32, memref<10xi8>>, "l_comment": dictionary<i32, memref<44xi8>>, "size": i32> -> dictionary<i32, memref<25xi8>>
  %get_37 = sdql.lookup_dictionary %fieldnode_38 [%i : i32] : dictionary<i32, memref<25xi8>> -> memref<25xi8>
  %fieldnode_40 = sdql.access_record %lineitem "l_shipmode" : record<"l_orderkey": dictionary<i32, i32>, "l_partkey": dictionary<i32, i32>, "l_suppkey": dictionary<i32, i32>, "l_linenumber": dictionary<i32, i32>, "l_quantity": dictionary<i32, f64>, "l_extendedprice": dictionary<i32, f64>, "l_discount": dictionary<i32, f64>, "l_tax": dictionary<i32, f64>, "l_returnflag": dictionary<i32, memref<1xi8>>, "l_linestatus": dictionary<i32, memref<1xi8>>, "l_shipdate": dictionary<i32, i32>, "l_commitdate": dictionary<i32, i32>, "l_receiptdate": dictionary<i32, i32>, "l_shipinstruct": dictionary<i32, memref<25xi8>>, "l_shipmode": dictionary<i32, memref<10xi8>>, "l_comment": dictionary<i32, memref<44xi8>>, "size": i32> -> dictionary<i32, memref<10xi8>>
  %get_39 = sdql.lookup_dictionary %fieldnode_40 [%i : i32] : dictionary<i32, memref<10xi8>> -> memref<10xi8>
  %fieldnode_42 = sdql.access_record %lineitem "l_comment" : record<"l_orderkey": dictionary<i32, i32>, "l_partkey": dictionary<i32, i32>, "l_suppkey": dictionary<i32, i32>, "l_linenumber": dictionary<i32, i32>, "l_quantity": dictionary<i32, f64>, "l_extendedprice": dictionary<i32, f64>, "l_discount": dictionary<i32, f64>, "l_tax": dictionary<i32, f64>, "l_returnflag": dictionary<i32, memref<1xi8>>, "l_linestatus": dictionary<i32, memref<1xi8>>, "l_shipdate": dictionary<i32, i32>, "l_commitdate": dictionary<i32, i32>, "l_receiptdate": dictionary<i32, i32>, "l_shipinstruct": dictionary<i32, memref<25xi8>>, "l_shipmode": dictionary<i32, memref<10xi8>>, "l_comment": dictionary<i32, memref<44xi8>>, "size": i32> -> dictionary<i32, memref<44xi8>>
  %get_41 = sdql.lookup_dictionary %fieldnode_42 [%i : i32] : dictionary<i32, memref<44xi8>> -> memref<44xi8>
  %l = sdql.create_record {fields = ["l_orderkey", "l_partkey", "l_suppkey", "l_linenumber", "l_quantity", "l_extendedprice", "l_discount", "l_tax", "l_returnflag", "l_linestatus", "l_shipdate", "l_commitdate", "l_receiptdate", "l_shipinstruct", "l_shipmode", "l_comment"]} %get_11, %get_13, %get_15, %get_17, %get_19, %get_21, %get_23, %get_25, %get_27, %get_29, %get_31, %get_33, %get_35, %get_37, %get_39, %get_41 : i32, i32, i32, i32, f64, f64, f64, f64, memref<1xi8>, memref<1xi8>, i32, i32, i32, memref<25xi8>, memref<10xi8>, memref<44xi8> -> record<"l_orderkey": i32, "l_partkey": i32, "l_suppkey": i32, "l_linenumber": i32, "l_quantity": f64, "l_extendedprice": f64, "l_discount": f64, "l_tax": f64, "l_returnflag": memref<1xi8>, "l_linestatus": memref<1xi8>, "l_shipdate": i32, "l_commitdate": i32, "l_receiptdate": i32, "l_shipinstruct": memref<25xi8>, "l_shipmode": memref<10xi8>, "l_comment": memref<44xi8>>
  %fieldnode_43 = sdql.access_record %l "l_shipmode" : record<"l_orderkey": i32, "l_partkey": i32, "l_suppkey": i32, "l_linenumber": i32, "l_quantity": f64, "l_extendedprice": f64, "l_discount": f64, "l_tax": f64, "l_returnflag": memref<1xi8>, "l_linestatus": memref<1xi8>, "l_shipdate": i32, "l_commitdate": i32, "l_receiptdate": i32, "l_shipinstruct": memref<25xi8>, "l_shipmode": memref<10xi8>, "l_comment": memref<44xi8>> -> memref<10xi8>
  %cmp_44 = sdql.cmp %fieldnode_43, %mail : memref<10xi8>, memref<4xi8> -> i1
  %if_45 = "scf.if"(%cmp_44) ({
    %constv_46 = "arith.constant"() <{value = 1 : i1}> : () -> i1
    "scf.yield"(%constv_46) : (i1) -> ()
  }, {
    %fieldnode_47 = sdql.access_record %l "l_shipmode" : record<"l_orderkey": i32, "l_partkey": i32, "l_suppkey": i32, "l_linenumber": i32, "l_quantity": f64, "l_extendedprice": f64, "l_discount": f64, "l_tax": f64, "l_returnflag": memref<1xi8>, "l_linestatus": memref<1xi8>, "l_shipdate": i32, "l_commitdate": i32, "l_receiptdate": i32, "l_shipinstruct": memref<25xi8>, "l_shipmode": memref<10xi8>, "l_comment": memref<44xi8>> -> memref<10xi8>
    %cmp_48 = sdql.cmp %fieldnode_47, %ship : memref<10xi8>, memref<4xi8> -> i1
    "scf.yield"(%cmp_48) : (i1) -> ()
  }) : (i1) -> i1
  %if_49 = "scf.if"(%if_45) ({
    %constda_50 = "arith.constant"() <{value = 19940101 : i32}> : () -> i32
    %fieldnode_51 = sdql.access_record %l "l_receiptdate" : record<"l_orderkey": i32, "l_partkey": i32, "l_suppkey": i32, "l_linenumber": i32, "l_quantity": f64, "l_extendedprice": f64, "l_discount": f64, "l_tax": f64, "l_returnflag": memref<1xi8>, "l_linestatus": memref<1xi8>, "l_shipdate": i32, "l_commitdate": i32, "l_receiptdate": i32, "l_shipinstruct": memref<25xi8>, "l_shipmode": memref<10xi8>, "l_comment": memref<44xi8>> -> i32
    %cmpi_52 = "arith.cmpi"(%constda_50, %fieldnode_51) <{predicate = 3}> : (i32, i32) -> i1
    "scf.yield"(%cmpi_52) : (i1) -> ()
  }, {
    %constv_53 = "arith.constant"() <{value = 0 : i1}> : () -> i1
    "scf.yield"(%constv_53) : (i1) -> ()
  }) : (i1) -> i1
  %if_54 = "scf.if"(%if_49) ({
    %fieldnode_55 = sdql.access_record %l "l_receiptdate" : record<"l_orderkey": i32, "l_partkey": i32, "l_suppkey": i32, "l_linenumber": i32, "l_quantity": f64, "l_extendedprice": f64, "l_discount": f64, "l_tax": f64, "l_returnflag": memref<1xi8>, "l_linestatus": memref<1xi8>, "l_shipdate": i32, "l_commitdate": i32, "l_receiptdate": i32, "l_shipinstruct": memref<25xi8>, "l_shipmode": memref<10xi8>, "l_comment": memref<44xi8>> -> i32
    %constda_56 = "arith.constant"() <{value = 19950101 : i32}> : () -> i32
    %cmpi_57 = "arith.cmpi"(%fieldnode_55, %constda_56) <{predicate = 2}> : (i32, i32) -> i1
    "scf.yield"(%cmpi_57) : (i1) -> ()
  }, {
    %constv_58 = "arith.constant"() <{value = 0 : i1}> : () -> i1
    "scf.yield"(%constv_58) : (i1) -> ()
  }) : (i1) -> i1
  %if_59 = "scf.if"(%if_54) ({
    %fieldnode_60 = sdql.access_record %l "l_shipdate" : record<"l_orderkey": i32, "l_partkey": i32, "l_suppkey": i32, "l_linenumber": i32, "l_quantity": f64, "l_extendedprice": f64, "l_discount": f64, "l_tax": f64, "l_returnflag": memref<1xi8>, "l_linestatus": memref<1xi8>, "l_shipdate": i32, "l_commitdate": i32, "l_receiptdate": i32, "l_shipinstruct": memref<25xi8>, "l_shipmode": memref<10xi8>, "l_comment": memref<44xi8>> -> i32
    %fieldnode_61 = sdql.access_record %l "l_commitdate" : record<"l_orderkey": i32, "l_partkey": i32, "l_suppkey": i32, "l_linenumber": i32, "l_quantity": f64, "l_extendedprice": f64, "l_discount": f64, "l_tax": f64, "l_returnflag": memref<1xi8>, "l_linestatus": memref<1xi8>, "l_shipdate": i32, "l_commitdate": i32, "l_receiptdate": i32, "l_shipinstruct": memref<25xi8>, "l_shipmode": memref<10xi8>, "l_comment": memref<44xi8>> -> i32
    %cmpi_62 = "arith.cmpi"(%fieldnode_60, %fieldnode_61) <{predicate = 2}> : (i32, i32) -> i1
    "scf.yield"(%cmpi_62) : (i1) -> ()
  }, {
    %constv_63 = "arith.constant"() <{value = 0 : i1}> : () -> i1
    "scf.yield"(%constv_63) : (i1) -> ()
  }) : (i1) -> i1
  %if_64 = "scf.if"(%if_59) ({
    %fieldnode_65 = sdql.access_record %l "l_commitdate" : record<"l_orderkey": i32, "l_partkey": i32, "l_suppkey": i32, "l_linenumber": i32, "l_quantity": f64, "l_extendedprice": f64, "l_discount": f64, "l_tax": f64, "l_returnflag": memref<1xi8>, "l_linestatus": memref<1xi8>, "l_shipdate": i32, "l_commitdate": i32, "l_receiptdate": i32, "l_shipinstruct": memref<25xi8>, "l_shipmode": memref<10xi8>, "l_comment": memref<44xi8>> -> i32
    %fieldnode_66 = sdql.access_record %l "l_receiptdate" : record<"l_orderkey": i32, "l_partkey": i32, "l_suppkey": i32, "l_linenumber": i32, "l_quantity": f64, "l_extendedprice": f64, "l_discount": f64, "l_tax": f64, "l_returnflag": memref<1xi8>, "l_linestatus": memref<1xi8>, "l_shipdate": i32, "l_commitdate": i32, "l_receiptdate": i32, "l_shipinstruct": memref<25xi8>, "l_shipmode": memref<10xi8>, "l_comment": memref<44xi8>> -> i32
    %cmpi_67 = "arith.cmpi"(%fieldnode_65, %fieldnode_66) <{predicate = 2}> : (i32, i32) -> i1
    "scf.yield"(%cmpi_67) : (i1) -> ()
  }, {
    %constv_68 = "arith.constant"() <{value = 0 : i1}> : () -> i1
    "scf.yield"(%constv_68) : (i1) -> ()
  }) : (i1) -> i1
  %if_69 = "scf.if"(%if_64) ({
    %fieldnode_71 = sdql.access_record %l "l_orderkey" : record<"l_orderkey": i32, "l_partkey": i32, "l_suppkey": i32, "l_linenumber": i32, "l_quantity": f64, "l_extendedprice": f64, "l_discount": f64, "l_tax": f64, "l_returnflag": memref<1xi8>, "l_linestatus": memref<1xi8>, "l_shipdate": i32, "l_commitdate": i32, "l_receiptdate": i32, "l_shipinstruct": memref<25xi8>, "l_shipmode": memref<10xi8>, "l_comment": memref<44xi8>> -> i32
    %fieldnode_73 = sdql.access_record %l "l_shipmode" : record<"l_orderkey": i32, "l_partkey": i32, "l_suppkey": i32, "l_linenumber": i32, "l_quantity": f64, "l_extendedprice": f64, "l_discount": f64, "l_tax": f64, "l_returnflag": memref<1xi8>, "l_linestatus": memref<1xi8>, "l_shipdate": i32, "l_commitdate": i32, "l_receiptdate": i32, "l_shipinstruct": memref<25xi8>, "l_shipmode": memref<10xi8>, "l_comment": memref<44xi8>> -> memref<10xi8>
    %consti_74 = "arith.constant"() <{value = 1 : i32}> : () -> i32
    %dict_72 = sdql.create_dictionary %fieldnode_73, %consti_74 : memref<10xi8>, i32 -> dictionary<memref<10xi8>, i32>
    %dict_70 = sdql.create_dictionary %fieldnode_71, %dict_72 : i32, dictionary<memref<10xi8>, i32> -> dictionary<i32, dictionary<memref<10xi8>, i32>>
    "scf.yield"(%dict_70) : (dictionary<i32, dictionary<memref<10xi8>, i32>>) -> ()
  }, {
    %dict_75 = sdql.empty_dictionary : dictionary<i32, dictionary<memref<10xi8>, i32>>
    "scf.yield"(%dict_75) : (dictionary<i32, dictionary<memref<10xi8>, i32>>) -> ()
  }) : (i1) -> dictionary<i32, dictionary<memref<10xi8>, i32>>
  sdql.yield %if_69 : dictionary<i32, dictionary<memref<10xi8>, i32>>
}
%fieldnode_76 = sdql.access_record %orders "size" : record<"o_orderkey": dictionary<i32, i32>, "o_custkey": dictionary<i32, i32>, "o_orderstatus": dictionary<i32, memref<1xi8>>, "o_totalprice": dictionary<i32, f64>, "o_orderdate": dictionary<i32, i32>, "o_orderpriority": dictionary<i32, memref<15xi8>>, "o_clerk": dictionary<i32, memref<15xi8>>, "o_shippriority": dictionary<i32, i32>, "o_comment": dictionary<i32, memref<79xi8>>, "size": i32> -> i32
%rangenode_77 = "func.call"(%fieldnode_76) <{callee = @range_builtin}> : (i32) -> dictionary<i32, i32>
%o_h = sdql.sum %rangenode_77 : dictionary<i32, i32> -> dictionary<record<"_" : memref<10xi8>>, record<"high_line_count" : i32, "low_line_count" : i32>> {
^bb_78(%i: i32, %_: i32):
  %fieldnode_80 = sdql.access_record %orders "o_orderkey" : record<"o_orderkey": dictionary<i32, i32>, "o_custkey": dictionary<i32, i32>, "o_orderstatus": dictionary<i32, memref<1xi8>>, "o_totalprice": dictionary<i32, f64>, "o_orderdate": dictionary<i32, i32>, "o_orderpriority": dictionary<i32, memref<15xi8>>, "o_clerk": dictionary<i32, memref<15xi8>>, "o_shippriority": dictionary<i32, i32>, "o_comment": dictionary<i32, memref<79xi8>>, "size": i32> -> dictionary<i32, i32>
  %get_79 = sdql.lookup_dictionary %fieldnode_80 [%i : i32] : dictionary<i32, i32> -> i32
  %fieldnode_82 = sdql.access_record %orders "o_custkey" : record<"o_orderkey": dictionary<i32, i32>, "o_custkey": dictionary<i32, i32>, "o_orderstatus": dictionary<i32, memref<1xi8>>, "o_totalprice": dictionary<i32, f64>, "o_orderdate": dictionary<i32, i32>, "o_orderpriority": dictionary<i32, memref<15xi8>>, "o_clerk": dictionary<i32, memref<15xi8>>, "o_shippriority": dictionary<i32, i32>, "o_comment": dictionary<i32, memref<79xi8>>, "size": i32> -> dictionary<i32, i32>
  %get_81 = sdql.lookup_dictionary %fieldnode_82 [%i : i32] : dictionary<i32, i32> -> i32
  %fieldnode_84 = sdql.access_record %orders "o_orderstatus" : record<"o_orderkey": dictionary<i32, i32>, "o_custkey": dictionary<i32, i32>, "o_orderstatus": dictionary<i32, memref<1xi8>>, "o_totalprice": dictionary<i32, f64>, "o_orderdate": dictionary<i32, i32>, "o_orderpriority": dictionary<i32, memref<15xi8>>, "o_clerk": dictionary<i32, memref<15xi8>>, "o_shippriority": dictionary<i32, i32>, "o_comment": dictionary<i32, memref<79xi8>>, "size": i32> -> dictionary<i32, memref<1xi8>>
  %get_83 = sdql.lookup_dictionary %fieldnode_84 [%i : i32] : dictionary<i32, memref<1xi8>> -> memref<1xi8>
  %fieldnode_86 = sdql.access_record %orders "o_totalprice" : record<"o_orderkey": dictionary<i32, i32>, "o_custkey": dictionary<i32, i32>, "o_orderstatus": dictionary<i32, memref<1xi8>>, "o_totalprice": dictionary<i32, f64>, "o_orderdate": dictionary<i32, i32>, "o_orderpriority": dictionary<i32, memref<15xi8>>, "o_clerk": dictionary<i32, memref<15xi8>>, "o_shippriority": dictionary<i32, i32>, "o_comment": dictionary<i32, memref<79xi8>>, "size": i32> -> dictionary<i32, f64>
  %get_85 = sdql.lookup_dictionary %fieldnode_86 [%i : i32] : dictionary<i32, f64> -> f64
  %fieldnode_88 = sdql.access_record %orders "o_orderdate" : record<"o_orderkey": dictionary<i32, i32>, "o_custkey": dictionary<i32, i32>, "o_orderstatus": dictionary<i32, memref<1xi8>>, "o_totalprice": dictionary<i32, f64>, "o_orderdate": dictionary<i32, i32>, "o_orderpriority": dictionary<i32, memref<15xi8>>, "o_clerk": dictionary<i32, memref<15xi8>>, "o_shippriority": dictionary<i32, i32>, "o_comment": dictionary<i32, memref<79xi8>>, "size": i32> -> dictionary<i32, i32>
  %get_87 = sdql.lookup_dictionary %fieldnode_88 [%i : i32] : dictionary<i32, i32> -> i32
  %fieldnode_90 = sdql.access_record %orders "o_orderpriority" : record<"o_orderkey": dictionary<i32, i32>, "o_custkey": dictionary<i32, i32>, "o_orderstatus": dictionary<i32, memref<1xi8>>, "o_totalprice": dictionary<i32, f64>, "o_orderdate": dictionary<i32, i32>, "o_orderpriority": dictionary<i32, memref<15xi8>>, "o_clerk": dictionary<i32, memref<15xi8>>, "o_shippriority": dictionary<i32, i32>, "o_comment": dictionary<i32, memref<79xi8>>, "size": i32> -> dictionary<i32, memref<15xi8>>
  %get_89 = sdql.lookup_dictionary %fieldnode_90 [%i : i32] : dictionary<i32, memref<15xi8>> -> memref<15xi8>
  %fieldnode_92 = sdql.access_record %orders "o_clerk" : record<"o_orderkey": dictionary<i32, i32>, "o_custkey": dictionary<i32, i32>, "o_orderstatus": dictionary<i32, memref<1xi8>>, "o_totalprice": dictionary<i32, f64>, "o_orderdate": dictionary<i32, i32>, "o_orderpriority": dictionary<i32, memref<15xi8>>, "o_clerk": dictionary<i32, memref<15xi8>>, "o_shippriority": dictionary<i32, i32>, "o_comment": dictionary<i32, memref<79xi8>>, "size": i32> -> dictionary<i32, memref<15xi8>>
  %get_91 = sdql.lookup_dictionary %fieldnode_92 [%i : i32] : dictionary<i32, memref<15xi8>> -> memref<15xi8>
  %fieldnode_94 = sdql.access_record %orders "o_shippriority" : record<"o_orderkey": dictionary<i32, i32>, "o_custkey": dictionary<i32, i32>, "o_orderstatus": dictionary<i32, memref<1xi8>>, "o_totalprice": dictionary<i32, f64>, "o_orderdate": dictionary<i32, i32>, "o_orderpriority": dictionary<i32, memref<15xi8>>, "o_clerk": dictionary<i32, memref<15xi8>>, "o_shippriority": dictionary<i32, i32>, "o_comment": dictionary<i32, memref<79xi8>>, "size": i32> -> dictionary<i32, i32>
  %get_93 = sdql.lookup_dictionary %fieldnode_94 [%i : i32] : dictionary<i32, i32> -> i32
  %fieldnode_96 = sdql.access_record %orders "o_comment" : record<"o_orderkey": dictionary<i32, i32>, "o_custkey": dictionary<i32, i32>, "o_orderstatus": dictionary<i32, memref<1xi8>>, "o_totalprice": dictionary<i32, f64>, "o_orderdate": dictionary<i32, i32>, "o_orderpriority": dictionary<i32, memref<15xi8>>, "o_clerk": dictionary<i32, memref<15xi8>>, "o_shippriority": dictionary<i32, i32>, "o_comment": dictionary<i32, memref<79xi8>>, "size": i32> -> dictionary<i32, memref<79xi8>>
  %get_95 = sdql.lookup_dictionary %fieldnode_96 [%i : i32] : dictionary<i32, memref<79xi8>> -> memref<79xi8>
  %o = sdql.create_record {fields = ["o_orderkey", "o_custkey", "o_orderstatus", "o_totalprice", "o_orderdate", "o_orderpriority", "o_clerk", "o_shippriority", "o_comment"]} %get_79, %get_81, %get_83, %get_85, %get_87, %get_89, %get_91, %get_93, %get_95 : i32, i32, memref<1xi8>, f64, i32, memref<15xi8>, memref<15xi8>, i32, memref<79xi8> -> record<"o_orderkey": i32, "o_custkey": i32, "o_orderstatus": memref<1xi8>, "o_totalprice": f64, "o_orderdate": i32, "o_orderpriority": memref<15xi8>, "o_clerk": memref<15xi8>, "o_shippriority": i32, "o_comment": memref<79xi8>>
  %fieldnode_100 = sdql.access_record %o "o_orderkey" : record<"o_orderkey": i32, "o_custkey": i32, "o_orderstatus": memref<1xi8>, "o_totalprice": f64, "o_orderdate": i32, "o_orderpriority": memref<15xi8>, "o_clerk": memref<15xi8>, "o_shippriority": i32, "o_comment": memref<79xi8>> -> i32
  %get_99 = sdql.lookup_dictionary %l_h [%fieldnode_100 : i32] : dictionary<i32, dictionary<memref<10xi8>, i32>> -> dictionary<memref<10xi8>, i32>
  %dict_101 = sdql.empty_dictionary : dictionary<memref<10xi8>, i32>
  %cmp_102 = sdql.cmp %get_99, %dict_101 : dictionary<memref<10xi8>, i32>, dictionary<memref<10xi8>, i32> -> i1
  %zero_97 = "arith.constant"() <{value = 0 : i1}> : () -> i1
  %neg_98 = "arith.cmpi"(%cmp_102, %zero_97) <{predicate = 0}> : (i1, i1) -> i1
  %if_103 = "scf.if"(%neg_98) ({
    %fieldnode_105 = sdql.access_record %o "o_orderkey" : record<"o_orderkey": i32, "o_custkey": i32, "o_orderstatus": memref<1xi8>, "o_totalprice": f64, "o_orderdate": i32, "o_orderpriority": memref<15xi8>, "o_clerk": memref<15xi8>, "o_shippriority": i32, "o_comment": memref<79xi8>> -> i32
    %get_104 = sdql.lookup_dictionary %l_h [%fieldnode_105 : i32] : dictionary<i32, dictionary<memref<10xi8>, i32>> -> dictionary<memref<10xi8>, i32>
    %sum_106 = sdql.sum %get_104 : dictionary<memref<10xi8>, i32> -> dictionary<record<"_" : memref<10xi8>>, record<"high_line_count" : i32, "low_line_count" : i32>> {
    ^bb_107(%l_shipmode: memref<10xi8>, %c: i32):
      %recnode_109 = sdql.create_record {fields = ["_"]} %l_shipmode : memref<10xi8> -> record<"_": memref<10xi8>>
      %fieldnode_110 = sdql.access_record %o "o_orderpriority" : record<"o_orderkey": i32, "o_custkey": i32, "o_orderstatus": memref<1xi8>, "o_totalprice": f64, "o_orderdate": i32, "o_orderpriority": memref<15xi8>, "o_clerk": memref<15xi8>, "o_shippriority": i32, "o_comment": memref<79xi8>> -> memref<15xi8>
      %cmp_111 = sdql.cmp %fieldnode_110, %urgent : memref<15xi8>, memref<8xi8> -> i1
      %if_112 = "scf.if"(%cmp_111) ({
        %constv_113 = "arith.constant"() <{value = 1 : i1}> : () -> i1
        "scf.yield"(%constv_113) : (i1) -> ()
      }, {
        %fieldnode_114 = sdql.access_record %o "o_orderpriority" : record<"o_orderkey": i32, "o_custkey": i32, "o_orderstatus": memref<1xi8>, "o_totalprice": f64, "o_orderdate": i32, "o_orderpriority": memref<15xi8>, "o_clerk": memref<15xi8>, "o_shippriority": i32, "o_comment": memref<79xi8>> -> memref<15xi8>
        %cmp_115 = sdql.cmp %fieldnode_114, %high : memref<15xi8>, memref<6xi8> -> i1
        "scf.yield"(%cmp_115) : (i1) -> ()
      }) : (i1) -> i1
      %if_116 = "scf.if"(%if_112) ({
        "scf.yield"(%c) : (i32) -> ()
      }, {
        %consti_117 = "arith.constant"() <{value = 0 : i32}> : () -> i32
        "scf.yield"(%consti_117) : (i32) -> ()
      }) : (i1) -> i32
      %fieldnode_120 = sdql.access_record %o "o_orderpriority" : record<"o_orderkey": i32, "o_custkey": i32, "o_orderstatus": memref<1xi8>, "o_totalprice": f64, "o_orderdate": i32, "o_orderpriority": memref<15xi8>, "o_clerk": memref<15xi8>, "o_shippriority": i32, "o_comment": memref<79xi8>> -> memref<15xi8>
      %cmp_121 = sdql.cmp %fieldnode_120, %urgent : memref<15xi8>, memref<8xi8> -> i1
      %zero_118 = "arith.constant"() <{value = 0 : i1}> : () -> i1
      %neg_119 = "arith.cmpi"(%cmp_121, %zero_118) <{predicate = 0}> : (i1, i1) -> i1
      %if_122 = "scf.if"(%neg_119) ({
        %fieldnode_125 = sdql.access_record %o "o_orderpriority" : record<"o_orderkey": i32, "o_custkey": i32, "o_orderstatus": memref<1xi8>, "o_totalprice": f64, "o_orderdate": i32, "o_orderpriority": memref<15xi8>, "o_clerk": memref<15xi8>, "o_shippriority": i32, "o_comment": memref<79xi8>> -> memref<15xi8>
        %cmp_126 = sdql.cmp %fieldnode_125, %high : memref<15xi8>, memref<6xi8> -> i1
        %zero_123 = "arith.constant"() <{value = 0 : i1}> : () -> i1
        %neg_124 = "arith.cmpi"(%cmp_126, %zero_123) <{predicate = 0}> : (i1, i1) -> i1
        "scf.yield"(%neg_124) : (i1) -> ()
      }, {
        %constv_127 = "arith.constant"() <{value = 0 : i1}> : () -> i1
        "scf.yield"(%constv_127) : (i1) -> ()
      }) : (i1) -> i1
      %if_128 = "scf.if"(%if_122) ({
        "scf.yield"(%c) : (i32) -> ()
      }, {
        %consti_129 = "arith.constant"() <{value = 0 : i32}> : () -> i32
        "scf.yield"(%consti_129) : (i32) -> ()
      }) : (i1) -> i32
      %recnode_130 = sdql.create_record {fields = ["high_line_count", "low_line_count"]} %if_116, %if_128 : i32, i32 -> record<"high_line_count": i32, "low_line_count": i32>
      %dict_108 = sdql.create_dictionary %recnode_109, %recnode_130 : record<"_" : memref<10xi8>>, record<"high_line_count" : i32, "low_line_count" : i32> -> dictionary<record<"_" : memref<10xi8>>, record<"high_line_count" : i32, "low_line_count" : i32>>
      sdql.yield %dict_108 : dictionary<record<"_" : memref<10xi8>>, record<"high_line_count" : i32, "low_line_count" : i32>>
    }
    "scf.yield"(%sum_106) : (dictionary<record<"_" : memref<10xi8>>, record<"high_line_count" : i32, "low_line_count" : i32>>) -> ()
  }, {
    %dict_131 = sdql.empty_dictionary : dictionary<record<"_" : memref<10xi8>>, record<"high_line_count" : i32, "low_line_count" : i32>>
    "scf.yield"(%dict_131) : (dictionary<record<"_" : memref<10xi8>>, record<"high_line_count" : i32, "low_line_count" : i32>>) -> ()
  }) : (i1) -> dictionary<record<"_" : memref<10xi8>>, record<"high_line_count" : i32, "low_line_count" : i32>>
  sdql.yield %if_103 : dictionary<record<"_" : memref<10xi8>>, record<"high_line_count" : i32, "low_line_count" : i32>>
}
%sum_132 = sdql.sum %o_h : dictionary<record<"_" : memref<10xi8>>, record<"high_line_count" : i32, "low_line_count" : i32>> -> dictionary<record<"_" : memref<10xi8>, "high_line_count" : i32, "low_line_count" : i32>, i32> {
^bb_133(%k: record<"_" : memref<10xi8>>, %v: record<"high_line_count" : i32, "low_line_count" : i32>):
  %concat_135 = sdql.concat %k, %v : record<"_" : memref<10xi8>>, record<"high_line_count" : i32, "low_line_count" : i32> -> record<"_" : memref<10xi8>, "high_line_count" : i32, "low_line_count" : i32>
  %consti_136 = "arith.constant"() <{value = 1 : i32}> : () -> i32
  %dict_134 = sdql.create_dictionary %concat_135, %consti_136 : record<"_" : memref<10xi8>, "high_line_count" : i32, "low_line_count" : i32>, i32 -> dictionary<record<"_" : memref<10xi8>, "high_line_count" : i32, "low_line_count" : i32>, i32>
  sdql.yield %dict_134 : dictionary<record<"_" : memref<10xi8>, "high_line_count" : i32, "low_line_count" : i32>, i32>
}
