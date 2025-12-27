func.func @range_builtin(%n: i32) -> dictionary<i32, i32> {
  %zero = "arith.constant"() <{value = 0}> : () -> i32
  %is_terminal = "arith.cmpi"(%n, %zero) <{"predicate" = 3}> : (i32, i32) -> i1
  %res = "scf.if"(%is_terminal) ({
    %empty = sdql.empty_dictionary : dictionary<i32, i32>
    "scf.yield"(%empty) : (dictionary<i32, i32>) -> ()
  }, {
    %one = "arith.constant"() <{value = 1}> : () -> i32
    %smaller_n = "arith.subi"(%n, %one) : (i32, i32) -> i32
    %prev_range = "func.call"(%smaller_n) <{callee = @range_builtin}> : (i32) -> dictionary<i32, i32>
    %true = "arith.constant"() <{value = 1}> : () -> i32
    %extension = sdql.create_dictionary %smaller_n, %true : i32, i32 -> dictionary<i32, i32>

    // return %extension + %prev_range
    %added = sdql.dictionary_add %extension %prev_range : dictionary<i32, i32>, dictionary<i32, i32> -> dictionary<i32, i32>
    "scf.yield"(%added) : (dictionary<i32, i32>) -> ()
  }) : (i1) -> dictionary<i32, i32>
  func.return %res : dictionary<i32, i32>
}
%supplier = sdql.load "datasets/tpch/supplier.tbl" : record<"s_suppkey" : dictionary<i32, i32>, "s_name" : dictionary<i32, memref<25xi8>>, "s_address" : dictionary<i32, memref<40xi8>>, "s_nationkey" : dictionary<i32, i32>, "s_phone" : dictionary<i32, memref<15xi8>>, "s_acctbal" : dictionary<i32, f64>, "s_comment" : dictionary<i32, memref<101xi8>>, "size" : i32>
%lineitem = sdql.load "datasets/tpch/lineitem.tbl" : record<"l_orderkey" : dictionary<i32, i32>, "l_partkey" : dictionary<i32, i32>, "l_suppkey" : dictionary<i32, i32>, "l_linenumber" : dictionary<i32, i32>, "l_quantity" : dictionary<i32, f64>, "l_extendedprice" : dictionary<i32, f64>, "l_discount" : dictionary<i32, f64>, "l_tax" : dictionary<i32, f64>, "l_returnflag" : dictionary<i32, memref<1xi8>>, "l_linestatus" : dictionary<i32, memref<1xi8>>, "l_shipdate" : dictionary<i32, i32>, "l_commitdate" : dictionary<i32, i32>, "l_receiptdate" : dictionary<i32, i32>, "l_shipinstruct" : dictionary<i32, memref<25xi8>>, "l_shipmode" : dictionary<i32, memref<10xi8>>, "l_comment" : dictionary<i32, memref<44xi8>>, "size" : i32>
%orders = sdql.load "datasets/tpch/orders.tbl" : record<"o_orderkey" : dictionary<i32, i32>, "o_custkey" : dictionary<i32, i32>, "o_orderstatus" : dictionary<i32, memref<1xi8>>, "o_totalprice" : dictionary<i32, f64>, "o_orderdate" : dictionary<i32, i32>, "o_orderpriority" : dictionary<i32, memref<15xi8>>, "o_clerk" : dictionary<i32, memref<15xi8>>, "o_shippriority" : dictionary<i32, i32>, "o_comment" : dictionary<i32, memref<79xi8>>, "size" : i32>
%customer = sdql.load "datasets/tpch/customer.tbl" : record<"c_custkey" : dictionary<i32, i32>, "c_name" : dictionary<i32, memref<25xi8>>, "c_address" : dictionary<i32, memref<40xi8>>, "c_nationkey" : dictionary<i32, i32>, "c_phone" : dictionary<i32, memref<15xi8>>, "c_acctbal" : dictionary<i32, f64>, "c_mktsegment" : dictionary<i32, memref<10xi8>>, "c_comment" : dictionary<i32, memref<117xi8>>, "size" : i32>
%nation = sdql.load "datasets/tpch/nation.tbl" : record<"n_nationkey" : dictionary<i32, i32>, "n_name" : dictionary<i32, memref<25xi8>>, "n_regionkey" : dictionary<i32, i32>, "n_comment" : dictionary<i32, memref<152xi8>>, "size" : i32>
// FRANCE
%constds_0 = "arith.constant"() <{value = dense<[70, 82, 65, 78, 67, 69]> : memref<?xi8>}> : () -> memref<?xi8>
%consti_1 = "arith.constant"() <{value = 7 : i32}> : () -> i32
%france = sdql.external "ConstantString", %constds_0, %consti_1 : memref<?xi8>, i32 -> memref<6xi8>
// GERMANY
%constds_2 = "arith.constant"() <{value = dense<[71, 69, 82, 77, 65, 78, 89]> : memref<?xi8>}> : () -> memref<?xi8>
%consti_3 = "arith.constant"() <{value = 8 : i32}> : () -> i32
%germany = sdql.external "ConstantString", %constds_2, %consti_3 : memref<?xi8>, i32 -> memref<7xi8>
%fieldnode_4 = sdql.access_record %nation "size" : record<"n_nationkey": dictionary<i32, i32>, "n_name": dictionary<i32, memref<25xi8>>, "n_regionkey": dictionary<i32, i32>, "n_comment": dictionary<i32, memref<152xi8>>, "size": i32> -> i32
%rangenode_5 = "func.call"(%fieldnode_4) <{callee = @range_builtin}> : (i32) -> dictionary<i32, i32>
%nationkey_to_name = sdql.sum %rangenode_5 : dictionary<i32, i32> -> dictionary<i32, record<"_" : memref<25xi8>>> {
^bb_6(%i: i32, %_: i32):
  %fieldnode_8 = sdql.access_record %nation "n_nationkey" : record<"n_nationkey": dictionary<i32, i32>, "n_name": dictionary<i32, memref<25xi8>>, "n_regionkey": dictionary<i32, i32>, "n_comment": dictionary<i32, memref<152xi8>>, "size": i32> -> dictionary<i32, i32>
  %get_7 = sdql.lookup_dictionary %fieldnode_8 [%i : i32] : dictionary<i32, i32> -> i32
  %fieldnode_10 = sdql.access_record %nation "n_name" : record<"n_nationkey": dictionary<i32, i32>, "n_name": dictionary<i32, memref<25xi8>>, "n_regionkey": dictionary<i32, i32>, "n_comment": dictionary<i32, memref<152xi8>>, "size": i32> -> dictionary<i32, memref<25xi8>>
  %get_9 = sdql.lookup_dictionary %fieldnode_10 [%i : i32] : dictionary<i32, memref<25xi8>> -> memref<25xi8>
  %fieldnode_12 = sdql.access_record %nation "n_regionkey" : record<"n_nationkey": dictionary<i32, i32>, "n_name": dictionary<i32, memref<25xi8>>, "n_regionkey": dictionary<i32, i32>, "n_comment": dictionary<i32, memref<152xi8>>, "size": i32> -> dictionary<i32, i32>
  %get_11 = sdql.lookup_dictionary %fieldnode_12 [%i : i32] : dictionary<i32, i32> -> i32
  %fieldnode_14 = sdql.access_record %nation "n_comment" : record<"n_nationkey": dictionary<i32, i32>, "n_name": dictionary<i32, memref<25xi8>>, "n_regionkey": dictionary<i32, i32>, "n_comment": dictionary<i32, memref<152xi8>>, "size": i32> -> dictionary<i32, memref<152xi8>>
  %get_13 = sdql.lookup_dictionary %fieldnode_14 [%i : i32] : dictionary<i32, memref<152xi8>> -> memref<152xi8>
  %n = sdql.create_record {fields = ["n_nationkey", "n_name", "n_regionkey", "n_comment"]} %get_7, %get_9, %get_11, %get_13 : i32, memref<25xi8>, i32, memref<152xi8> -> record<"n_nationkey": i32, "n_name": memref<25xi8>, "n_regionkey": i32, "n_comment": memref<152xi8>>
  %fieldnode_15 = sdql.access_record %n "n_name" : record<"n_nationkey": i32, "n_name": memref<25xi8>, "n_regionkey": i32, "n_comment": memref<152xi8>> -> memref<25xi8>
  %cmp_16 = sdql.cmp %fieldnode_15, %france : memref<25xi8>, memref<6xi8> -> i1
  %if_17 = "scf.if"(%cmp_16) ({
    %constv_18 = "arith.constant"() <{value = 1 : i1}> : () -> i1
    "scf.yield"(%constv_18) : (i1) -> ()
  }, {
    %fieldnode_19 = sdql.access_record %n "n_name" : record<"n_nationkey": i32, "n_name": memref<25xi8>, "n_regionkey": i32, "n_comment": memref<152xi8>> -> memref<25xi8>
    %cmp_20 = sdql.cmp %fieldnode_19, %germany : memref<25xi8>, memref<7xi8> -> i1
    "scf.yield"(%cmp_20) : (i1) -> ()
  }) : (i1) -> i1
  %if_21 = "scf.if"(%if_17) ({
    %fieldnode_24 = sdql.access_record %n "n_nationkey" : record<"n_nationkey": i32, "n_name": memref<25xi8>, "n_regionkey": i32, "n_comment": memref<152xi8>> -> i32
    %uniq_23 = sdql.unique %fieldnode_24 : i32 -> i32
    %fieldnode_25 = sdql.access_record %n "n_name" : record<"n_nationkey": i32, "n_name": memref<25xi8>, "n_regionkey": i32, "n_comment": memref<152xi8>> -> memref<25xi8>
    %recnode_26 = sdql.create_record {fields = ["_"]} %fieldnode_25 : memref<25xi8> -> record<"_": memref<25xi8>>
    %dict_22 = sdql.create_dictionary %uniq_23, %recnode_26 : i32, record<"_" : memref<25xi8>> -> dictionary<i32, record<"_" : memref<25xi8>>>
    "scf.yield"(%dict_22) : (dictionary<i32, record<"_" : memref<25xi8>>>) -> ()
  }, {
    %dict_27 = sdql.empty_dictionary : dictionary<i32, record<"_" : memref<25xi8>>>
    "scf.yield"(%dict_27) : (dictionary<i32, record<"_" : memref<25xi8>>>) -> ()
  }) : (i1) -> dictionary<i32, record<"_" : memref<25xi8>>>
  sdql.yield %if_21 : dictionary<i32, record<"_" : memref<25xi8>>>
}
%fieldnode_28 = sdql.access_record %customer "size" : record<"c_custkey": dictionary<i32, i32>, "c_name": dictionary<i32, memref<25xi8>>, "c_address": dictionary<i32, memref<40xi8>>, "c_nationkey": dictionary<i32, i32>, "c_phone": dictionary<i32, memref<15xi8>>, "c_acctbal": dictionary<i32, f64>, "c_mktsegment": dictionary<i32, memref<10xi8>>, "c_comment": dictionary<i32, memref<117xi8>>, "size": i32> -> i32
%rangenode_29 = "func.call"(%fieldnode_28) <{callee = @range_builtin}> : (i32) -> dictionary<i32, i32>
%custkey_to_name = sdql.sum %rangenode_29 : dictionary<i32, i32> -> dictionary<i32, memref<25xi8>> {
^bb_30(%i: i32, %_: i32):
  %fieldnode_32 = sdql.access_record %customer "c_custkey" : record<"c_custkey": dictionary<i32, i32>, "c_name": dictionary<i32, memref<25xi8>>, "c_address": dictionary<i32, memref<40xi8>>, "c_nationkey": dictionary<i32, i32>, "c_phone": dictionary<i32, memref<15xi8>>, "c_acctbal": dictionary<i32, f64>, "c_mktsegment": dictionary<i32, memref<10xi8>>, "c_comment": dictionary<i32, memref<117xi8>>, "size": i32> -> dictionary<i32, i32>
  %get_31 = sdql.lookup_dictionary %fieldnode_32 [%i : i32] : dictionary<i32, i32> -> i32
  %fieldnode_34 = sdql.access_record %customer "c_name" : record<"c_custkey": dictionary<i32, i32>, "c_name": dictionary<i32, memref<25xi8>>, "c_address": dictionary<i32, memref<40xi8>>, "c_nationkey": dictionary<i32, i32>, "c_phone": dictionary<i32, memref<15xi8>>, "c_acctbal": dictionary<i32, f64>, "c_mktsegment": dictionary<i32, memref<10xi8>>, "c_comment": dictionary<i32, memref<117xi8>>, "size": i32> -> dictionary<i32, memref<25xi8>>
  %get_33 = sdql.lookup_dictionary %fieldnode_34 [%i : i32] : dictionary<i32, memref<25xi8>> -> memref<25xi8>
  %fieldnode_36 = sdql.access_record %customer "c_address" : record<"c_custkey": dictionary<i32, i32>, "c_name": dictionary<i32, memref<25xi8>>, "c_address": dictionary<i32, memref<40xi8>>, "c_nationkey": dictionary<i32, i32>, "c_phone": dictionary<i32, memref<15xi8>>, "c_acctbal": dictionary<i32, f64>, "c_mktsegment": dictionary<i32, memref<10xi8>>, "c_comment": dictionary<i32, memref<117xi8>>, "size": i32> -> dictionary<i32, memref<40xi8>>
  %get_35 = sdql.lookup_dictionary %fieldnode_36 [%i : i32] : dictionary<i32, memref<40xi8>> -> memref<40xi8>
  %fieldnode_38 = sdql.access_record %customer "c_nationkey" : record<"c_custkey": dictionary<i32, i32>, "c_name": dictionary<i32, memref<25xi8>>, "c_address": dictionary<i32, memref<40xi8>>, "c_nationkey": dictionary<i32, i32>, "c_phone": dictionary<i32, memref<15xi8>>, "c_acctbal": dictionary<i32, f64>, "c_mktsegment": dictionary<i32, memref<10xi8>>, "c_comment": dictionary<i32, memref<117xi8>>, "size": i32> -> dictionary<i32, i32>
  %get_37 = sdql.lookup_dictionary %fieldnode_38 [%i : i32] : dictionary<i32, i32> -> i32
  %fieldnode_40 = sdql.access_record %customer "c_phone" : record<"c_custkey": dictionary<i32, i32>, "c_name": dictionary<i32, memref<25xi8>>, "c_address": dictionary<i32, memref<40xi8>>, "c_nationkey": dictionary<i32, i32>, "c_phone": dictionary<i32, memref<15xi8>>, "c_acctbal": dictionary<i32, f64>, "c_mktsegment": dictionary<i32, memref<10xi8>>, "c_comment": dictionary<i32, memref<117xi8>>, "size": i32> -> dictionary<i32, memref<15xi8>>
  %get_39 = sdql.lookup_dictionary %fieldnode_40 [%i : i32] : dictionary<i32, memref<15xi8>> -> memref<15xi8>
  %fieldnode_42 = sdql.access_record %customer "c_acctbal" : record<"c_custkey": dictionary<i32, i32>, "c_name": dictionary<i32, memref<25xi8>>, "c_address": dictionary<i32, memref<40xi8>>, "c_nationkey": dictionary<i32, i32>, "c_phone": dictionary<i32, memref<15xi8>>, "c_acctbal": dictionary<i32, f64>, "c_mktsegment": dictionary<i32, memref<10xi8>>, "c_comment": dictionary<i32, memref<117xi8>>, "size": i32> -> dictionary<i32, f64>
  %get_41 = sdql.lookup_dictionary %fieldnode_42 [%i : i32] : dictionary<i32, f64> -> f64
  %fieldnode_44 = sdql.access_record %customer "c_mktsegment" : record<"c_custkey": dictionary<i32, i32>, "c_name": dictionary<i32, memref<25xi8>>, "c_address": dictionary<i32, memref<40xi8>>, "c_nationkey": dictionary<i32, i32>, "c_phone": dictionary<i32, memref<15xi8>>, "c_acctbal": dictionary<i32, f64>, "c_mktsegment": dictionary<i32, memref<10xi8>>, "c_comment": dictionary<i32, memref<117xi8>>, "size": i32> -> dictionary<i32, memref<10xi8>>
  %get_43 = sdql.lookup_dictionary %fieldnode_44 [%i : i32] : dictionary<i32, memref<10xi8>> -> memref<10xi8>
  %fieldnode_46 = sdql.access_record %customer "c_comment" : record<"c_custkey": dictionary<i32, i32>, "c_name": dictionary<i32, memref<25xi8>>, "c_address": dictionary<i32, memref<40xi8>>, "c_nationkey": dictionary<i32, i32>, "c_phone": dictionary<i32, memref<15xi8>>, "c_acctbal": dictionary<i32, f64>, "c_mktsegment": dictionary<i32, memref<10xi8>>, "c_comment": dictionary<i32, memref<117xi8>>, "size": i32> -> dictionary<i32, memref<117xi8>>
  %get_45 = sdql.lookup_dictionary %fieldnode_46 [%i : i32] : dictionary<i32, memref<117xi8>> -> memref<117xi8>
  %c = sdql.create_record {fields = ["c_custkey", "c_name", "c_address", "c_nationkey", "c_phone", "c_acctbal", "c_mktsegment", "c_comment"]} %get_31, %get_33, %get_35, %get_37, %get_39, %get_41, %get_43, %get_45 : i32, memref<25xi8>, memref<40xi8>, i32, memref<15xi8>, f64, memref<10xi8>, memref<117xi8> -> record<"c_custkey": i32, "c_name": memref<25xi8>, "c_address": memref<40xi8>, "c_nationkey": i32, "c_phone": memref<15xi8>, "c_acctbal": f64, "c_mktsegment": memref<10xi8>, "c_comment": memref<117xi8>>
  %fieldnode_50 = sdql.access_record %c "c_nationkey" : record<"c_custkey": i32, "c_name": memref<25xi8>, "c_address": memref<40xi8>, "c_nationkey": i32, "c_phone": memref<15xi8>, "c_acctbal": f64, "c_mktsegment": memref<10xi8>, "c_comment": memref<117xi8>> -> i32
  %get_49 = sdql.lookup_dictionary %nationkey_to_name [%fieldnode_50 : i32] : dictionary<i32, record<"_" : memref<25xi8>>> -> record<"_" : memref<25xi8>>
  %dict_51 = sdql.empty_dictionary : record<"_" : memref<25xi8>>
  %cmp_52 = sdql.cmp %get_49, %dict_51 : record<"_" : memref<25xi8>>, record<"_" : memref<25xi8>> -> i1
  %zero_47 = "arith.constant"() <{value = 0 : i1}> : () -> i1
  %neg_48 = "arith.cmpi"(%cmp_52, %zero_47) <{predicate = 0}> : (i1, i1) -> i1
  %if_53 = "scf.if"(%neg_48) ({
    %fieldnode_56 = sdql.access_record %c "c_custkey" : record<"c_custkey": i32, "c_name": memref<25xi8>, "c_address": memref<40xi8>, "c_nationkey": i32, "c_phone": memref<15xi8>, "c_acctbal": f64, "c_mktsegment": memref<10xi8>, "c_comment": memref<117xi8>> -> i32
    %uniq_55 = sdql.unique %fieldnode_56 : i32 -> i32
    %fieldnode_59 = sdql.access_record %c "c_nationkey" : record<"c_custkey": i32, "c_name": memref<25xi8>, "c_address": memref<40xi8>, "c_nationkey": i32, "c_phone": memref<15xi8>, "c_acctbal": f64, "c_mktsegment": memref<10xi8>, "c_comment": memref<117xi8>> -> i32
    %get_58 = sdql.lookup_dictionary %nationkey_to_name [%fieldnode_59 : i32] : dictionary<i32, record<"_" : memref<25xi8>>> -> record<"_" : memref<25xi8>>
    %consti_60 = "arith.constant"() <{value = 0 : i32}> : () -> i32
    %get_57 = sdql.access_record %get_58 "%consti_60" : record<"_" : memref<25xi8>> -> memref<25xi8>
    %dict_54 = sdql.create_dictionary %uniq_55, %get_57 : i32, memref<25xi8> -> dictionary<i32, memref<25xi8>>
    "scf.yield"(%dict_54) : (dictionary<i32, memref<25xi8>>) -> ()
  }, {
    %dict_61 = sdql.empty_dictionary : dictionary<i32, memref<25xi8>>
    "scf.yield"(%dict_61) : (dictionary<i32, memref<25xi8>>) -> ()
  }) : (i1) -> dictionary<i32, memref<25xi8>>
  sdql.yield %if_53 : dictionary<i32, memref<25xi8>>
}
%fieldnode_62 = sdql.access_record %orders "size" : record<"o_orderkey": dictionary<i32, i32>, "o_custkey": dictionary<i32, i32>, "o_orderstatus": dictionary<i32, memref<1xi8>>, "o_totalprice": dictionary<i32, f64>, "o_orderdate": dictionary<i32, i32>, "o_orderpriority": dictionary<i32, memref<15xi8>>, "o_clerk": dictionary<i32, memref<15xi8>>, "o_shippriority": dictionary<i32, i32>, "o_comment": dictionary<i32, memref<79xi8>>, "size": i32> -> i32
%rangenode_63 = "func.call"(%fieldnode_62) <{callee = @range_builtin}> : (i32) -> dictionary<i32, i32>
%orderkey_to_name = sdql.sum %rangenode_63 : dictionary<i32, i32> -> dictionary<i32, memref<25xi8>> {
^bb_64(%i: i32, %_: i32):
  %fieldnode_66 = sdql.access_record %orders "o_orderkey" : record<"o_orderkey": dictionary<i32, i32>, "o_custkey": dictionary<i32, i32>, "o_orderstatus": dictionary<i32, memref<1xi8>>, "o_totalprice": dictionary<i32, f64>, "o_orderdate": dictionary<i32, i32>, "o_orderpriority": dictionary<i32, memref<15xi8>>, "o_clerk": dictionary<i32, memref<15xi8>>, "o_shippriority": dictionary<i32, i32>, "o_comment": dictionary<i32, memref<79xi8>>, "size": i32> -> dictionary<i32, i32>
  %get_65 = sdql.lookup_dictionary %fieldnode_66 [%i : i32] : dictionary<i32, i32> -> i32
  %fieldnode_68 = sdql.access_record %orders "o_custkey" : record<"o_orderkey": dictionary<i32, i32>, "o_custkey": dictionary<i32, i32>, "o_orderstatus": dictionary<i32, memref<1xi8>>, "o_totalprice": dictionary<i32, f64>, "o_orderdate": dictionary<i32, i32>, "o_orderpriority": dictionary<i32, memref<15xi8>>, "o_clerk": dictionary<i32, memref<15xi8>>, "o_shippriority": dictionary<i32, i32>, "o_comment": dictionary<i32, memref<79xi8>>, "size": i32> -> dictionary<i32, i32>
  %get_67 = sdql.lookup_dictionary %fieldnode_68 [%i : i32] : dictionary<i32, i32> -> i32
  %fieldnode_70 = sdql.access_record %orders "o_orderstatus" : record<"o_orderkey": dictionary<i32, i32>, "o_custkey": dictionary<i32, i32>, "o_orderstatus": dictionary<i32, memref<1xi8>>, "o_totalprice": dictionary<i32, f64>, "o_orderdate": dictionary<i32, i32>, "o_orderpriority": dictionary<i32, memref<15xi8>>, "o_clerk": dictionary<i32, memref<15xi8>>, "o_shippriority": dictionary<i32, i32>, "o_comment": dictionary<i32, memref<79xi8>>, "size": i32> -> dictionary<i32, memref<1xi8>>
  %get_69 = sdql.lookup_dictionary %fieldnode_70 [%i : i32] : dictionary<i32, memref<1xi8>> -> memref<1xi8>
  %fieldnode_72 = sdql.access_record %orders "o_totalprice" : record<"o_orderkey": dictionary<i32, i32>, "o_custkey": dictionary<i32, i32>, "o_orderstatus": dictionary<i32, memref<1xi8>>, "o_totalprice": dictionary<i32, f64>, "o_orderdate": dictionary<i32, i32>, "o_orderpriority": dictionary<i32, memref<15xi8>>, "o_clerk": dictionary<i32, memref<15xi8>>, "o_shippriority": dictionary<i32, i32>, "o_comment": dictionary<i32, memref<79xi8>>, "size": i32> -> dictionary<i32, f64>
  %get_71 = sdql.lookup_dictionary %fieldnode_72 [%i : i32] : dictionary<i32, f64> -> f64
  %fieldnode_74 = sdql.access_record %orders "o_orderdate" : record<"o_orderkey": dictionary<i32, i32>, "o_custkey": dictionary<i32, i32>, "o_orderstatus": dictionary<i32, memref<1xi8>>, "o_totalprice": dictionary<i32, f64>, "o_orderdate": dictionary<i32, i32>, "o_orderpriority": dictionary<i32, memref<15xi8>>, "o_clerk": dictionary<i32, memref<15xi8>>, "o_shippriority": dictionary<i32, i32>, "o_comment": dictionary<i32, memref<79xi8>>, "size": i32> -> dictionary<i32, i32>
  %get_73 = sdql.lookup_dictionary %fieldnode_74 [%i : i32] : dictionary<i32, i32> -> i32
  %fieldnode_76 = sdql.access_record %orders "o_orderpriority" : record<"o_orderkey": dictionary<i32, i32>, "o_custkey": dictionary<i32, i32>, "o_orderstatus": dictionary<i32, memref<1xi8>>, "o_totalprice": dictionary<i32, f64>, "o_orderdate": dictionary<i32, i32>, "o_orderpriority": dictionary<i32, memref<15xi8>>, "o_clerk": dictionary<i32, memref<15xi8>>, "o_shippriority": dictionary<i32, i32>, "o_comment": dictionary<i32, memref<79xi8>>, "size": i32> -> dictionary<i32, memref<15xi8>>
  %get_75 = sdql.lookup_dictionary %fieldnode_76 [%i : i32] : dictionary<i32, memref<15xi8>> -> memref<15xi8>
  %fieldnode_78 = sdql.access_record %orders "o_clerk" : record<"o_orderkey": dictionary<i32, i32>, "o_custkey": dictionary<i32, i32>, "o_orderstatus": dictionary<i32, memref<1xi8>>, "o_totalprice": dictionary<i32, f64>, "o_orderdate": dictionary<i32, i32>, "o_orderpriority": dictionary<i32, memref<15xi8>>, "o_clerk": dictionary<i32, memref<15xi8>>, "o_shippriority": dictionary<i32, i32>, "o_comment": dictionary<i32, memref<79xi8>>, "size": i32> -> dictionary<i32, memref<15xi8>>
  %get_77 = sdql.lookup_dictionary %fieldnode_78 [%i : i32] : dictionary<i32, memref<15xi8>> -> memref<15xi8>
  %fieldnode_80 = sdql.access_record %orders "o_shippriority" : record<"o_orderkey": dictionary<i32, i32>, "o_custkey": dictionary<i32, i32>, "o_orderstatus": dictionary<i32, memref<1xi8>>, "o_totalprice": dictionary<i32, f64>, "o_orderdate": dictionary<i32, i32>, "o_orderpriority": dictionary<i32, memref<15xi8>>, "o_clerk": dictionary<i32, memref<15xi8>>, "o_shippriority": dictionary<i32, i32>, "o_comment": dictionary<i32, memref<79xi8>>, "size": i32> -> dictionary<i32, i32>
  %get_79 = sdql.lookup_dictionary %fieldnode_80 [%i : i32] : dictionary<i32, i32> -> i32
  %fieldnode_82 = sdql.access_record %orders "o_comment" : record<"o_orderkey": dictionary<i32, i32>, "o_custkey": dictionary<i32, i32>, "o_orderstatus": dictionary<i32, memref<1xi8>>, "o_totalprice": dictionary<i32, f64>, "o_orderdate": dictionary<i32, i32>, "o_orderpriority": dictionary<i32, memref<15xi8>>, "o_clerk": dictionary<i32, memref<15xi8>>, "o_shippriority": dictionary<i32, i32>, "o_comment": dictionary<i32, memref<79xi8>>, "size": i32> -> dictionary<i32, memref<79xi8>>
  %get_81 = sdql.lookup_dictionary %fieldnode_82 [%i : i32] : dictionary<i32, memref<79xi8>> -> memref<79xi8>
  %o = sdql.create_record {fields = ["o_orderkey", "o_custkey", "o_orderstatus", "o_totalprice", "o_orderdate", "o_orderpriority", "o_clerk", "o_shippriority", "o_comment"]} %get_65, %get_67, %get_69, %get_71, %get_73, %get_75, %get_77, %get_79, %get_81 : i32, i32, memref<1xi8>, f64, i32, memref<15xi8>, memref<15xi8>, i32, memref<79xi8> -> record<"o_orderkey": i32, "o_custkey": i32, "o_orderstatus": memref<1xi8>, "o_totalprice": f64, "o_orderdate": i32, "o_orderpriority": memref<15xi8>, "o_clerk": memref<15xi8>, "o_shippriority": i32, "o_comment": memref<79xi8>>
  %fieldnode_86 = sdql.access_record %o "o_custkey" : record<"o_orderkey": i32, "o_custkey": i32, "o_orderstatus": memref<1xi8>, "o_totalprice": f64, "o_orderdate": i32, "o_orderpriority": memref<15xi8>, "o_clerk": memref<15xi8>, "o_shippriority": i32, "o_comment": memref<79xi8>> -> i32
  %get_85 = sdql.lookup_dictionary %custkey_to_name [%fieldnode_86 : i32] : dictionary<i32, memref<25xi8>> -> memref<25xi8>
  %dict_87 = sdql.empty_dictionary : memref<25xi8>
  %cmp_88 = sdql.cmp %get_85, %dict_87 : memref<25xi8>, memref<25xi8> -> i1
  %zero_83 = "arith.constant"() <{value = 0 : i1}> : () -> i1
  %neg_84 = "arith.cmpi"(%cmp_88, %zero_83) <{predicate = 0}> : (i1, i1) -> i1
  %if_89 = "scf.if"(%neg_84) ({
    %fieldnode_92 = sdql.access_record %o "o_orderkey" : record<"o_orderkey": i32, "o_custkey": i32, "o_orderstatus": memref<1xi8>, "o_totalprice": f64, "o_orderdate": i32, "o_orderpriority": memref<15xi8>, "o_clerk": memref<15xi8>, "o_shippriority": i32, "o_comment": memref<79xi8>> -> i32
    %uniq_91 = sdql.unique %fieldnode_92 : i32 -> i32
    %fieldnode_94 = sdql.access_record %o "o_custkey" : record<"o_orderkey": i32, "o_custkey": i32, "o_orderstatus": memref<1xi8>, "o_totalprice": f64, "o_orderdate": i32, "o_orderpriority": memref<15xi8>, "o_clerk": memref<15xi8>, "o_shippriority": i32, "o_comment": memref<79xi8>> -> i32
    %get_93 = sdql.lookup_dictionary %custkey_to_name [%fieldnode_94 : i32] : dictionary<i32, memref<25xi8>> -> memref<25xi8>
    %dict_90 = sdql.create_dictionary %uniq_91, %get_93 : i32, memref<25xi8> -> dictionary<i32, memref<25xi8>>
    "scf.yield"(%dict_90) : (dictionary<i32, memref<25xi8>>) -> ()
  }, {
    %dict_95 = sdql.empty_dictionary : dictionary<i32, memref<25xi8>>
    "scf.yield"(%dict_95) : (dictionary<i32, memref<25xi8>>) -> ()
  }) : (i1) -> dictionary<i32, memref<25xi8>>
  sdql.yield %if_89 : dictionary<i32, memref<25xi8>>
}
%fieldnode_96 = sdql.access_record %supplier "size" : record<"s_suppkey": dictionary<i32, i32>, "s_name": dictionary<i32, memref<25xi8>>, "s_address": dictionary<i32, memref<40xi8>>, "s_nationkey": dictionary<i32, i32>, "s_phone": dictionary<i32, memref<15xi8>>, "s_acctbal": dictionary<i32, f64>, "s_comment": dictionary<i32, memref<101xi8>>, "size": i32> -> i32
%rangenode_97 = "func.call"(%fieldnode_96) <{callee = @range_builtin}> : (i32) -> dictionary<i32, i32>
%suppkey_to_name = sdql.sum %rangenode_97 : dictionary<i32, i32> -> dictionary<i32, memref<25xi8>> {
^bb_98(%i: i32, %_: i32):
  %fieldnode_100 = sdql.access_record %supplier "s_suppkey" : record<"s_suppkey": dictionary<i32, i32>, "s_name": dictionary<i32, memref<25xi8>>, "s_address": dictionary<i32, memref<40xi8>>, "s_nationkey": dictionary<i32, i32>, "s_phone": dictionary<i32, memref<15xi8>>, "s_acctbal": dictionary<i32, f64>, "s_comment": dictionary<i32, memref<101xi8>>, "size": i32> -> dictionary<i32, i32>
  %get_99 = sdql.lookup_dictionary %fieldnode_100 [%i : i32] : dictionary<i32, i32> -> i32
  %fieldnode_102 = sdql.access_record %supplier "s_name" : record<"s_suppkey": dictionary<i32, i32>, "s_name": dictionary<i32, memref<25xi8>>, "s_address": dictionary<i32, memref<40xi8>>, "s_nationkey": dictionary<i32, i32>, "s_phone": dictionary<i32, memref<15xi8>>, "s_acctbal": dictionary<i32, f64>, "s_comment": dictionary<i32, memref<101xi8>>, "size": i32> -> dictionary<i32, memref<25xi8>>
  %get_101 = sdql.lookup_dictionary %fieldnode_102 [%i : i32] : dictionary<i32, memref<25xi8>> -> memref<25xi8>
  %fieldnode_104 = sdql.access_record %supplier "s_address" : record<"s_suppkey": dictionary<i32, i32>, "s_name": dictionary<i32, memref<25xi8>>, "s_address": dictionary<i32, memref<40xi8>>, "s_nationkey": dictionary<i32, i32>, "s_phone": dictionary<i32, memref<15xi8>>, "s_acctbal": dictionary<i32, f64>, "s_comment": dictionary<i32, memref<101xi8>>, "size": i32> -> dictionary<i32, memref<40xi8>>
  %get_103 = sdql.lookup_dictionary %fieldnode_104 [%i : i32] : dictionary<i32, memref<40xi8>> -> memref<40xi8>
  %fieldnode_106 = sdql.access_record %supplier "s_nationkey" : record<"s_suppkey": dictionary<i32, i32>, "s_name": dictionary<i32, memref<25xi8>>, "s_address": dictionary<i32, memref<40xi8>>, "s_nationkey": dictionary<i32, i32>, "s_phone": dictionary<i32, memref<15xi8>>, "s_acctbal": dictionary<i32, f64>, "s_comment": dictionary<i32, memref<101xi8>>, "size": i32> -> dictionary<i32, i32>
  %get_105 = sdql.lookup_dictionary %fieldnode_106 [%i : i32] : dictionary<i32, i32> -> i32
  %fieldnode_108 = sdql.access_record %supplier "s_phone" : record<"s_suppkey": dictionary<i32, i32>, "s_name": dictionary<i32, memref<25xi8>>, "s_address": dictionary<i32, memref<40xi8>>, "s_nationkey": dictionary<i32, i32>, "s_phone": dictionary<i32, memref<15xi8>>, "s_acctbal": dictionary<i32, f64>, "s_comment": dictionary<i32, memref<101xi8>>, "size": i32> -> dictionary<i32, memref<15xi8>>
  %get_107 = sdql.lookup_dictionary %fieldnode_108 [%i : i32] : dictionary<i32, memref<15xi8>> -> memref<15xi8>
  %fieldnode_110 = sdql.access_record %supplier "s_acctbal" : record<"s_suppkey": dictionary<i32, i32>, "s_name": dictionary<i32, memref<25xi8>>, "s_address": dictionary<i32, memref<40xi8>>, "s_nationkey": dictionary<i32, i32>, "s_phone": dictionary<i32, memref<15xi8>>, "s_acctbal": dictionary<i32, f64>, "s_comment": dictionary<i32, memref<101xi8>>, "size": i32> -> dictionary<i32, f64>
  %get_109 = sdql.lookup_dictionary %fieldnode_110 [%i : i32] : dictionary<i32, f64> -> f64
  %fieldnode_112 = sdql.access_record %supplier "s_comment" : record<"s_suppkey": dictionary<i32, i32>, "s_name": dictionary<i32, memref<25xi8>>, "s_address": dictionary<i32, memref<40xi8>>, "s_nationkey": dictionary<i32, i32>, "s_phone": dictionary<i32, memref<15xi8>>, "s_acctbal": dictionary<i32, f64>, "s_comment": dictionary<i32, memref<101xi8>>, "size": i32> -> dictionary<i32, memref<101xi8>>
  %get_111 = sdql.lookup_dictionary %fieldnode_112 [%i : i32] : dictionary<i32, memref<101xi8>> -> memref<101xi8>
  %s = sdql.create_record {fields = ["s_suppkey", "s_name", "s_address", "s_nationkey", "s_phone", "s_acctbal", "s_comment"]} %get_99, %get_101, %get_103, %get_105, %get_107, %get_109, %get_111 : i32, memref<25xi8>, memref<40xi8>, i32, memref<15xi8>, f64, memref<101xi8> -> record<"s_suppkey": i32, "s_name": memref<25xi8>, "s_address": memref<40xi8>, "s_nationkey": i32, "s_phone": memref<15xi8>, "s_acctbal": f64, "s_comment": memref<101xi8>>
  %fieldnode_116 = sdql.access_record %s "s_nationkey" : record<"s_suppkey": i32, "s_name": memref<25xi8>, "s_address": memref<40xi8>, "s_nationkey": i32, "s_phone": memref<15xi8>, "s_acctbal": f64, "s_comment": memref<101xi8>> -> i32
  %get_115 = sdql.lookup_dictionary %nationkey_to_name [%fieldnode_116 : i32] : dictionary<i32, record<"_" : memref<25xi8>>> -> record<"_" : memref<25xi8>>
  %dict_117 = sdql.empty_dictionary : record<"_" : memref<25xi8>>
  %cmp_118 = sdql.cmp %get_115, %dict_117 : record<"_" : memref<25xi8>>, record<"_" : memref<25xi8>> -> i1
  %zero_113 = "arith.constant"() <{value = 0 : i1}> : () -> i1
  %neg_114 = "arith.cmpi"(%cmp_118, %zero_113) <{predicate = 0}> : (i1, i1) -> i1
  %if_119 = "scf.if"(%neg_114) ({
    %fieldnode_122 = sdql.access_record %s "s_suppkey" : record<"s_suppkey": i32, "s_name": memref<25xi8>, "s_address": memref<40xi8>, "s_nationkey": i32, "s_phone": memref<15xi8>, "s_acctbal": f64, "s_comment": memref<101xi8>> -> i32
    %uniq_121 = sdql.unique %fieldnode_122 : i32 -> i32
    %fieldnode_125 = sdql.access_record %s "s_nationkey" : record<"s_suppkey": i32, "s_name": memref<25xi8>, "s_address": memref<40xi8>, "s_nationkey": i32, "s_phone": memref<15xi8>, "s_acctbal": f64, "s_comment": memref<101xi8>> -> i32
    %get_124 = sdql.lookup_dictionary %nationkey_to_name [%fieldnode_125 : i32] : dictionary<i32, record<"_" : memref<25xi8>>> -> record<"_" : memref<25xi8>>
    %consti_126 = "arith.constant"() <{value = 0 : i32}> : () -> i32
    %get_123 = sdql.access_record %get_124 "%consti_126" : record<"_" : memref<25xi8>> -> memref<25xi8>
    %dict_120 = sdql.create_dictionary %uniq_121, %get_123 : i32, memref<25xi8> -> dictionary<i32, memref<25xi8>>
    "scf.yield"(%dict_120) : (dictionary<i32, memref<25xi8>>) -> ()
  }, {
    %dict_127 = sdql.empty_dictionary : dictionary<i32, memref<25xi8>>
    "scf.yield"(%dict_127) : (dictionary<i32, memref<25xi8>>) -> ()
  }) : (i1) -> dictionary<i32, memref<25xi8>>
  sdql.yield %if_119 : dictionary<i32, memref<25xi8>>
}
%fieldnode_128 = sdql.access_record %lineitem "size" : record<"l_orderkey": dictionary<i32, i32>, "l_partkey": dictionary<i32, i32>, "l_suppkey": dictionary<i32, i32>, "l_linenumber": dictionary<i32, i32>, "l_quantity": dictionary<i32, f64>, "l_extendedprice": dictionary<i32, f64>, "l_discount": dictionary<i32, f64>, "l_tax": dictionary<i32, f64>, "l_returnflag": dictionary<i32, memref<1xi8>>, "l_linestatus": dictionary<i32, memref<1xi8>>, "l_shipdate": dictionary<i32, i32>, "l_commitdate": dictionary<i32, i32>, "l_receiptdate": dictionary<i32, i32>, "l_shipinstruct": dictionary<i32, memref<25xi8>>, "l_shipmode": dictionary<i32, memref<10xi8>>, "l_comment": dictionary<i32, memref<44xi8>>, "size": i32> -> i32
%rangenode_129 = "func.call"(%fieldnode_128) <{callee = @range_builtin}> : (i32) -> dictionary<i32, i32>
%l_h = sdql.sum %rangenode_129 : dictionary<i32, i32> -> dictionary<record<"supp_nation" : memref<25xi8>, "cust_nation" : memref<25xi8>, "l_year" : i32>, record<"revenue" : f64>> {
^bb_130(%i: i32, %_: i32):
  %fieldnode_132 = sdql.access_record %lineitem "l_orderkey" : record<"l_orderkey": dictionary<i32, i32>, "l_partkey": dictionary<i32, i32>, "l_suppkey": dictionary<i32, i32>, "l_linenumber": dictionary<i32, i32>, "l_quantity": dictionary<i32, f64>, "l_extendedprice": dictionary<i32, f64>, "l_discount": dictionary<i32, f64>, "l_tax": dictionary<i32, f64>, "l_returnflag": dictionary<i32, memref<1xi8>>, "l_linestatus": dictionary<i32, memref<1xi8>>, "l_shipdate": dictionary<i32, i32>, "l_commitdate": dictionary<i32, i32>, "l_receiptdate": dictionary<i32, i32>, "l_shipinstruct": dictionary<i32, memref<25xi8>>, "l_shipmode": dictionary<i32, memref<10xi8>>, "l_comment": dictionary<i32, memref<44xi8>>, "size": i32> -> dictionary<i32, i32>
  %get_131 = sdql.lookup_dictionary %fieldnode_132 [%i : i32] : dictionary<i32, i32> -> i32
  %fieldnode_134 = sdql.access_record %lineitem "l_partkey" : record<"l_orderkey": dictionary<i32, i32>, "l_partkey": dictionary<i32, i32>, "l_suppkey": dictionary<i32, i32>, "l_linenumber": dictionary<i32, i32>, "l_quantity": dictionary<i32, f64>, "l_extendedprice": dictionary<i32, f64>, "l_discount": dictionary<i32, f64>, "l_tax": dictionary<i32, f64>, "l_returnflag": dictionary<i32, memref<1xi8>>, "l_linestatus": dictionary<i32, memref<1xi8>>, "l_shipdate": dictionary<i32, i32>, "l_commitdate": dictionary<i32, i32>, "l_receiptdate": dictionary<i32, i32>, "l_shipinstruct": dictionary<i32, memref<25xi8>>, "l_shipmode": dictionary<i32, memref<10xi8>>, "l_comment": dictionary<i32, memref<44xi8>>, "size": i32> -> dictionary<i32, i32>
  %get_133 = sdql.lookup_dictionary %fieldnode_134 [%i : i32] : dictionary<i32, i32> -> i32
  %fieldnode_136 = sdql.access_record %lineitem "l_suppkey" : record<"l_orderkey": dictionary<i32, i32>, "l_partkey": dictionary<i32, i32>, "l_suppkey": dictionary<i32, i32>, "l_linenumber": dictionary<i32, i32>, "l_quantity": dictionary<i32, f64>, "l_extendedprice": dictionary<i32, f64>, "l_discount": dictionary<i32, f64>, "l_tax": dictionary<i32, f64>, "l_returnflag": dictionary<i32, memref<1xi8>>, "l_linestatus": dictionary<i32, memref<1xi8>>, "l_shipdate": dictionary<i32, i32>, "l_commitdate": dictionary<i32, i32>, "l_receiptdate": dictionary<i32, i32>, "l_shipinstruct": dictionary<i32, memref<25xi8>>, "l_shipmode": dictionary<i32, memref<10xi8>>, "l_comment": dictionary<i32, memref<44xi8>>, "size": i32> -> dictionary<i32, i32>
  %get_135 = sdql.lookup_dictionary %fieldnode_136 [%i : i32] : dictionary<i32, i32> -> i32
  %fieldnode_138 = sdql.access_record %lineitem "l_linenumber" : record<"l_orderkey": dictionary<i32, i32>, "l_partkey": dictionary<i32, i32>, "l_suppkey": dictionary<i32, i32>, "l_linenumber": dictionary<i32, i32>, "l_quantity": dictionary<i32, f64>, "l_extendedprice": dictionary<i32, f64>, "l_discount": dictionary<i32, f64>, "l_tax": dictionary<i32, f64>, "l_returnflag": dictionary<i32, memref<1xi8>>, "l_linestatus": dictionary<i32, memref<1xi8>>, "l_shipdate": dictionary<i32, i32>, "l_commitdate": dictionary<i32, i32>, "l_receiptdate": dictionary<i32, i32>, "l_shipinstruct": dictionary<i32, memref<25xi8>>, "l_shipmode": dictionary<i32, memref<10xi8>>, "l_comment": dictionary<i32, memref<44xi8>>, "size": i32> -> dictionary<i32, i32>
  %get_137 = sdql.lookup_dictionary %fieldnode_138 [%i : i32] : dictionary<i32, i32> -> i32
  %fieldnode_140 = sdql.access_record %lineitem "l_quantity" : record<"l_orderkey": dictionary<i32, i32>, "l_partkey": dictionary<i32, i32>, "l_suppkey": dictionary<i32, i32>, "l_linenumber": dictionary<i32, i32>, "l_quantity": dictionary<i32, f64>, "l_extendedprice": dictionary<i32, f64>, "l_discount": dictionary<i32, f64>, "l_tax": dictionary<i32, f64>, "l_returnflag": dictionary<i32, memref<1xi8>>, "l_linestatus": dictionary<i32, memref<1xi8>>, "l_shipdate": dictionary<i32, i32>, "l_commitdate": dictionary<i32, i32>, "l_receiptdate": dictionary<i32, i32>, "l_shipinstruct": dictionary<i32, memref<25xi8>>, "l_shipmode": dictionary<i32, memref<10xi8>>, "l_comment": dictionary<i32, memref<44xi8>>, "size": i32> -> dictionary<i32, f64>
  %get_139 = sdql.lookup_dictionary %fieldnode_140 [%i : i32] : dictionary<i32, f64> -> f64
  %fieldnode_142 = sdql.access_record %lineitem "l_extendedprice" : record<"l_orderkey": dictionary<i32, i32>, "l_partkey": dictionary<i32, i32>, "l_suppkey": dictionary<i32, i32>, "l_linenumber": dictionary<i32, i32>, "l_quantity": dictionary<i32, f64>, "l_extendedprice": dictionary<i32, f64>, "l_discount": dictionary<i32, f64>, "l_tax": dictionary<i32, f64>, "l_returnflag": dictionary<i32, memref<1xi8>>, "l_linestatus": dictionary<i32, memref<1xi8>>, "l_shipdate": dictionary<i32, i32>, "l_commitdate": dictionary<i32, i32>, "l_receiptdate": dictionary<i32, i32>, "l_shipinstruct": dictionary<i32, memref<25xi8>>, "l_shipmode": dictionary<i32, memref<10xi8>>, "l_comment": dictionary<i32, memref<44xi8>>, "size": i32> -> dictionary<i32, f64>
  %get_141 = sdql.lookup_dictionary %fieldnode_142 [%i : i32] : dictionary<i32, f64> -> f64
  %fieldnode_144 = sdql.access_record %lineitem "l_discount" : record<"l_orderkey": dictionary<i32, i32>, "l_partkey": dictionary<i32, i32>, "l_suppkey": dictionary<i32, i32>, "l_linenumber": dictionary<i32, i32>, "l_quantity": dictionary<i32, f64>, "l_extendedprice": dictionary<i32, f64>, "l_discount": dictionary<i32, f64>, "l_tax": dictionary<i32, f64>, "l_returnflag": dictionary<i32, memref<1xi8>>, "l_linestatus": dictionary<i32, memref<1xi8>>, "l_shipdate": dictionary<i32, i32>, "l_commitdate": dictionary<i32, i32>, "l_receiptdate": dictionary<i32, i32>, "l_shipinstruct": dictionary<i32, memref<25xi8>>, "l_shipmode": dictionary<i32, memref<10xi8>>, "l_comment": dictionary<i32, memref<44xi8>>, "size": i32> -> dictionary<i32, f64>
  %get_143 = sdql.lookup_dictionary %fieldnode_144 [%i : i32] : dictionary<i32, f64> -> f64
  %fieldnode_146 = sdql.access_record %lineitem "l_tax" : record<"l_orderkey": dictionary<i32, i32>, "l_partkey": dictionary<i32, i32>, "l_suppkey": dictionary<i32, i32>, "l_linenumber": dictionary<i32, i32>, "l_quantity": dictionary<i32, f64>, "l_extendedprice": dictionary<i32, f64>, "l_discount": dictionary<i32, f64>, "l_tax": dictionary<i32, f64>, "l_returnflag": dictionary<i32, memref<1xi8>>, "l_linestatus": dictionary<i32, memref<1xi8>>, "l_shipdate": dictionary<i32, i32>, "l_commitdate": dictionary<i32, i32>, "l_receiptdate": dictionary<i32, i32>, "l_shipinstruct": dictionary<i32, memref<25xi8>>, "l_shipmode": dictionary<i32, memref<10xi8>>, "l_comment": dictionary<i32, memref<44xi8>>, "size": i32> -> dictionary<i32, f64>
  %get_145 = sdql.lookup_dictionary %fieldnode_146 [%i : i32] : dictionary<i32, f64> -> f64
  %fieldnode_148 = sdql.access_record %lineitem "l_returnflag" : record<"l_orderkey": dictionary<i32, i32>, "l_partkey": dictionary<i32, i32>, "l_suppkey": dictionary<i32, i32>, "l_linenumber": dictionary<i32, i32>, "l_quantity": dictionary<i32, f64>, "l_extendedprice": dictionary<i32, f64>, "l_discount": dictionary<i32, f64>, "l_tax": dictionary<i32, f64>, "l_returnflag": dictionary<i32, memref<1xi8>>, "l_linestatus": dictionary<i32, memref<1xi8>>, "l_shipdate": dictionary<i32, i32>, "l_commitdate": dictionary<i32, i32>, "l_receiptdate": dictionary<i32, i32>, "l_shipinstruct": dictionary<i32, memref<25xi8>>, "l_shipmode": dictionary<i32, memref<10xi8>>, "l_comment": dictionary<i32, memref<44xi8>>, "size": i32> -> dictionary<i32, memref<1xi8>>
  %get_147 = sdql.lookup_dictionary %fieldnode_148 [%i : i32] : dictionary<i32, memref<1xi8>> -> memref<1xi8>
  %fieldnode_150 = sdql.access_record %lineitem "l_linestatus" : record<"l_orderkey": dictionary<i32, i32>, "l_partkey": dictionary<i32, i32>, "l_suppkey": dictionary<i32, i32>, "l_linenumber": dictionary<i32, i32>, "l_quantity": dictionary<i32, f64>, "l_extendedprice": dictionary<i32, f64>, "l_discount": dictionary<i32, f64>, "l_tax": dictionary<i32, f64>, "l_returnflag": dictionary<i32, memref<1xi8>>, "l_linestatus": dictionary<i32, memref<1xi8>>, "l_shipdate": dictionary<i32, i32>, "l_commitdate": dictionary<i32, i32>, "l_receiptdate": dictionary<i32, i32>, "l_shipinstruct": dictionary<i32, memref<25xi8>>, "l_shipmode": dictionary<i32, memref<10xi8>>, "l_comment": dictionary<i32, memref<44xi8>>, "size": i32> -> dictionary<i32, memref<1xi8>>
  %get_149 = sdql.lookup_dictionary %fieldnode_150 [%i : i32] : dictionary<i32, memref<1xi8>> -> memref<1xi8>
  %fieldnode_152 = sdql.access_record %lineitem "l_shipdate" : record<"l_orderkey": dictionary<i32, i32>, "l_partkey": dictionary<i32, i32>, "l_suppkey": dictionary<i32, i32>, "l_linenumber": dictionary<i32, i32>, "l_quantity": dictionary<i32, f64>, "l_extendedprice": dictionary<i32, f64>, "l_discount": dictionary<i32, f64>, "l_tax": dictionary<i32, f64>, "l_returnflag": dictionary<i32, memref<1xi8>>, "l_linestatus": dictionary<i32, memref<1xi8>>, "l_shipdate": dictionary<i32, i32>, "l_commitdate": dictionary<i32, i32>, "l_receiptdate": dictionary<i32, i32>, "l_shipinstruct": dictionary<i32, memref<25xi8>>, "l_shipmode": dictionary<i32, memref<10xi8>>, "l_comment": dictionary<i32, memref<44xi8>>, "size": i32> -> dictionary<i32, i32>
  %get_151 = sdql.lookup_dictionary %fieldnode_152 [%i : i32] : dictionary<i32, i32> -> i32
  %fieldnode_154 = sdql.access_record %lineitem "l_commitdate" : record<"l_orderkey": dictionary<i32, i32>, "l_partkey": dictionary<i32, i32>, "l_suppkey": dictionary<i32, i32>, "l_linenumber": dictionary<i32, i32>, "l_quantity": dictionary<i32, f64>, "l_extendedprice": dictionary<i32, f64>, "l_discount": dictionary<i32, f64>, "l_tax": dictionary<i32, f64>, "l_returnflag": dictionary<i32, memref<1xi8>>, "l_linestatus": dictionary<i32, memref<1xi8>>, "l_shipdate": dictionary<i32, i32>, "l_commitdate": dictionary<i32, i32>, "l_receiptdate": dictionary<i32, i32>, "l_shipinstruct": dictionary<i32, memref<25xi8>>, "l_shipmode": dictionary<i32, memref<10xi8>>, "l_comment": dictionary<i32, memref<44xi8>>, "size": i32> -> dictionary<i32, i32>
  %get_153 = sdql.lookup_dictionary %fieldnode_154 [%i : i32] : dictionary<i32, i32> -> i32
  %fieldnode_156 = sdql.access_record %lineitem "l_receiptdate" : record<"l_orderkey": dictionary<i32, i32>, "l_partkey": dictionary<i32, i32>, "l_suppkey": dictionary<i32, i32>, "l_linenumber": dictionary<i32, i32>, "l_quantity": dictionary<i32, f64>, "l_extendedprice": dictionary<i32, f64>, "l_discount": dictionary<i32, f64>, "l_tax": dictionary<i32, f64>, "l_returnflag": dictionary<i32, memref<1xi8>>, "l_linestatus": dictionary<i32, memref<1xi8>>, "l_shipdate": dictionary<i32, i32>, "l_commitdate": dictionary<i32, i32>, "l_receiptdate": dictionary<i32, i32>, "l_shipinstruct": dictionary<i32, memref<25xi8>>, "l_shipmode": dictionary<i32, memref<10xi8>>, "l_comment": dictionary<i32, memref<44xi8>>, "size": i32> -> dictionary<i32, i32>
  %get_155 = sdql.lookup_dictionary %fieldnode_156 [%i : i32] : dictionary<i32, i32> -> i32
  %fieldnode_158 = sdql.access_record %lineitem "l_shipinstruct" : record<"l_orderkey": dictionary<i32, i32>, "l_partkey": dictionary<i32, i32>, "l_suppkey": dictionary<i32, i32>, "l_linenumber": dictionary<i32, i32>, "l_quantity": dictionary<i32, f64>, "l_extendedprice": dictionary<i32, f64>, "l_discount": dictionary<i32, f64>, "l_tax": dictionary<i32, f64>, "l_returnflag": dictionary<i32, memref<1xi8>>, "l_linestatus": dictionary<i32, memref<1xi8>>, "l_shipdate": dictionary<i32, i32>, "l_commitdate": dictionary<i32, i32>, "l_receiptdate": dictionary<i32, i32>, "l_shipinstruct": dictionary<i32, memref<25xi8>>, "l_shipmode": dictionary<i32, memref<10xi8>>, "l_comment": dictionary<i32, memref<44xi8>>, "size": i32> -> dictionary<i32, memref<25xi8>>
  %get_157 = sdql.lookup_dictionary %fieldnode_158 [%i : i32] : dictionary<i32, memref<25xi8>> -> memref<25xi8>
  %fieldnode_160 = sdql.access_record %lineitem "l_shipmode" : record<"l_orderkey": dictionary<i32, i32>, "l_partkey": dictionary<i32, i32>, "l_suppkey": dictionary<i32, i32>, "l_linenumber": dictionary<i32, i32>, "l_quantity": dictionary<i32, f64>, "l_extendedprice": dictionary<i32, f64>, "l_discount": dictionary<i32, f64>, "l_tax": dictionary<i32, f64>, "l_returnflag": dictionary<i32, memref<1xi8>>, "l_linestatus": dictionary<i32, memref<1xi8>>, "l_shipdate": dictionary<i32, i32>, "l_commitdate": dictionary<i32, i32>, "l_receiptdate": dictionary<i32, i32>, "l_shipinstruct": dictionary<i32, memref<25xi8>>, "l_shipmode": dictionary<i32, memref<10xi8>>, "l_comment": dictionary<i32, memref<44xi8>>, "size": i32> -> dictionary<i32, memref<10xi8>>
  %get_159 = sdql.lookup_dictionary %fieldnode_160 [%i : i32] : dictionary<i32, memref<10xi8>> -> memref<10xi8>
  %fieldnode_162 = sdql.access_record %lineitem "l_comment" : record<"l_orderkey": dictionary<i32, i32>, "l_partkey": dictionary<i32, i32>, "l_suppkey": dictionary<i32, i32>, "l_linenumber": dictionary<i32, i32>, "l_quantity": dictionary<i32, f64>, "l_extendedprice": dictionary<i32, f64>, "l_discount": dictionary<i32, f64>, "l_tax": dictionary<i32, f64>, "l_returnflag": dictionary<i32, memref<1xi8>>, "l_linestatus": dictionary<i32, memref<1xi8>>, "l_shipdate": dictionary<i32, i32>, "l_commitdate": dictionary<i32, i32>, "l_receiptdate": dictionary<i32, i32>, "l_shipinstruct": dictionary<i32, memref<25xi8>>, "l_shipmode": dictionary<i32, memref<10xi8>>, "l_comment": dictionary<i32, memref<44xi8>>, "size": i32> -> dictionary<i32, memref<44xi8>>
  %get_161 = sdql.lookup_dictionary %fieldnode_162 [%i : i32] : dictionary<i32, memref<44xi8>> -> memref<44xi8>
  %l = sdql.create_record {fields = ["l_orderkey", "l_partkey", "l_suppkey", "l_linenumber", "l_quantity", "l_extendedprice", "l_discount", "l_tax", "l_returnflag", "l_linestatus", "l_shipdate", "l_commitdate", "l_receiptdate", "l_shipinstruct", "l_shipmode", "l_comment"]} %get_131, %get_133, %get_135, %get_137, %get_139, %get_141, %get_143, %get_145, %get_147, %get_149, %get_151, %get_153, %get_155, %get_157, %get_159, %get_161 : i32, i32, i32, i32, f64, f64, f64, f64, memref<1xi8>, memref<1xi8>, i32, i32, i32, memref<25xi8>, memref<10xi8>, memref<44xi8> -> record<"l_orderkey": i32, "l_partkey": i32, "l_suppkey": i32, "l_linenumber": i32, "l_quantity": f64, "l_extendedprice": f64, "l_discount": f64, "l_tax": f64, "l_returnflag": memref<1xi8>, "l_linestatus": memref<1xi8>, "l_shipdate": i32, "l_commitdate": i32, "l_receiptdate": i32, "l_shipinstruct": memref<25xi8>, "l_shipmode": memref<10xi8>, "l_comment": memref<44xi8>>
  %constda_163 = "arith.constant"() <{value = 19950101 : i32}> : () -> i32
  %fieldnode_164 = sdql.access_record %l "l_shipdate" : record<"l_orderkey": i32, "l_partkey": i32, "l_suppkey": i32, "l_linenumber": i32, "l_quantity": f64, "l_extendedprice": f64, "l_discount": f64, "l_tax": f64, "l_returnflag": memref<1xi8>, "l_linestatus": memref<1xi8>, "l_shipdate": i32, "l_commitdate": i32, "l_receiptdate": i32, "l_shipinstruct": memref<25xi8>, "l_shipmode": memref<10xi8>, "l_comment": memref<44xi8>> -> i32
  %cmpi_165 = "arith.cmpi"(%constda_163, %fieldnode_164) <{predicate = 3}> : (i32, i32) -> i1
  %if_166 = "scf.if"(%cmpi_165) ({
    %fieldnode_167 = sdql.access_record %l "l_shipdate" : record<"l_orderkey": i32, "l_partkey": i32, "l_suppkey": i32, "l_linenumber": i32, "l_quantity": f64, "l_extendedprice": f64, "l_discount": f64, "l_tax": f64, "l_returnflag": memref<1xi8>, "l_linestatus": memref<1xi8>, "l_shipdate": i32, "l_commitdate": i32, "l_receiptdate": i32, "l_shipinstruct": memref<25xi8>, "l_shipmode": memref<10xi8>, "l_comment": memref<44xi8>> -> i32
    %constda_168 = "arith.constant"() <{value = 19961231 : i32}> : () -> i32
    %cmpi_169 = "arith.cmpi"(%fieldnode_167, %constda_168) <{predicate = 3}> : (i32, i32) -> i1
    "scf.yield"(%cmpi_169) : (i1) -> ()
  }, {
    %constv_170 = "arith.constant"() <{value = 0 : i1}> : () -> i1
    "scf.yield"(%constv_170) : (i1) -> ()
  }) : (i1) -> i1
  %if_171 = "scf.if"(%if_166) ({
    %fieldnode_175 = sdql.access_record %l "l_orderkey" : record<"l_orderkey": i32, "l_partkey": i32, "l_suppkey": i32, "l_linenumber": i32, "l_quantity": f64, "l_extendedprice": f64, "l_discount": f64, "l_tax": f64, "l_returnflag": memref<1xi8>, "l_linestatus": memref<1xi8>, "l_shipdate": i32, "l_commitdate": i32, "l_receiptdate": i32, "l_shipinstruct": memref<25xi8>, "l_shipmode": memref<10xi8>, "l_comment": memref<44xi8>> -> i32
    %get_174 = sdql.lookup_dictionary %orderkey_to_name [%fieldnode_175 : i32] : dictionary<i32, memref<25xi8>> -> memref<25xi8>
    %dict_176 = sdql.empty_dictionary : memref<25xi8>
    %cmp_177 = sdql.cmp %get_174, %dict_176 : memref<25xi8>, memref<25xi8> -> i1
    %zero_172 = "arith.constant"() <{value = 0 : i1}> : () -> i1
    %neg_173 = "arith.cmpi"(%cmp_177, %zero_172) <{predicate = 0}> : (i1, i1) -> i1
    "scf.yield"(%neg_173) : (i1) -> ()
  }, {
    %constv_178 = "arith.constant"() <{value = 0 : i1}> : () -> i1
    "scf.yield"(%constv_178) : (i1) -> ()
  }) : (i1) -> i1
  %if_179 = "scf.if"(%if_171) ({
    %fieldnode_183 = sdql.access_record %l "l_suppkey" : record<"l_orderkey": i32, "l_partkey": i32, "l_suppkey": i32, "l_linenumber": i32, "l_quantity": f64, "l_extendedprice": f64, "l_discount": f64, "l_tax": f64, "l_returnflag": memref<1xi8>, "l_linestatus": memref<1xi8>, "l_shipdate": i32, "l_commitdate": i32, "l_receiptdate": i32, "l_shipinstruct": memref<25xi8>, "l_shipmode": memref<10xi8>, "l_comment": memref<44xi8>> -> i32
    %get_182 = sdql.lookup_dictionary %suppkey_to_name [%fieldnode_183 : i32] : dictionary<i32, memref<25xi8>> -> memref<25xi8>
    %dict_184 = sdql.empty_dictionary : memref<25xi8>
    %cmp_185 = sdql.cmp %get_182, %dict_184 : memref<25xi8>, memref<25xi8> -> i1
    %zero_180 = "arith.constant"() <{value = 0 : i1}> : () -> i1
    %neg_181 = "arith.cmpi"(%cmp_185, %zero_180) <{predicate = 0}> : (i1, i1) -> i1
    "scf.yield"(%neg_181) : (i1) -> ()
  }, {
    %constv_186 = "arith.constant"() <{value = 0 : i1}> : () -> i1
    "scf.yield"(%constv_186) : (i1) -> ()
  }) : (i1) -> i1
  %if_187 = "scf.if"(%if_179) ({
    %fieldnode_189 = sdql.access_record %l "l_orderkey" : record<"l_orderkey": i32, "l_partkey": i32, "l_suppkey": i32, "l_linenumber": i32, "l_quantity": f64, "l_extendedprice": f64, "l_discount": f64, "l_tax": f64, "l_returnflag": memref<1xi8>, "l_linestatus": memref<1xi8>, "l_shipdate": i32, "l_commitdate": i32, "l_receiptdate": i32, "l_shipinstruct": memref<25xi8>, "l_shipmode": memref<10xi8>, "l_comment": memref<44xi8>> -> i32
    %get_188 = sdql.lookup_dictionary %orderkey_to_name [%fieldnode_189 : i32] : dictionary<i32, memref<25xi8>> -> memref<25xi8>
    %cmp_190 = sdql.cmp %get_188, %france : memref<25xi8>, memref<6xi8> -> i1
    %if_191 = "scf.if"(%cmp_190) ({
      %fieldnode_193 = sdql.access_record %l "l_suppkey" : record<"l_orderkey": i32, "l_partkey": i32, "l_suppkey": i32, "l_linenumber": i32, "l_quantity": f64, "l_extendedprice": f64, "l_discount": f64, "l_tax": f64, "l_returnflag": memref<1xi8>, "l_linestatus": memref<1xi8>, "l_shipdate": i32, "l_commitdate": i32, "l_receiptdate": i32, "l_shipinstruct": memref<25xi8>, "l_shipmode": memref<10xi8>, "l_comment": memref<44xi8>> -> i32
      %get_192 = sdql.lookup_dictionary %suppkey_to_name [%fieldnode_193 : i32] : dictionary<i32, memref<25xi8>> -> memref<25xi8>
      %cmp_194 = sdql.cmp %get_192, %germany : memref<25xi8>, memref<7xi8> -> i1
      "scf.yield"(%cmp_194) : (i1) -> ()
    }, {
      %constv_195 = "arith.constant"() <{value = 0 : i1}> : () -> i1
      "scf.yield"(%constv_195) : (i1) -> ()
    }) : (i1) -> i1
    %if_196 = "scf.if"(%if_191) ({
      %constv_197 = "arith.constant"() <{value = 1 : i1}> : () -> i1
      "scf.yield"(%constv_197) : (i1) -> ()
    }, {
      %fieldnode_199 = sdql.access_record %l "l_orderkey" : record<"l_orderkey": i32, "l_partkey": i32, "l_suppkey": i32, "l_linenumber": i32, "l_quantity": f64, "l_extendedprice": f64, "l_discount": f64, "l_tax": f64, "l_returnflag": memref<1xi8>, "l_linestatus": memref<1xi8>, "l_shipdate": i32, "l_commitdate": i32, "l_receiptdate": i32, "l_shipinstruct": memref<25xi8>, "l_shipmode": memref<10xi8>, "l_comment": memref<44xi8>> -> i32
      %get_198 = sdql.lookup_dictionary %orderkey_to_name [%fieldnode_199 : i32] : dictionary<i32, memref<25xi8>> -> memref<25xi8>
      %cmp_200 = sdql.cmp %get_198, %germany : memref<25xi8>, memref<7xi8> -> i1
      %if_201 = "scf.if"(%cmp_200) ({
        %fieldnode_203 = sdql.access_record %l "l_suppkey" : record<"l_orderkey": i32, "l_partkey": i32, "l_suppkey": i32, "l_linenumber": i32, "l_quantity": f64, "l_extendedprice": f64, "l_discount": f64, "l_tax": f64, "l_returnflag": memref<1xi8>, "l_linestatus": memref<1xi8>, "l_shipdate": i32, "l_commitdate": i32, "l_receiptdate": i32, "l_shipinstruct": memref<25xi8>, "l_shipmode": memref<10xi8>, "l_comment": memref<44xi8>> -> i32
        %get_202 = sdql.lookup_dictionary %suppkey_to_name [%fieldnode_203 : i32] : dictionary<i32, memref<25xi8>> -> memref<25xi8>
        %cmp_204 = sdql.cmp %get_202, %france : memref<25xi8>, memref<6xi8> -> i1
        "scf.yield"(%cmp_204) : (i1) -> ()
      }, {
        %constv_205 = "arith.constant"() <{value = 0 : i1}> : () -> i1
        "scf.yield"(%constv_205) : (i1) -> ()
      }) : (i1) -> i1
      "scf.yield"(%if_201) : (i1) -> ()
    }) : (i1) -> i1
    "scf.yield"(%if_196) : (i1) -> ()
  }, {
    %constv_206 = "arith.constant"() <{value = 0 : i1}> : () -> i1
    "scf.yield"(%constv_206) : (i1) -> ()
  }) : (i1) -> i1
  %if_207 = "scf.if"(%if_187) ({
    %fieldnode_210 = sdql.access_record %l "l_suppkey" : record<"l_orderkey": i32, "l_partkey": i32, "l_suppkey": i32, "l_linenumber": i32, "l_quantity": f64, "l_extendedprice": f64, "l_discount": f64, "l_tax": f64, "l_returnflag": memref<1xi8>, "l_linestatus": memref<1xi8>, "l_shipdate": i32, "l_commitdate": i32, "l_receiptdate": i32, "l_shipinstruct": memref<25xi8>, "l_shipmode": memref<10xi8>, "l_comment": memref<44xi8>> -> i32
    %get_209 = sdql.lookup_dictionary %suppkey_to_name [%fieldnode_210 : i32] : dictionary<i32, memref<25xi8>> -> memref<25xi8>
    %fieldnode_212 = sdql.access_record %l "l_orderkey" : record<"l_orderkey": i32, "l_partkey": i32, "l_suppkey": i32, "l_linenumber": i32, "l_quantity": f64, "l_extendedprice": f64, "l_discount": f64, "l_tax": f64, "l_returnflag": memref<1xi8>, "l_linestatus": memref<1xi8>, "l_shipdate": i32, "l_commitdate": i32, "l_receiptdate": i32, "l_shipinstruct": memref<25xi8>, "l_shipmode": memref<10xi8>, "l_comment": memref<44xi8>> -> i32
    %get_211 = sdql.lookup_dictionary %orderkey_to_name [%fieldnode_212 : i32] : dictionary<i32, memref<25xi8>> -> memref<25xi8>
    %fieldnode_214 = sdql.access_record %l "l_shipdate" : record<"l_orderkey": i32, "l_partkey": i32, "l_suppkey": i32, "l_linenumber": i32, "l_quantity": f64, "l_extendedprice": f64, "l_discount": f64, "l_tax": f64, "l_returnflag": memref<1xi8>, "l_linestatus": memref<1xi8>, "l_shipdate": i32, "l_commitdate": i32, "l_receiptdate": i32, "l_shipinstruct": memref<25xi8>, "l_shipmode": memref<10xi8>, "l_comment": memref<44xi8>> -> i32
    %consti_215 = "arith.constant"() <{value = 10000 : i32}> : () -> i32
    %external_216 = sdql.external "inv", %consti_215 : i32 -> i32
    %multi_213 = "arith.muli"(%fieldnode_214, %external_216) : (i32, i32) -> i32
    %recnode_217 = sdql.create_record {fields = ["supp_nation", "cust_nation", "l_year"]} %get_209, %get_211, %multi_213 : memref<25xi8>, memref<25xi8>, i32 -> record<"supp_nation": memref<25xi8>, "cust_nation": memref<25xi8>, "l_year": i32>
    %fieldnode_219 = sdql.access_record %l "l_extendedprice" : record<"l_orderkey": i32, "l_partkey": i32, "l_suppkey": i32, "l_linenumber": i32, "l_quantity": f64, "l_extendedprice": f64, "l_discount": f64, "l_tax": f64, "l_returnflag": memref<1xi8>, "l_linestatus": memref<1xi8>, "l_shipdate": i32, "l_commitdate": i32, "l_receiptdate": i32, "l_shipinstruct": memref<25xi8>, "l_shipmode": memref<10xi8>, "l_comment": memref<44xi8>> -> f64
    %constd_221 = "arith.constant"() <{value = 1.0  : f64}> : () -> f64
    %fieldnode_223 = sdql.access_record %l "l_discount" : record<"l_orderkey": i32, "l_partkey": i32, "l_suppkey": i32, "l_linenumber": i32, "l_quantity": f64, "l_extendedprice": f64, "l_discount": f64, "l_tax": f64, "l_returnflag": memref<1xi8>, "l_linestatus": memref<1xi8>, "l_shipdate": i32, "l_commitdate": i32, "l_receiptdate": i32, "l_shipinstruct": memref<25xi8>, "l_shipmode": memref<10xi8>, "l_comment": memref<44xi8>> -> f64
    %zero_224 = "arith.constant"() <{value = 0.0 : f64}> : () -> f64
    %negf_222 = "arith.subf"(%zero_224, %fieldnode_223) <{fastmath = #arith.fastmath<none>}> : (f64, f64) -> f64
    %addf_220 = "arith.addf"(%constd_221, %negf_222) <{fastmath = #arith.fastmath<none>}> : (f64, f64) -> f64
    %multf_218 = "arith.mulf"(%fieldnode_219, %addf_220) <{fastmath = #arith.fastmath<none>}> : (f64, f64) -> f64
    %recnode_225 = sdql.create_record {fields = ["revenue"]} %multf_218 : f64 -> record<"revenue": f64>
    %dict_208 = sdql.create_dictionary %recnode_217, %recnode_225 : record<"supp_nation" : memref<25xi8>, "cust_nation" : memref<25xi8>, "l_year" : i32>, record<"revenue" : f64> -> dictionary<record<"supp_nation" : memref<25xi8>, "cust_nation" : memref<25xi8>, "l_year" : i32>, record<"revenue" : f64>>
    "scf.yield"(%dict_208) : (dictionary<record<"supp_nation" : memref<25xi8>, "cust_nation" : memref<25xi8>, "l_year" : i32>, record<"revenue" : f64>>) -> ()
  }, {
    %dict_226 = sdql.empty_dictionary : dictionary<record<"supp_nation" : memref<25xi8>, "cust_nation" : memref<25xi8>, "l_year" : i32>, record<"revenue" : f64>>
    "scf.yield"(%dict_226) : (dictionary<record<"supp_nation" : memref<25xi8>, "cust_nation" : memref<25xi8>, "l_year" : i32>, record<"revenue" : f64>>) -> ()
  }) : (i1) -> dictionary<record<"supp_nation" : memref<25xi8>, "cust_nation" : memref<25xi8>, "l_year" : i32>, record<"revenue" : f64>>
  sdql.yield %if_207 : dictionary<record<"supp_nation" : memref<25xi8>, "cust_nation" : memref<25xi8>, "l_year" : i32>, record<"revenue" : f64>>
}
%sum_227 = sdql.sum %l_h : dictionary<record<"supp_nation" : memref<25xi8>, "cust_nation" : memref<25xi8>, "l_year" : i32>, record<"revenue" : f64>> -> dictionary<record<"supp_nation" : memref<25xi8>, "cust_nation" : memref<25xi8>, "l_year" : i32, "revenue" : f64>, i32> {
^bb_228(%k: record<"supp_nation" : memref<25xi8>, "cust_nation" : memref<25xi8>, "l_year" : i32>, %v: record<"revenue" : f64>):
  %concat_231 = sdql.concat %k, %v : record<"supp_nation" : memref<25xi8>, "cust_nation" : memref<25xi8>, "l_year" : i32>, record<"revenue" : f64> -> record<"supp_nation" : memref<25xi8>, "cust_nation" : memref<25xi8>, "l_year" : i32, "revenue" : f64>
  %uniq_230 = sdql.unique %concat_231 : record<"supp_nation" : memref<25xi8>, "cust_nation" : memref<25xi8>, "l_year" : i32, "revenue" : f64> -> record<"supp_nation" : memref<25xi8>, "cust_nation" : memref<25xi8>, "l_year" : i32, "revenue" : f64>
  %consti_232 = "arith.constant"() <{value = 1 : i32}> : () -> i32
  %dict_229 = sdql.create_dictionary %uniq_230, %consti_232 : record<"supp_nation" : memref<25xi8>, "cust_nation" : memref<25xi8>, "l_year" : i32, "revenue" : f64>, i32 -> dictionary<record<"supp_nation" : memref<25xi8>, "cust_nation" : memref<25xi8>, "l_year" : i32, "revenue" : f64>, i32>
  sdql.yield %dict_229 : dictionary<record<"supp_nation" : memref<25xi8>, "cust_nation" : memref<25xi8>, "l_year" : i32, "revenue" : f64>, i32>
}
