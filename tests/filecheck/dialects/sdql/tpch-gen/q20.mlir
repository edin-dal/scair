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
%supplier = sdql.load "datasets/tpch/supplier.tbl" : record<"s_suppkey" : dictionary<i32, i32>, "s_name" : dictionary<i32, memref<25xi8>>, "s_address" : dictionary<i32, memref<40xi8>>, "s_nationkey" : dictionary<i32, i32>, "s_phone" : dictionary<i32, memref<15xi8>>, "s_acctbal" : dictionary<i32, f64>, "s_comment" : dictionary<i32, memref<101xi8>>, "size" : i32>
%nation = sdql.load "datasets/tpch/nation.tbl" : record<"n_nationkey" : dictionary<i32, i32>, "n_name" : dictionary<i32, memref<25xi8>>, "n_regionkey" : dictionary<i32, i32>, "n_comment" : dictionary<i32, memref<152xi8>>, "size" : i32>
%part = sdql.load "datasets/tpch/part.tbl" : record<"p_partkey" : dictionary<i32, i32>, "p_name" : dictionary<i32, memref<55xi8>>, "p_mfgr" : dictionary<i32, memref<25xi8>>, "p_brand" : dictionary<i32, memref<10xi8>>, "p_type" : dictionary<i32, memref<25xi8>>, "p_size" : dictionary<i32, i32>, "p_container" : dictionary<i32, memref<10xi8>>, "p_retailprice" : dictionary<i32, f64>, "p_comment" : dictionary<i32, memref<23xi8>>, "size" : i32>
%partsupp = sdql.load "datasets/tpch/partsupp.tbl" : record<"ps_partkey" : dictionary<i32, i32>, "ps_suppkey" : dictionary<i32, i32>, "ps_availqty" : dictionary<i32, f64>, "ps_supplycost" : dictionary<i32, f64>, "ps_comment" : dictionary<i32, memref<199xi8>>, "size" : i32>
%lineitem = sdql.load "datasets/tpch/lineitem.tbl" : record<"l_orderkey" : dictionary<i32, i32>, "l_partkey" : dictionary<i32, i32>, "l_suppkey" : dictionary<i32, i32>, "l_linenumber" : dictionary<i32, i32>, "l_quantity" : dictionary<i32, f64>, "l_extendedprice" : dictionary<i32, f64>, "l_discount" : dictionary<i32, f64>, "l_tax" : dictionary<i32, f64>, "l_returnflag" : dictionary<i32, memref<1xi8>>, "l_linestatus" : dictionary<i32, memref<1xi8>>, "l_shipdate" : dictionary<i32, i32>, "l_commitdate" : dictionary<i32, i32>, "l_receiptdate" : dictionary<i32, i32>, "l_shipinstruct" : dictionary<i32, memref<25xi8>>, "l_shipmode" : dictionary<i32, memref<10xi8>>, "l_comment" : dictionary<i32, memref<44xi8>>, "size" : i32>
// forest
%constds_0 = "arith.constant"() <{value = dense<[102, 111, 114, 101, 115, 116]> : memref<?xi8>}> : () -> memref<?xi8>
%consti_1 = "arith.constant"() <{value = 7 : i32}> : () -> i32
%forest = sdql.external "ConstantString", %constds_0, %consti_1 : memref<?xi8>, i32 -> memref<6xi8>
// CANADA
%constds_2 = "arith.constant"() <{value = dense<[67, 65, 78, 65, 68, 65]> : memref<?xi8>}> : () -> memref<?xi8>
%consti_3 = "arith.constant"() <{value = 7 : i32}> : () -> i32
%canada = sdql.external "ConstantString", %constds_2, %consti_3 : memref<?xi8>, i32 -> memref<6xi8>
%fieldnode_4 = sdql.access_record %part "size" : record<"p_partkey": dictionary<i32, i32>, "p_name": dictionary<i32, memref<55xi8>>, "p_mfgr": dictionary<i32, memref<25xi8>>, "p_brand": dictionary<i32, memref<10xi8>>, "p_type": dictionary<i32, memref<25xi8>>, "p_size": dictionary<i32, i32>, "p_container": dictionary<i32, memref<10xi8>>, "p_retailprice": dictionary<i32, f64>, "p_comment": dictionary<i32, memref<23xi8>>, "size": i32> -> i32
%rangenode_5 = "func.call"(%fieldnode_4) <{callee = @range_builtin}> : (i32) -> dictionary<i32, i32>
%p_h = sdql.sum %rangenode_5 : dictionary<i32, i32> -> dictionary<i32, record<"_" : i32>> {
^bb_6(%i: i32, %_: i32):
  %fieldnode_8 = sdql.access_record %part "p_partkey" : record<"p_partkey": dictionary<i32, i32>, "p_name": dictionary<i32, memref<55xi8>>, "p_mfgr": dictionary<i32, memref<25xi8>>, "p_brand": dictionary<i32, memref<10xi8>>, "p_type": dictionary<i32, memref<25xi8>>, "p_size": dictionary<i32, i32>, "p_container": dictionary<i32, memref<10xi8>>, "p_retailprice": dictionary<i32, f64>, "p_comment": dictionary<i32, memref<23xi8>>, "size": i32> -> dictionary<i32, i32>
  %get_7 = sdql.lookup_dictionary %fieldnode_8 [%i : i32] : dictionary<i32, i32> -> i32
  %fieldnode_10 = sdql.access_record %part "p_name" : record<"p_partkey": dictionary<i32, i32>, "p_name": dictionary<i32, memref<55xi8>>, "p_mfgr": dictionary<i32, memref<25xi8>>, "p_brand": dictionary<i32, memref<10xi8>>, "p_type": dictionary<i32, memref<25xi8>>, "p_size": dictionary<i32, i32>, "p_container": dictionary<i32, memref<10xi8>>, "p_retailprice": dictionary<i32, f64>, "p_comment": dictionary<i32, memref<23xi8>>, "size": i32> -> dictionary<i32, memref<55xi8>>
  %get_9 = sdql.lookup_dictionary %fieldnode_10 [%i : i32] : dictionary<i32, memref<55xi8>> -> memref<55xi8>
  %fieldnode_12 = sdql.access_record %part "p_mfgr" : record<"p_partkey": dictionary<i32, i32>, "p_name": dictionary<i32, memref<55xi8>>, "p_mfgr": dictionary<i32, memref<25xi8>>, "p_brand": dictionary<i32, memref<10xi8>>, "p_type": dictionary<i32, memref<25xi8>>, "p_size": dictionary<i32, i32>, "p_container": dictionary<i32, memref<10xi8>>, "p_retailprice": dictionary<i32, f64>, "p_comment": dictionary<i32, memref<23xi8>>, "size": i32> -> dictionary<i32, memref<25xi8>>
  %get_11 = sdql.lookup_dictionary %fieldnode_12 [%i : i32] : dictionary<i32, memref<25xi8>> -> memref<25xi8>
  %fieldnode_14 = sdql.access_record %part "p_brand" : record<"p_partkey": dictionary<i32, i32>, "p_name": dictionary<i32, memref<55xi8>>, "p_mfgr": dictionary<i32, memref<25xi8>>, "p_brand": dictionary<i32, memref<10xi8>>, "p_type": dictionary<i32, memref<25xi8>>, "p_size": dictionary<i32, i32>, "p_container": dictionary<i32, memref<10xi8>>, "p_retailprice": dictionary<i32, f64>, "p_comment": dictionary<i32, memref<23xi8>>, "size": i32> -> dictionary<i32, memref<10xi8>>
  %get_13 = sdql.lookup_dictionary %fieldnode_14 [%i : i32] : dictionary<i32, memref<10xi8>> -> memref<10xi8>
  %fieldnode_16 = sdql.access_record %part "p_type" : record<"p_partkey": dictionary<i32, i32>, "p_name": dictionary<i32, memref<55xi8>>, "p_mfgr": dictionary<i32, memref<25xi8>>, "p_brand": dictionary<i32, memref<10xi8>>, "p_type": dictionary<i32, memref<25xi8>>, "p_size": dictionary<i32, i32>, "p_container": dictionary<i32, memref<10xi8>>, "p_retailprice": dictionary<i32, f64>, "p_comment": dictionary<i32, memref<23xi8>>, "size": i32> -> dictionary<i32, memref<25xi8>>
  %get_15 = sdql.lookup_dictionary %fieldnode_16 [%i : i32] : dictionary<i32, memref<25xi8>> -> memref<25xi8>
  %fieldnode_18 = sdql.access_record %part "p_size" : record<"p_partkey": dictionary<i32, i32>, "p_name": dictionary<i32, memref<55xi8>>, "p_mfgr": dictionary<i32, memref<25xi8>>, "p_brand": dictionary<i32, memref<10xi8>>, "p_type": dictionary<i32, memref<25xi8>>, "p_size": dictionary<i32, i32>, "p_container": dictionary<i32, memref<10xi8>>, "p_retailprice": dictionary<i32, f64>, "p_comment": dictionary<i32, memref<23xi8>>, "size": i32> -> dictionary<i32, i32>
  %get_17 = sdql.lookup_dictionary %fieldnode_18 [%i : i32] : dictionary<i32, i32> -> i32
  %fieldnode_20 = sdql.access_record %part "p_container" : record<"p_partkey": dictionary<i32, i32>, "p_name": dictionary<i32, memref<55xi8>>, "p_mfgr": dictionary<i32, memref<25xi8>>, "p_brand": dictionary<i32, memref<10xi8>>, "p_type": dictionary<i32, memref<25xi8>>, "p_size": dictionary<i32, i32>, "p_container": dictionary<i32, memref<10xi8>>, "p_retailprice": dictionary<i32, f64>, "p_comment": dictionary<i32, memref<23xi8>>, "size": i32> -> dictionary<i32, memref<10xi8>>
  %get_19 = sdql.lookup_dictionary %fieldnode_20 [%i : i32] : dictionary<i32, memref<10xi8>> -> memref<10xi8>
  %fieldnode_22 = sdql.access_record %part "p_retailprice" : record<"p_partkey": dictionary<i32, i32>, "p_name": dictionary<i32, memref<55xi8>>, "p_mfgr": dictionary<i32, memref<25xi8>>, "p_brand": dictionary<i32, memref<10xi8>>, "p_type": dictionary<i32, memref<25xi8>>, "p_size": dictionary<i32, i32>, "p_container": dictionary<i32, memref<10xi8>>, "p_retailprice": dictionary<i32, f64>, "p_comment": dictionary<i32, memref<23xi8>>, "size": i32> -> dictionary<i32, f64>
  %get_21 = sdql.lookup_dictionary %fieldnode_22 [%i : i32] : dictionary<i32, f64> -> f64
  %fieldnode_24 = sdql.access_record %part "p_comment" : record<"p_partkey": dictionary<i32, i32>, "p_name": dictionary<i32, memref<55xi8>>, "p_mfgr": dictionary<i32, memref<25xi8>>, "p_brand": dictionary<i32, memref<10xi8>>, "p_type": dictionary<i32, memref<25xi8>>, "p_size": dictionary<i32, i32>, "p_container": dictionary<i32, memref<10xi8>>, "p_retailprice": dictionary<i32, f64>, "p_comment": dictionary<i32, memref<23xi8>>, "size": i32> -> dictionary<i32, memref<23xi8>>
  %get_23 = sdql.lookup_dictionary %fieldnode_24 [%i : i32] : dictionary<i32, memref<23xi8>> -> memref<23xi8>
  %p = sdql.create_record {fields = ["p_partkey", "p_name", "p_mfgr", "p_brand", "p_type", "p_size", "p_container", "p_retailprice", "p_comment"]} %get_7, %get_9, %get_11, %get_13, %get_15, %get_17, %get_19, %get_21, %get_23 : i32, memref<55xi8>, memref<25xi8>, memref<10xi8>, memref<25xi8>, i32, memref<10xi8>, f64, memref<23xi8> -> record<"p_partkey": i32, "p_name": memref<55xi8>, "p_mfgr": memref<25xi8>, "p_brand": memref<10xi8>, "p_type": memref<25xi8>, "p_size": i32, "p_container": memref<10xi8>, "p_retailprice": f64, "p_comment": memref<23xi8>>
  %fieldnode_25 = sdql.access_record %p "p_name" : record<"p_partkey": i32, "p_name": memref<55xi8>, "p_mfgr": memref<25xi8>, "p_brand": memref<10xi8>, "p_type": memref<25xi8>, "p_size": i32, "p_container": memref<10xi8>, "p_retailprice": f64, "p_comment": memref<23xi8>> -> memref<55xi8>
  %external_26 = sdql.external "StrStartsWith", %fieldnode_25, %forest : memref<55xi8>, memref<6xi8> -> i1
  %if_27 = "scf.if"(%external_26) ({
    %fieldnode_30 = sdql.access_record %p "p_partkey" : record<"p_partkey": i32, "p_name": memref<55xi8>, "p_mfgr": memref<25xi8>, "p_brand": memref<10xi8>, "p_type": memref<25xi8>, "p_size": i32, "p_container": memref<10xi8>, "p_retailprice": f64, "p_comment": memref<23xi8>> -> i32
    %uniq_29 = sdql.unique %fieldnode_30 : i32 -> i32
    %fieldnode_31 = sdql.access_record %p "p_partkey" : record<"p_partkey": i32, "p_name": memref<55xi8>, "p_mfgr": memref<25xi8>, "p_brand": memref<10xi8>, "p_type": memref<25xi8>, "p_size": i32, "p_container": memref<10xi8>, "p_retailprice": f64, "p_comment": memref<23xi8>> -> i32
    %recnode_32 = sdql.create_record {fields = ["_"]} %fieldnode_31 : i32 -> record<"_": i32>
    %dict_28 = sdql.create_dictionary %uniq_29, %recnode_32 : i32, record<"_" : i32> -> dictionary<i32, record<"_" : i32>>
    "scf.yield"(%dict_28) : (dictionary<i32, record<"_" : i32>>) -> ()
  }, {
    %dict_33 = sdql.empty_dictionary : dictionary<i32, record<"_" : i32>>
    "scf.yield"(%dict_33) : (dictionary<i32, record<"_" : i32>>) -> ()
  }) : (i1) -> dictionary<i32, record<"_" : i32>>
  sdql.yield %if_27 : dictionary<i32, record<"_" : i32>>
}
%fieldnode_34 = sdql.access_record %nation "size" : record<"n_nationkey": dictionary<i32, i32>, "n_name": dictionary<i32, memref<25xi8>>, "n_regionkey": dictionary<i32, i32>, "n_comment": dictionary<i32, memref<152xi8>>, "size": i32> -> i32
%rangenode_35 = "func.call"(%fieldnode_34) <{callee = @range_builtin}> : (i32) -> dictionary<i32, i32>
%n_h = sdql.sum %rangenode_35 : dictionary<i32, i32> -> dictionary<i32, record<"_" : i32>> {
^bb_36(%i: i32, %_: i32):
  %fieldnode_38 = sdql.access_record %nation "n_nationkey" : record<"n_nationkey": dictionary<i32, i32>, "n_name": dictionary<i32, memref<25xi8>>, "n_regionkey": dictionary<i32, i32>, "n_comment": dictionary<i32, memref<152xi8>>, "size": i32> -> dictionary<i32, i32>
  %get_37 = sdql.lookup_dictionary %fieldnode_38 [%i : i32] : dictionary<i32, i32> -> i32
  %fieldnode_40 = sdql.access_record %nation "n_name" : record<"n_nationkey": dictionary<i32, i32>, "n_name": dictionary<i32, memref<25xi8>>, "n_regionkey": dictionary<i32, i32>, "n_comment": dictionary<i32, memref<152xi8>>, "size": i32> -> dictionary<i32, memref<25xi8>>
  %get_39 = sdql.lookup_dictionary %fieldnode_40 [%i : i32] : dictionary<i32, memref<25xi8>> -> memref<25xi8>
  %fieldnode_42 = sdql.access_record %nation "n_regionkey" : record<"n_nationkey": dictionary<i32, i32>, "n_name": dictionary<i32, memref<25xi8>>, "n_regionkey": dictionary<i32, i32>, "n_comment": dictionary<i32, memref<152xi8>>, "size": i32> -> dictionary<i32, i32>
  %get_41 = sdql.lookup_dictionary %fieldnode_42 [%i : i32] : dictionary<i32, i32> -> i32
  %fieldnode_44 = sdql.access_record %nation "n_comment" : record<"n_nationkey": dictionary<i32, i32>, "n_name": dictionary<i32, memref<25xi8>>, "n_regionkey": dictionary<i32, i32>, "n_comment": dictionary<i32, memref<152xi8>>, "size": i32> -> dictionary<i32, memref<152xi8>>
  %get_43 = sdql.lookup_dictionary %fieldnode_44 [%i : i32] : dictionary<i32, memref<152xi8>> -> memref<152xi8>
  %n = sdql.create_record {fields = ["n_nationkey", "n_name", "n_regionkey", "n_comment"]} %get_37, %get_39, %get_41, %get_43 : i32, memref<25xi8>, i32, memref<152xi8> -> record<"n_nationkey": i32, "n_name": memref<25xi8>, "n_regionkey": i32, "n_comment": memref<152xi8>>
  %fieldnode_45 = sdql.access_record %n "n_name" : record<"n_nationkey": i32, "n_name": memref<25xi8>, "n_regionkey": i32, "n_comment": memref<152xi8>> -> memref<25xi8>
  %cmp_46 = sdql.cmp %fieldnode_45, %canada : memref<25xi8>, memref<6xi8> -> i1
  %if_47 = "scf.if"(%cmp_46) ({
    %fieldnode_50 = sdql.access_record %n "n_nationkey" : record<"n_nationkey": i32, "n_name": memref<25xi8>, "n_regionkey": i32, "n_comment": memref<152xi8>> -> i32
    %uniq_49 = sdql.unique %fieldnode_50 : i32 -> i32
    %fieldnode_51 = sdql.access_record %n "n_nationkey" : record<"n_nationkey": i32, "n_name": memref<25xi8>, "n_regionkey": i32, "n_comment": memref<152xi8>> -> i32
    %recnode_52 = sdql.create_record {fields = ["_"]} %fieldnode_51 : i32 -> record<"_": i32>
    %dict_48 = sdql.create_dictionary %uniq_49, %recnode_52 : i32, record<"_" : i32> -> dictionary<i32, record<"_" : i32>>
    "scf.yield"(%dict_48) : (dictionary<i32, record<"_" : i32>>) -> ()
  }, {
    %dict_53 = sdql.empty_dictionary : dictionary<i32, record<"_" : i32>>
    "scf.yield"(%dict_53) : (dictionary<i32, record<"_" : i32>>) -> ()
  }) : (i1) -> dictionary<i32, record<"_" : i32>>
  sdql.yield %if_47 : dictionary<i32, record<"_" : i32>>
}
%fieldnode_54 = sdql.access_record %supplier "size" : record<"s_suppkey": dictionary<i32, i32>, "s_name": dictionary<i32, memref<25xi8>>, "s_address": dictionary<i32, memref<40xi8>>, "s_nationkey": dictionary<i32, i32>, "s_phone": dictionary<i32, memref<15xi8>>, "s_acctbal": dictionary<i32, f64>, "s_comment": dictionary<i32, memref<101xi8>>, "size": i32> -> i32
%rangenode_55 = "func.call"(%fieldnode_54) <{callee = @range_builtin}> : (i32) -> dictionary<i32, i32>
%s_h = sdql.sum %rangenode_55 : dictionary<i32, i32> -> dictionary<i32, record<"_" : i32>> {
^bb_56(%i: i32, %_: i32):
  %fieldnode_58 = sdql.access_record %supplier "s_suppkey" : record<"s_suppkey": dictionary<i32, i32>, "s_name": dictionary<i32, memref<25xi8>>, "s_address": dictionary<i32, memref<40xi8>>, "s_nationkey": dictionary<i32, i32>, "s_phone": dictionary<i32, memref<15xi8>>, "s_acctbal": dictionary<i32, f64>, "s_comment": dictionary<i32, memref<101xi8>>, "size": i32> -> dictionary<i32, i32>
  %get_57 = sdql.lookup_dictionary %fieldnode_58 [%i : i32] : dictionary<i32, i32> -> i32
  %fieldnode_60 = sdql.access_record %supplier "s_name" : record<"s_suppkey": dictionary<i32, i32>, "s_name": dictionary<i32, memref<25xi8>>, "s_address": dictionary<i32, memref<40xi8>>, "s_nationkey": dictionary<i32, i32>, "s_phone": dictionary<i32, memref<15xi8>>, "s_acctbal": dictionary<i32, f64>, "s_comment": dictionary<i32, memref<101xi8>>, "size": i32> -> dictionary<i32, memref<25xi8>>
  %get_59 = sdql.lookup_dictionary %fieldnode_60 [%i : i32] : dictionary<i32, memref<25xi8>> -> memref<25xi8>
  %fieldnode_62 = sdql.access_record %supplier "s_address" : record<"s_suppkey": dictionary<i32, i32>, "s_name": dictionary<i32, memref<25xi8>>, "s_address": dictionary<i32, memref<40xi8>>, "s_nationkey": dictionary<i32, i32>, "s_phone": dictionary<i32, memref<15xi8>>, "s_acctbal": dictionary<i32, f64>, "s_comment": dictionary<i32, memref<101xi8>>, "size": i32> -> dictionary<i32, memref<40xi8>>
  %get_61 = sdql.lookup_dictionary %fieldnode_62 [%i : i32] : dictionary<i32, memref<40xi8>> -> memref<40xi8>
  %fieldnode_64 = sdql.access_record %supplier "s_nationkey" : record<"s_suppkey": dictionary<i32, i32>, "s_name": dictionary<i32, memref<25xi8>>, "s_address": dictionary<i32, memref<40xi8>>, "s_nationkey": dictionary<i32, i32>, "s_phone": dictionary<i32, memref<15xi8>>, "s_acctbal": dictionary<i32, f64>, "s_comment": dictionary<i32, memref<101xi8>>, "size": i32> -> dictionary<i32, i32>
  %get_63 = sdql.lookup_dictionary %fieldnode_64 [%i : i32] : dictionary<i32, i32> -> i32
  %fieldnode_66 = sdql.access_record %supplier "s_phone" : record<"s_suppkey": dictionary<i32, i32>, "s_name": dictionary<i32, memref<25xi8>>, "s_address": dictionary<i32, memref<40xi8>>, "s_nationkey": dictionary<i32, i32>, "s_phone": dictionary<i32, memref<15xi8>>, "s_acctbal": dictionary<i32, f64>, "s_comment": dictionary<i32, memref<101xi8>>, "size": i32> -> dictionary<i32, memref<15xi8>>
  %get_65 = sdql.lookup_dictionary %fieldnode_66 [%i : i32] : dictionary<i32, memref<15xi8>> -> memref<15xi8>
  %fieldnode_68 = sdql.access_record %supplier "s_acctbal" : record<"s_suppkey": dictionary<i32, i32>, "s_name": dictionary<i32, memref<25xi8>>, "s_address": dictionary<i32, memref<40xi8>>, "s_nationkey": dictionary<i32, i32>, "s_phone": dictionary<i32, memref<15xi8>>, "s_acctbal": dictionary<i32, f64>, "s_comment": dictionary<i32, memref<101xi8>>, "size": i32> -> dictionary<i32, f64>
  %get_67 = sdql.lookup_dictionary %fieldnode_68 [%i : i32] : dictionary<i32, f64> -> f64
  %fieldnode_70 = sdql.access_record %supplier "s_comment" : record<"s_suppkey": dictionary<i32, i32>, "s_name": dictionary<i32, memref<25xi8>>, "s_address": dictionary<i32, memref<40xi8>>, "s_nationkey": dictionary<i32, i32>, "s_phone": dictionary<i32, memref<15xi8>>, "s_acctbal": dictionary<i32, f64>, "s_comment": dictionary<i32, memref<101xi8>>, "size": i32> -> dictionary<i32, memref<101xi8>>
  %get_69 = sdql.lookup_dictionary %fieldnode_70 [%i : i32] : dictionary<i32, memref<101xi8>> -> memref<101xi8>
  %s = sdql.create_record {fields = ["s_suppkey", "s_name", "s_address", "s_nationkey", "s_phone", "s_acctbal", "s_comment"]} %get_57, %get_59, %get_61, %get_63, %get_65, %get_67, %get_69 : i32, memref<25xi8>, memref<40xi8>, i32, memref<15xi8>, f64, memref<101xi8> -> record<"s_suppkey": i32, "s_name": memref<25xi8>, "s_address": memref<40xi8>, "s_nationkey": i32, "s_phone": memref<15xi8>, "s_acctbal": f64, "s_comment": memref<101xi8>>
  %fieldnode_74 = sdql.access_record %s "s_nationkey" : record<"s_suppkey": i32, "s_name": memref<25xi8>, "s_address": memref<40xi8>, "s_nationkey": i32, "s_phone": memref<15xi8>, "s_acctbal": f64, "s_comment": memref<101xi8>> -> i32
  %get_73 = sdql.lookup_dictionary %n_h [%fieldnode_74 : i32] : dictionary<i32, record<"_" : i32>> -> record<"_" : i32>
  %dict_75 = sdql.empty_dictionary : record<"_" : i32>
  %cmp_76 = sdql.cmp %get_73, %dict_75 : record<"_" : i32>, record<"_" : i32> -> i1
  %zero_71 = "arith.constant"() <{value = 0 : i1}> : () -> i1
  %neg_72 = "arith.cmpi"(%cmp_76, %zero_71) <{predicate = 0}> : (i1, i1) -> i1
  %if_77 = "scf.if"(%neg_72) ({
    %fieldnode_80 = sdql.access_record %s "s_suppkey" : record<"s_suppkey": i32, "s_name": memref<25xi8>, "s_address": memref<40xi8>, "s_nationkey": i32, "s_phone": memref<15xi8>, "s_acctbal": f64, "s_comment": memref<101xi8>> -> i32
    %uniq_79 = sdql.unique %fieldnode_80 : i32 -> i32
    %fieldnode_81 = sdql.access_record %s "s_suppkey" : record<"s_suppkey": i32, "s_name": memref<25xi8>, "s_address": memref<40xi8>, "s_nationkey": i32, "s_phone": memref<15xi8>, "s_acctbal": f64, "s_comment": memref<101xi8>> -> i32
    %recnode_82 = sdql.create_record {fields = ["_"]} %fieldnode_81 : i32 -> record<"_": i32>
    %dict_78 = sdql.create_dictionary %uniq_79, %recnode_82 : i32, record<"_" : i32> -> dictionary<i32, record<"_" : i32>>
    "scf.yield"(%dict_78) : (dictionary<i32, record<"_" : i32>>) -> ()
  }, {
    %dict_83 = sdql.empty_dictionary : dictionary<i32, record<"_" : i32>>
    "scf.yield"(%dict_83) : (dictionary<i32, record<"_" : i32>>) -> ()
  }) : (i1) -> dictionary<i32, record<"_" : i32>>
  sdql.yield %if_77 : dictionary<i32, record<"_" : i32>>
}
%fieldnode_84 = sdql.access_record %lineitem "size" : record<"l_orderkey": dictionary<i32, i32>, "l_partkey": dictionary<i32, i32>, "l_suppkey": dictionary<i32, i32>, "l_linenumber": dictionary<i32, i32>, "l_quantity": dictionary<i32, f64>, "l_extendedprice": dictionary<i32, f64>, "l_discount": dictionary<i32, f64>, "l_tax": dictionary<i32, f64>, "l_returnflag": dictionary<i32, memref<1xi8>>, "l_linestatus": dictionary<i32, memref<1xi8>>, "l_shipdate": dictionary<i32, i32>, "l_commitdate": dictionary<i32, i32>, "l_receiptdate": dictionary<i32, i32>, "l_shipinstruct": dictionary<i32, memref<25xi8>>, "l_shipmode": dictionary<i32, memref<10xi8>>, "l_comment": dictionary<i32, memref<44xi8>>, "size": i32> -> i32
%rangenode_85 = "func.call"(%fieldnode_84) <{callee = @range_builtin}> : (i32) -> dictionary<i32, i32>
%l_h = sdql.sum %rangenode_85 : dictionary<i32, i32> -> dictionary<record<"_" : i32, "_" : i32>, f64> {
^bb_86(%i: i32, %_: i32):
  %fieldnode_88 = sdql.access_record %lineitem "l_orderkey" : record<"l_orderkey": dictionary<i32, i32>, "l_partkey": dictionary<i32, i32>, "l_suppkey": dictionary<i32, i32>, "l_linenumber": dictionary<i32, i32>, "l_quantity": dictionary<i32, f64>, "l_extendedprice": dictionary<i32, f64>, "l_discount": dictionary<i32, f64>, "l_tax": dictionary<i32, f64>, "l_returnflag": dictionary<i32, memref<1xi8>>, "l_linestatus": dictionary<i32, memref<1xi8>>, "l_shipdate": dictionary<i32, i32>, "l_commitdate": dictionary<i32, i32>, "l_receiptdate": dictionary<i32, i32>, "l_shipinstruct": dictionary<i32, memref<25xi8>>, "l_shipmode": dictionary<i32, memref<10xi8>>, "l_comment": dictionary<i32, memref<44xi8>>, "size": i32> -> dictionary<i32, i32>
  %get_87 = sdql.lookup_dictionary %fieldnode_88 [%i : i32] : dictionary<i32, i32> -> i32
  %fieldnode_90 = sdql.access_record %lineitem "l_partkey" : record<"l_orderkey": dictionary<i32, i32>, "l_partkey": dictionary<i32, i32>, "l_suppkey": dictionary<i32, i32>, "l_linenumber": dictionary<i32, i32>, "l_quantity": dictionary<i32, f64>, "l_extendedprice": dictionary<i32, f64>, "l_discount": dictionary<i32, f64>, "l_tax": dictionary<i32, f64>, "l_returnflag": dictionary<i32, memref<1xi8>>, "l_linestatus": dictionary<i32, memref<1xi8>>, "l_shipdate": dictionary<i32, i32>, "l_commitdate": dictionary<i32, i32>, "l_receiptdate": dictionary<i32, i32>, "l_shipinstruct": dictionary<i32, memref<25xi8>>, "l_shipmode": dictionary<i32, memref<10xi8>>, "l_comment": dictionary<i32, memref<44xi8>>, "size": i32> -> dictionary<i32, i32>
  %get_89 = sdql.lookup_dictionary %fieldnode_90 [%i : i32] : dictionary<i32, i32> -> i32
  %fieldnode_92 = sdql.access_record %lineitem "l_suppkey" : record<"l_orderkey": dictionary<i32, i32>, "l_partkey": dictionary<i32, i32>, "l_suppkey": dictionary<i32, i32>, "l_linenumber": dictionary<i32, i32>, "l_quantity": dictionary<i32, f64>, "l_extendedprice": dictionary<i32, f64>, "l_discount": dictionary<i32, f64>, "l_tax": dictionary<i32, f64>, "l_returnflag": dictionary<i32, memref<1xi8>>, "l_linestatus": dictionary<i32, memref<1xi8>>, "l_shipdate": dictionary<i32, i32>, "l_commitdate": dictionary<i32, i32>, "l_receiptdate": dictionary<i32, i32>, "l_shipinstruct": dictionary<i32, memref<25xi8>>, "l_shipmode": dictionary<i32, memref<10xi8>>, "l_comment": dictionary<i32, memref<44xi8>>, "size": i32> -> dictionary<i32, i32>
  %get_91 = sdql.lookup_dictionary %fieldnode_92 [%i : i32] : dictionary<i32, i32> -> i32
  %fieldnode_94 = sdql.access_record %lineitem "l_linenumber" : record<"l_orderkey": dictionary<i32, i32>, "l_partkey": dictionary<i32, i32>, "l_suppkey": dictionary<i32, i32>, "l_linenumber": dictionary<i32, i32>, "l_quantity": dictionary<i32, f64>, "l_extendedprice": dictionary<i32, f64>, "l_discount": dictionary<i32, f64>, "l_tax": dictionary<i32, f64>, "l_returnflag": dictionary<i32, memref<1xi8>>, "l_linestatus": dictionary<i32, memref<1xi8>>, "l_shipdate": dictionary<i32, i32>, "l_commitdate": dictionary<i32, i32>, "l_receiptdate": dictionary<i32, i32>, "l_shipinstruct": dictionary<i32, memref<25xi8>>, "l_shipmode": dictionary<i32, memref<10xi8>>, "l_comment": dictionary<i32, memref<44xi8>>, "size": i32> -> dictionary<i32, i32>
  %get_93 = sdql.lookup_dictionary %fieldnode_94 [%i : i32] : dictionary<i32, i32> -> i32
  %fieldnode_96 = sdql.access_record %lineitem "l_quantity" : record<"l_orderkey": dictionary<i32, i32>, "l_partkey": dictionary<i32, i32>, "l_suppkey": dictionary<i32, i32>, "l_linenumber": dictionary<i32, i32>, "l_quantity": dictionary<i32, f64>, "l_extendedprice": dictionary<i32, f64>, "l_discount": dictionary<i32, f64>, "l_tax": dictionary<i32, f64>, "l_returnflag": dictionary<i32, memref<1xi8>>, "l_linestatus": dictionary<i32, memref<1xi8>>, "l_shipdate": dictionary<i32, i32>, "l_commitdate": dictionary<i32, i32>, "l_receiptdate": dictionary<i32, i32>, "l_shipinstruct": dictionary<i32, memref<25xi8>>, "l_shipmode": dictionary<i32, memref<10xi8>>, "l_comment": dictionary<i32, memref<44xi8>>, "size": i32> -> dictionary<i32, f64>
  %get_95 = sdql.lookup_dictionary %fieldnode_96 [%i : i32] : dictionary<i32, f64> -> f64
  %fieldnode_98 = sdql.access_record %lineitem "l_extendedprice" : record<"l_orderkey": dictionary<i32, i32>, "l_partkey": dictionary<i32, i32>, "l_suppkey": dictionary<i32, i32>, "l_linenumber": dictionary<i32, i32>, "l_quantity": dictionary<i32, f64>, "l_extendedprice": dictionary<i32, f64>, "l_discount": dictionary<i32, f64>, "l_tax": dictionary<i32, f64>, "l_returnflag": dictionary<i32, memref<1xi8>>, "l_linestatus": dictionary<i32, memref<1xi8>>, "l_shipdate": dictionary<i32, i32>, "l_commitdate": dictionary<i32, i32>, "l_receiptdate": dictionary<i32, i32>, "l_shipinstruct": dictionary<i32, memref<25xi8>>, "l_shipmode": dictionary<i32, memref<10xi8>>, "l_comment": dictionary<i32, memref<44xi8>>, "size": i32> -> dictionary<i32, f64>
  %get_97 = sdql.lookup_dictionary %fieldnode_98 [%i : i32] : dictionary<i32, f64> -> f64
  %fieldnode_100 = sdql.access_record %lineitem "l_discount" : record<"l_orderkey": dictionary<i32, i32>, "l_partkey": dictionary<i32, i32>, "l_suppkey": dictionary<i32, i32>, "l_linenumber": dictionary<i32, i32>, "l_quantity": dictionary<i32, f64>, "l_extendedprice": dictionary<i32, f64>, "l_discount": dictionary<i32, f64>, "l_tax": dictionary<i32, f64>, "l_returnflag": dictionary<i32, memref<1xi8>>, "l_linestatus": dictionary<i32, memref<1xi8>>, "l_shipdate": dictionary<i32, i32>, "l_commitdate": dictionary<i32, i32>, "l_receiptdate": dictionary<i32, i32>, "l_shipinstruct": dictionary<i32, memref<25xi8>>, "l_shipmode": dictionary<i32, memref<10xi8>>, "l_comment": dictionary<i32, memref<44xi8>>, "size": i32> -> dictionary<i32, f64>
  %get_99 = sdql.lookup_dictionary %fieldnode_100 [%i : i32] : dictionary<i32, f64> -> f64
  %fieldnode_102 = sdql.access_record %lineitem "l_tax" : record<"l_orderkey": dictionary<i32, i32>, "l_partkey": dictionary<i32, i32>, "l_suppkey": dictionary<i32, i32>, "l_linenumber": dictionary<i32, i32>, "l_quantity": dictionary<i32, f64>, "l_extendedprice": dictionary<i32, f64>, "l_discount": dictionary<i32, f64>, "l_tax": dictionary<i32, f64>, "l_returnflag": dictionary<i32, memref<1xi8>>, "l_linestatus": dictionary<i32, memref<1xi8>>, "l_shipdate": dictionary<i32, i32>, "l_commitdate": dictionary<i32, i32>, "l_receiptdate": dictionary<i32, i32>, "l_shipinstruct": dictionary<i32, memref<25xi8>>, "l_shipmode": dictionary<i32, memref<10xi8>>, "l_comment": dictionary<i32, memref<44xi8>>, "size": i32> -> dictionary<i32, f64>
  %get_101 = sdql.lookup_dictionary %fieldnode_102 [%i : i32] : dictionary<i32, f64> -> f64
  %fieldnode_104 = sdql.access_record %lineitem "l_returnflag" : record<"l_orderkey": dictionary<i32, i32>, "l_partkey": dictionary<i32, i32>, "l_suppkey": dictionary<i32, i32>, "l_linenumber": dictionary<i32, i32>, "l_quantity": dictionary<i32, f64>, "l_extendedprice": dictionary<i32, f64>, "l_discount": dictionary<i32, f64>, "l_tax": dictionary<i32, f64>, "l_returnflag": dictionary<i32, memref<1xi8>>, "l_linestatus": dictionary<i32, memref<1xi8>>, "l_shipdate": dictionary<i32, i32>, "l_commitdate": dictionary<i32, i32>, "l_receiptdate": dictionary<i32, i32>, "l_shipinstruct": dictionary<i32, memref<25xi8>>, "l_shipmode": dictionary<i32, memref<10xi8>>, "l_comment": dictionary<i32, memref<44xi8>>, "size": i32> -> dictionary<i32, memref<1xi8>>
  %get_103 = sdql.lookup_dictionary %fieldnode_104 [%i : i32] : dictionary<i32, memref<1xi8>> -> memref<1xi8>
  %fieldnode_106 = sdql.access_record %lineitem "l_linestatus" : record<"l_orderkey": dictionary<i32, i32>, "l_partkey": dictionary<i32, i32>, "l_suppkey": dictionary<i32, i32>, "l_linenumber": dictionary<i32, i32>, "l_quantity": dictionary<i32, f64>, "l_extendedprice": dictionary<i32, f64>, "l_discount": dictionary<i32, f64>, "l_tax": dictionary<i32, f64>, "l_returnflag": dictionary<i32, memref<1xi8>>, "l_linestatus": dictionary<i32, memref<1xi8>>, "l_shipdate": dictionary<i32, i32>, "l_commitdate": dictionary<i32, i32>, "l_receiptdate": dictionary<i32, i32>, "l_shipinstruct": dictionary<i32, memref<25xi8>>, "l_shipmode": dictionary<i32, memref<10xi8>>, "l_comment": dictionary<i32, memref<44xi8>>, "size": i32> -> dictionary<i32, memref<1xi8>>
  %get_105 = sdql.lookup_dictionary %fieldnode_106 [%i : i32] : dictionary<i32, memref<1xi8>> -> memref<1xi8>
  %fieldnode_108 = sdql.access_record %lineitem "l_shipdate" : record<"l_orderkey": dictionary<i32, i32>, "l_partkey": dictionary<i32, i32>, "l_suppkey": dictionary<i32, i32>, "l_linenumber": dictionary<i32, i32>, "l_quantity": dictionary<i32, f64>, "l_extendedprice": dictionary<i32, f64>, "l_discount": dictionary<i32, f64>, "l_tax": dictionary<i32, f64>, "l_returnflag": dictionary<i32, memref<1xi8>>, "l_linestatus": dictionary<i32, memref<1xi8>>, "l_shipdate": dictionary<i32, i32>, "l_commitdate": dictionary<i32, i32>, "l_receiptdate": dictionary<i32, i32>, "l_shipinstruct": dictionary<i32, memref<25xi8>>, "l_shipmode": dictionary<i32, memref<10xi8>>, "l_comment": dictionary<i32, memref<44xi8>>, "size": i32> -> dictionary<i32, i32>
  %get_107 = sdql.lookup_dictionary %fieldnode_108 [%i : i32] : dictionary<i32, i32> -> i32
  %fieldnode_110 = sdql.access_record %lineitem "l_commitdate" : record<"l_orderkey": dictionary<i32, i32>, "l_partkey": dictionary<i32, i32>, "l_suppkey": dictionary<i32, i32>, "l_linenumber": dictionary<i32, i32>, "l_quantity": dictionary<i32, f64>, "l_extendedprice": dictionary<i32, f64>, "l_discount": dictionary<i32, f64>, "l_tax": dictionary<i32, f64>, "l_returnflag": dictionary<i32, memref<1xi8>>, "l_linestatus": dictionary<i32, memref<1xi8>>, "l_shipdate": dictionary<i32, i32>, "l_commitdate": dictionary<i32, i32>, "l_receiptdate": dictionary<i32, i32>, "l_shipinstruct": dictionary<i32, memref<25xi8>>, "l_shipmode": dictionary<i32, memref<10xi8>>, "l_comment": dictionary<i32, memref<44xi8>>, "size": i32> -> dictionary<i32, i32>
  %get_109 = sdql.lookup_dictionary %fieldnode_110 [%i : i32] : dictionary<i32, i32> -> i32
  %fieldnode_112 = sdql.access_record %lineitem "l_receiptdate" : record<"l_orderkey": dictionary<i32, i32>, "l_partkey": dictionary<i32, i32>, "l_suppkey": dictionary<i32, i32>, "l_linenumber": dictionary<i32, i32>, "l_quantity": dictionary<i32, f64>, "l_extendedprice": dictionary<i32, f64>, "l_discount": dictionary<i32, f64>, "l_tax": dictionary<i32, f64>, "l_returnflag": dictionary<i32, memref<1xi8>>, "l_linestatus": dictionary<i32, memref<1xi8>>, "l_shipdate": dictionary<i32, i32>, "l_commitdate": dictionary<i32, i32>, "l_receiptdate": dictionary<i32, i32>, "l_shipinstruct": dictionary<i32, memref<25xi8>>, "l_shipmode": dictionary<i32, memref<10xi8>>, "l_comment": dictionary<i32, memref<44xi8>>, "size": i32> -> dictionary<i32, i32>
  %get_111 = sdql.lookup_dictionary %fieldnode_112 [%i : i32] : dictionary<i32, i32> -> i32
  %fieldnode_114 = sdql.access_record %lineitem "l_shipinstruct" : record<"l_orderkey": dictionary<i32, i32>, "l_partkey": dictionary<i32, i32>, "l_suppkey": dictionary<i32, i32>, "l_linenumber": dictionary<i32, i32>, "l_quantity": dictionary<i32, f64>, "l_extendedprice": dictionary<i32, f64>, "l_discount": dictionary<i32, f64>, "l_tax": dictionary<i32, f64>, "l_returnflag": dictionary<i32, memref<1xi8>>, "l_linestatus": dictionary<i32, memref<1xi8>>, "l_shipdate": dictionary<i32, i32>, "l_commitdate": dictionary<i32, i32>, "l_receiptdate": dictionary<i32, i32>, "l_shipinstruct": dictionary<i32, memref<25xi8>>, "l_shipmode": dictionary<i32, memref<10xi8>>, "l_comment": dictionary<i32, memref<44xi8>>, "size": i32> -> dictionary<i32, memref<25xi8>>
  %get_113 = sdql.lookup_dictionary %fieldnode_114 [%i : i32] : dictionary<i32, memref<25xi8>> -> memref<25xi8>
  %fieldnode_116 = sdql.access_record %lineitem "l_shipmode" : record<"l_orderkey": dictionary<i32, i32>, "l_partkey": dictionary<i32, i32>, "l_suppkey": dictionary<i32, i32>, "l_linenumber": dictionary<i32, i32>, "l_quantity": dictionary<i32, f64>, "l_extendedprice": dictionary<i32, f64>, "l_discount": dictionary<i32, f64>, "l_tax": dictionary<i32, f64>, "l_returnflag": dictionary<i32, memref<1xi8>>, "l_linestatus": dictionary<i32, memref<1xi8>>, "l_shipdate": dictionary<i32, i32>, "l_commitdate": dictionary<i32, i32>, "l_receiptdate": dictionary<i32, i32>, "l_shipinstruct": dictionary<i32, memref<25xi8>>, "l_shipmode": dictionary<i32, memref<10xi8>>, "l_comment": dictionary<i32, memref<44xi8>>, "size": i32> -> dictionary<i32, memref<10xi8>>
  %get_115 = sdql.lookup_dictionary %fieldnode_116 [%i : i32] : dictionary<i32, memref<10xi8>> -> memref<10xi8>
  %fieldnode_118 = sdql.access_record %lineitem "l_comment" : record<"l_orderkey": dictionary<i32, i32>, "l_partkey": dictionary<i32, i32>, "l_suppkey": dictionary<i32, i32>, "l_linenumber": dictionary<i32, i32>, "l_quantity": dictionary<i32, f64>, "l_extendedprice": dictionary<i32, f64>, "l_discount": dictionary<i32, f64>, "l_tax": dictionary<i32, f64>, "l_returnflag": dictionary<i32, memref<1xi8>>, "l_linestatus": dictionary<i32, memref<1xi8>>, "l_shipdate": dictionary<i32, i32>, "l_commitdate": dictionary<i32, i32>, "l_receiptdate": dictionary<i32, i32>, "l_shipinstruct": dictionary<i32, memref<25xi8>>, "l_shipmode": dictionary<i32, memref<10xi8>>, "l_comment": dictionary<i32, memref<44xi8>>, "size": i32> -> dictionary<i32, memref<44xi8>>
  %get_117 = sdql.lookup_dictionary %fieldnode_118 [%i : i32] : dictionary<i32, memref<44xi8>> -> memref<44xi8>
  %l = sdql.create_record {fields = ["l_orderkey", "l_partkey", "l_suppkey", "l_linenumber", "l_quantity", "l_extendedprice", "l_discount", "l_tax", "l_returnflag", "l_linestatus", "l_shipdate", "l_commitdate", "l_receiptdate", "l_shipinstruct", "l_shipmode", "l_comment"]} %get_87, %get_89, %get_91, %get_93, %get_95, %get_97, %get_99, %get_101, %get_103, %get_105, %get_107, %get_109, %get_111, %get_113, %get_115, %get_117 : i32, i32, i32, i32, f64, f64, f64, f64, memref<1xi8>, memref<1xi8>, i32, i32, i32, memref<25xi8>, memref<10xi8>, memref<44xi8> -> record<"l_orderkey": i32, "l_partkey": i32, "l_suppkey": i32, "l_linenumber": i32, "l_quantity": f64, "l_extendedprice": f64, "l_discount": f64, "l_tax": f64, "l_returnflag": memref<1xi8>, "l_linestatus": memref<1xi8>, "l_shipdate": i32, "l_commitdate": i32, "l_receiptdate": i32, "l_shipinstruct": memref<25xi8>, "l_shipmode": memref<10xi8>, "l_comment": memref<44xi8>>
  %constda_119 = "arith.constant"() <{value = 19940101 : i32}> : () -> i32
  %fieldnode_120 = sdql.access_record %l "l_shipdate" : record<"l_orderkey": i32, "l_partkey": i32, "l_suppkey": i32, "l_linenumber": i32, "l_quantity": f64, "l_extendedprice": f64, "l_discount": f64, "l_tax": f64, "l_returnflag": memref<1xi8>, "l_linestatus": memref<1xi8>, "l_shipdate": i32, "l_commitdate": i32, "l_receiptdate": i32, "l_shipinstruct": memref<25xi8>, "l_shipmode": memref<10xi8>, "l_comment": memref<44xi8>> -> i32
  %cmpi_121 = "arith.cmpi"(%constda_119, %fieldnode_120) <{predicate = 3}> : (i32, i32) -> i1
  %if_122 = "scf.if"(%cmpi_121) ({
    %fieldnode_123 = sdql.access_record %l "l_shipdate" : record<"l_orderkey": i32, "l_partkey": i32, "l_suppkey": i32, "l_linenumber": i32, "l_quantity": f64, "l_extendedprice": f64, "l_discount": f64, "l_tax": f64, "l_returnflag": memref<1xi8>, "l_linestatus": memref<1xi8>, "l_shipdate": i32, "l_commitdate": i32, "l_receiptdate": i32, "l_shipinstruct": memref<25xi8>, "l_shipmode": memref<10xi8>, "l_comment": memref<44xi8>> -> i32
    %constda_124 = "arith.constant"() <{value = 19950101 : i32}> : () -> i32
    %cmpi_125 = "arith.cmpi"(%fieldnode_123, %constda_124) <{predicate = 2}> : (i32, i32) -> i1
    "scf.yield"(%cmpi_125) : (i1) -> ()
  }, {
    %constv_126 = "arith.constant"() <{value = 0 : i1}> : () -> i1
    "scf.yield"(%constv_126) : (i1) -> ()
  }) : (i1) -> i1
  %if_127 = "scf.if"(%if_122) ({
    %fieldnode_131 = sdql.access_record %l "l_partkey" : record<"l_orderkey": i32, "l_partkey": i32, "l_suppkey": i32, "l_linenumber": i32, "l_quantity": f64, "l_extendedprice": f64, "l_discount": f64, "l_tax": f64, "l_returnflag": memref<1xi8>, "l_linestatus": memref<1xi8>, "l_shipdate": i32, "l_commitdate": i32, "l_receiptdate": i32, "l_shipinstruct": memref<25xi8>, "l_shipmode": memref<10xi8>, "l_comment": memref<44xi8>> -> i32
    %get_130 = sdql.lookup_dictionary %p_h [%fieldnode_131 : i32] : dictionary<i32, record<"_" : i32>> -> record<"_" : i32>
    %dict_132 = sdql.empty_dictionary : record<"_" : i32>
    %cmp_133 = sdql.cmp %get_130, %dict_132 : record<"_" : i32>, record<"_" : i32> -> i1
    %zero_128 = "arith.constant"() <{value = 0 : i1}> : () -> i1
    %neg_129 = "arith.cmpi"(%cmp_133, %zero_128) <{predicate = 0}> : (i1, i1) -> i1
    "scf.yield"(%neg_129) : (i1) -> ()
  }, {
    %constv_134 = "arith.constant"() <{value = 0 : i1}> : () -> i1
    "scf.yield"(%constv_134) : (i1) -> ()
  }) : (i1) -> i1
  %if_135 = "scf.if"(%if_127) ({
    %fieldnode_139 = sdql.access_record %l "l_suppkey" : record<"l_orderkey": i32, "l_partkey": i32, "l_suppkey": i32, "l_linenumber": i32, "l_quantity": f64, "l_extendedprice": f64, "l_discount": f64, "l_tax": f64, "l_returnflag": memref<1xi8>, "l_linestatus": memref<1xi8>, "l_shipdate": i32, "l_commitdate": i32, "l_receiptdate": i32, "l_shipinstruct": memref<25xi8>, "l_shipmode": memref<10xi8>, "l_comment": memref<44xi8>> -> i32
    %get_138 = sdql.lookup_dictionary %s_h [%fieldnode_139 : i32] : dictionary<i32, record<"_" : i32>> -> record<"_" : i32>
    %dict_140 = sdql.empty_dictionary : record<"_" : i32>
    %cmp_141 = sdql.cmp %get_138, %dict_140 : record<"_" : i32>, record<"_" : i32> -> i1
    %zero_136 = "arith.constant"() <{value = 0 : i1}> : () -> i1
    %neg_137 = "arith.cmpi"(%cmp_141, %zero_136) <{predicate = 0}> : (i1, i1) -> i1
    "scf.yield"(%neg_137) : (i1) -> ()
  }, {
    %constv_142 = "arith.constant"() <{value = 0 : i1}> : () -> i1
    "scf.yield"(%constv_142) : (i1) -> ()
  }) : (i1) -> i1
  %if_143 = "scf.if"(%if_135) ({
    %fieldnode_146 = sdql.access_record %l "l_partkey" : record<"l_orderkey": i32, "l_partkey": i32, "l_suppkey": i32, "l_linenumber": i32, "l_quantity": f64, "l_extendedprice": f64, "l_discount": f64, "l_tax": f64, "l_returnflag": memref<1xi8>, "l_linestatus": memref<1xi8>, "l_shipdate": i32, "l_commitdate": i32, "l_receiptdate": i32, "l_shipinstruct": memref<25xi8>, "l_shipmode": memref<10xi8>, "l_comment": memref<44xi8>> -> i32
    %fieldnode_147 = sdql.access_record %l "l_suppkey" : record<"l_orderkey": i32, "l_partkey": i32, "l_suppkey": i32, "l_linenumber": i32, "l_quantity": f64, "l_extendedprice": f64, "l_discount": f64, "l_tax": f64, "l_returnflag": memref<1xi8>, "l_linestatus": memref<1xi8>, "l_shipdate": i32, "l_commitdate": i32, "l_receiptdate": i32, "l_shipinstruct": memref<25xi8>, "l_shipmode": memref<10xi8>, "l_comment": memref<44xi8>> -> i32
    %recnode_148 = sdql.create_record {fields = ["_", "_"]} %fieldnode_146, %fieldnode_147 : i32, i32 -> record<"_": i32, "_": i32>
    %uniq_145 = sdql.unique %recnode_148 : record<"_" : i32, "_" : i32> -> record<"_" : i32, "_" : i32>
    %constd_150 = "arith.constant"() <{value = 0.5  : f64}> : () -> f64
    %fieldnode_151 = sdql.access_record %l "l_quantity" : record<"l_orderkey": i32, "l_partkey": i32, "l_suppkey": i32, "l_linenumber": i32, "l_quantity": f64, "l_extendedprice": f64, "l_discount": f64, "l_tax": f64, "l_returnflag": memref<1xi8>, "l_linestatus": memref<1xi8>, "l_shipdate": i32, "l_commitdate": i32, "l_receiptdate": i32, "l_shipinstruct": memref<25xi8>, "l_shipmode": memref<10xi8>, "l_comment": memref<44xi8>> -> f64
    %multf_149 = "arith.mulf"(%constd_150, %fieldnode_151) <{fastmath = #arith.fastmath<none>}> : (f64, f64) -> f64
    %dict_144 = sdql.create_dictionary %uniq_145, %multf_149 : record<"_" : i32, "_" : i32>, f64 -> dictionary<record<"_" : i32, "_" : i32>, f64>
    "scf.yield"(%dict_144) : (dictionary<record<"_" : i32, "_" : i32>, f64>) -> ()
  }, {
    %dict_152 = sdql.empty_dictionary : dictionary<record<"_" : i32, "_" : i32>, f64>
    "scf.yield"(%dict_152) : (dictionary<record<"_" : i32, "_" : i32>, f64>) -> ()
  }) : (i1) -> dictionary<record<"_" : i32, "_" : i32>, f64>
  sdql.yield %if_143 : dictionary<record<"_" : i32, "_" : i32>, f64>
}
%fieldnode_153 = sdql.access_record %partsupp "size" : record<"ps_partkey": dictionary<i32, i32>, "ps_suppkey": dictionary<i32, i32>, "ps_availqty": dictionary<i32, f64>, "ps_supplycost": dictionary<i32, f64>, "ps_comment": dictionary<i32, memref<199xi8>>, "size": i32> -> i32
%rangenode_154 = "func.call"(%fieldnode_153) <{callee = @range_builtin}> : (i32) -> dictionary<i32, i32>
%ps_h = sdql.sum %rangenode_154 : dictionary<i32, i32> -> dictionary<i32, record<"_" : i32>> {
^bb_155(%i: i32, %_: i32):
  %fieldnode_157 = sdql.access_record %partsupp "ps_partkey" : record<"ps_partkey": dictionary<i32, i32>, "ps_suppkey": dictionary<i32, i32>, "ps_availqty": dictionary<i32, f64>, "ps_supplycost": dictionary<i32, f64>, "ps_comment": dictionary<i32, memref<199xi8>>, "size": i32> -> dictionary<i32, i32>
  %get_156 = sdql.lookup_dictionary %fieldnode_157 [%i : i32] : dictionary<i32, i32> -> i32
  %fieldnode_159 = sdql.access_record %partsupp "ps_suppkey" : record<"ps_partkey": dictionary<i32, i32>, "ps_suppkey": dictionary<i32, i32>, "ps_availqty": dictionary<i32, f64>, "ps_supplycost": dictionary<i32, f64>, "ps_comment": dictionary<i32, memref<199xi8>>, "size": i32> -> dictionary<i32, i32>
  %get_158 = sdql.lookup_dictionary %fieldnode_159 [%i : i32] : dictionary<i32, i32> -> i32
  %fieldnode_161 = sdql.access_record %partsupp "ps_availqty" : record<"ps_partkey": dictionary<i32, i32>, "ps_suppkey": dictionary<i32, i32>, "ps_availqty": dictionary<i32, f64>, "ps_supplycost": dictionary<i32, f64>, "ps_comment": dictionary<i32, memref<199xi8>>, "size": i32> -> dictionary<i32, f64>
  %get_160 = sdql.lookup_dictionary %fieldnode_161 [%i : i32] : dictionary<i32, f64> -> f64
  %fieldnode_163 = sdql.access_record %partsupp "ps_supplycost" : record<"ps_partkey": dictionary<i32, i32>, "ps_suppkey": dictionary<i32, i32>, "ps_availqty": dictionary<i32, f64>, "ps_supplycost": dictionary<i32, f64>, "ps_comment": dictionary<i32, memref<199xi8>>, "size": i32> -> dictionary<i32, f64>
  %get_162 = sdql.lookup_dictionary %fieldnode_163 [%i : i32] : dictionary<i32, f64> -> f64
  %fieldnode_165 = sdql.access_record %partsupp "ps_comment" : record<"ps_partkey": dictionary<i32, i32>, "ps_suppkey": dictionary<i32, i32>, "ps_availqty": dictionary<i32, f64>, "ps_supplycost": dictionary<i32, f64>, "ps_comment": dictionary<i32, memref<199xi8>>, "size": i32> -> dictionary<i32, memref<199xi8>>
  %get_164 = sdql.lookup_dictionary %fieldnode_165 [%i : i32] : dictionary<i32, memref<199xi8>> -> memref<199xi8>
  %ps = sdql.create_record {fields = ["ps_partkey", "ps_suppkey", "ps_availqty", "ps_supplycost", "ps_comment"]} %get_156, %get_158, %get_160, %get_162, %get_164 : i32, i32, f64, f64, memref<199xi8> -> record<"ps_partkey": i32, "ps_suppkey": i32, "ps_availqty": f64, "ps_supplycost": f64, "ps_comment": memref<199xi8>>
  %fieldnode_166 = sdql.access_record %ps "ps_partkey" : record<"ps_partkey": i32, "ps_suppkey": i32, "ps_availqty": f64, "ps_supplycost": f64, "ps_comment": memref<199xi8>> -> i32
  %fieldnode_167 = sdql.access_record %ps "ps_suppkey" : record<"ps_partkey": i32, "ps_suppkey": i32, "ps_availqty": f64, "ps_supplycost": f64, "ps_comment": memref<199xi8>> -> i32
  %key = sdql.create_record {fields = ["_", "_"]} %fieldnode_166, %fieldnode_167 : i32, i32 -> record<"_": i32, "_": i32>
  %get_168 = sdql.lookup_dictionary %l_h [%key : record<"_" : i32, "_" : i32>] : dictionary<record<"_" : i32, "_" : i32>, f64> -> f64
  %dict_169 = sdql.empty_dictionary : f64
  %cmpf_170 = "arith.cmpf"(%get_168, %dict_169) <{fastmath = #arith.fastmath<none>, predicate = 6}> : (f64, f64) -> i1
  %if_171 = "scf.if"(%cmpf_170) ({
    %get_172 = sdql.lookup_dictionary %l_h [%key : record<"_" : i32, "_" : i32>] : dictionary<record<"_" : i32, "_" : i32>, f64> -> f64
    %fieldnode_173 = sdql.access_record %ps "ps_availqty" : record<"ps_partkey": i32, "ps_suppkey": i32, "ps_availqty": f64, "ps_supplycost": f64, "ps_comment": memref<199xi8>> -> f64
    %cmpf_174 = "arith.cmpf"(%get_172, %fieldnode_173) <{fastmath = #arith.fastmath<none>, predicate = 4}> : (f64, f64) -> i1
    "scf.yield"(%cmpf_174) : (i1) -> ()
  }, {
    %constv_175 = "arith.constant"() <{value = 0 : i1}> : () -> i1
    "scf.yield"(%constv_175) : (i1) -> ()
  }) : (i1) -> i1
  %if_176 = "scf.if"(%if_171) ({
    %fieldnode_179 = sdql.access_record %ps "ps_suppkey" : record<"ps_partkey": i32, "ps_suppkey": i32, "ps_availqty": f64, "ps_supplycost": f64, "ps_comment": memref<199xi8>> -> i32
    %uniq_178 = sdql.unique %fieldnode_179 : i32 -> i32
    %fieldnode_180 = sdql.access_record %ps "ps_suppkey" : record<"ps_partkey": i32, "ps_suppkey": i32, "ps_availqty": f64, "ps_supplycost": f64, "ps_comment": memref<199xi8>> -> i32
    %recnode_181 = sdql.create_record {fields = ["_"]} %fieldnode_180 : i32 -> record<"_": i32>
    %dict_177 = sdql.create_dictionary %uniq_178, %recnode_181 : i32, record<"_" : i32> -> dictionary<i32, record<"_" : i32>>
    "scf.yield"(%dict_177) : (dictionary<i32, record<"_" : i32>>) -> ()
  }, {
    %dict_182 = sdql.empty_dictionary : dictionary<i32, record<"_" : i32>>
    "scf.yield"(%dict_182) : (dictionary<i32, record<"_" : i32>>) -> ()
  }) : (i1) -> dictionary<i32, record<"_" : i32>>
  sdql.yield %if_176 : dictionary<i32, record<"_" : i32>>
}
%fieldnode_183 = sdql.access_record %supplier "size" : record<"s_suppkey": dictionary<i32, i32>, "s_name": dictionary<i32, memref<25xi8>>, "s_address": dictionary<i32, memref<40xi8>>, "s_nationkey": dictionary<i32, i32>, "s_phone": dictionary<i32, memref<15xi8>>, "s_acctbal": dictionary<i32, f64>, "s_comment": dictionary<i32, memref<101xi8>>, "size": i32> -> i32
%rangenode_184 = "func.call"(%fieldnode_183) <{callee = @range_builtin}> : (i32) -> dictionary<i32, i32>
%sum_185 = sdql.sum %rangenode_184 : dictionary<i32, i32> -> dictionary<record<"_" : memref<25xi8>, "_" : memref<40xi8>>, i32> {
^bb_186(%i: i32, %_: i32):
  %fieldnode_188 = sdql.access_record %supplier "s_suppkey" : record<"s_suppkey": dictionary<i32, i32>, "s_name": dictionary<i32, memref<25xi8>>, "s_address": dictionary<i32, memref<40xi8>>, "s_nationkey": dictionary<i32, i32>, "s_phone": dictionary<i32, memref<15xi8>>, "s_acctbal": dictionary<i32, f64>, "s_comment": dictionary<i32, memref<101xi8>>, "size": i32> -> dictionary<i32, i32>
  %get_187 = sdql.lookup_dictionary %fieldnode_188 [%i : i32] : dictionary<i32, i32> -> i32
  %fieldnode_190 = sdql.access_record %supplier "s_name" : record<"s_suppkey": dictionary<i32, i32>, "s_name": dictionary<i32, memref<25xi8>>, "s_address": dictionary<i32, memref<40xi8>>, "s_nationkey": dictionary<i32, i32>, "s_phone": dictionary<i32, memref<15xi8>>, "s_acctbal": dictionary<i32, f64>, "s_comment": dictionary<i32, memref<101xi8>>, "size": i32> -> dictionary<i32, memref<25xi8>>
  %get_189 = sdql.lookup_dictionary %fieldnode_190 [%i : i32] : dictionary<i32, memref<25xi8>> -> memref<25xi8>
  %fieldnode_192 = sdql.access_record %supplier "s_address" : record<"s_suppkey": dictionary<i32, i32>, "s_name": dictionary<i32, memref<25xi8>>, "s_address": dictionary<i32, memref<40xi8>>, "s_nationkey": dictionary<i32, i32>, "s_phone": dictionary<i32, memref<15xi8>>, "s_acctbal": dictionary<i32, f64>, "s_comment": dictionary<i32, memref<101xi8>>, "size": i32> -> dictionary<i32, memref<40xi8>>
  %get_191 = sdql.lookup_dictionary %fieldnode_192 [%i : i32] : dictionary<i32, memref<40xi8>> -> memref<40xi8>
  %fieldnode_194 = sdql.access_record %supplier "s_nationkey" : record<"s_suppkey": dictionary<i32, i32>, "s_name": dictionary<i32, memref<25xi8>>, "s_address": dictionary<i32, memref<40xi8>>, "s_nationkey": dictionary<i32, i32>, "s_phone": dictionary<i32, memref<15xi8>>, "s_acctbal": dictionary<i32, f64>, "s_comment": dictionary<i32, memref<101xi8>>, "size": i32> -> dictionary<i32, i32>
  %get_193 = sdql.lookup_dictionary %fieldnode_194 [%i : i32] : dictionary<i32, i32> -> i32
  %fieldnode_196 = sdql.access_record %supplier "s_phone" : record<"s_suppkey": dictionary<i32, i32>, "s_name": dictionary<i32, memref<25xi8>>, "s_address": dictionary<i32, memref<40xi8>>, "s_nationkey": dictionary<i32, i32>, "s_phone": dictionary<i32, memref<15xi8>>, "s_acctbal": dictionary<i32, f64>, "s_comment": dictionary<i32, memref<101xi8>>, "size": i32> -> dictionary<i32, memref<15xi8>>
  %get_195 = sdql.lookup_dictionary %fieldnode_196 [%i : i32] : dictionary<i32, memref<15xi8>> -> memref<15xi8>
  %fieldnode_198 = sdql.access_record %supplier "s_acctbal" : record<"s_suppkey": dictionary<i32, i32>, "s_name": dictionary<i32, memref<25xi8>>, "s_address": dictionary<i32, memref<40xi8>>, "s_nationkey": dictionary<i32, i32>, "s_phone": dictionary<i32, memref<15xi8>>, "s_acctbal": dictionary<i32, f64>, "s_comment": dictionary<i32, memref<101xi8>>, "size": i32> -> dictionary<i32, f64>
  %get_197 = sdql.lookup_dictionary %fieldnode_198 [%i : i32] : dictionary<i32, f64> -> f64
  %fieldnode_200 = sdql.access_record %supplier "s_comment" : record<"s_suppkey": dictionary<i32, i32>, "s_name": dictionary<i32, memref<25xi8>>, "s_address": dictionary<i32, memref<40xi8>>, "s_nationkey": dictionary<i32, i32>, "s_phone": dictionary<i32, memref<15xi8>>, "s_acctbal": dictionary<i32, f64>, "s_comment": dictionary<i32, memref<101xi8>>, "size": i32> -> dictionary<i32, memref<101xi8>>
  %get_199 = sdql.lookup_dictionary %fieldnode_200 [%i : i32] : dictionary<i32, memref<101xi8>> -> memref<101xi8>
  %s = sdql.create_record {fields = ["s_suppkey", "s_name", "s_address", "s_nationkey", "s_phone", "s_acctbal", "s_comment"]} %get_187, %get_189, %get_191, %get_193, %get_195, %get_197, %get_199 : i32, memref<25xi8>, memref<40xi8>, i32, memref<15xi8>, f64, memref<101xi8> -> record<"s_suppkey": i32, "s_name": memref<25xi8>, "s_address": memref<40xi8>, "s_nationkey": i32, "s_phone": memref<15xi8>, "s_acctbal": f64, "s_comment": memref<101xi8>>
  %fieldnode_204 = sdql.access_record %s "s_suppkey" : record<"s_suppkey": i32, "s_name": memref<25xi8>, "s_address": memref<40xi8>, "s_nationkey": i32, "s_phone": memref<15xi8>, "s_acctbal": f64, "s_comment": memref<101xi8>> -> i32
  %get_203 = sdql.lookup_dictionary %ps_h [%fieldnode_204 : i32] : dictionary<i32, record<"_" : i32>> -> record<"_" : i32>
  %dict_205 = sdql.empty_dictionary : record<"_" : i32>
  %cmp_206 = sdql.cmp %get_203, %dict_205 : record<"_" : i32>, record<"_" : i32> -> i1
  %zero_201 = "arith.constant"() <{value = 0 : i1}> : () -> i1
  %neg_202 = "arith.cmpi"(%cmp_206, %zero_201) <{predicate = 0}> : (i1, i1) -> i1
  %if_207 = "scf.if"(%neg_202) ({
    %fieldnode_210 = sdql.access_record %s "s_name" : record<"s_suppkey": i32, "s_name": memref<25xi8>, "s_address": memref<40xi8>, "s_nationkey": i32, "s_phone": memref<15xi8>, "s_acctbal": f64, "s_comment": memref<101xi8>> -> memref<25xi8>
    %fieldnode_211 = sdql.access_record %s "s_address" : record<"s_suppkey": i32, "s_name": memref<25xi8>, "s_address": memref<40xi8>, "s_nationkey": i32, "s_phone": memref<15xi8>, "s_acctbal": f64, "s_comment": memref<101xi8>> -> memref<40xi8>
    %recnode_212 = sdql.create_record {fields = ["_", "_"]} %fieldnode_210, %fieldnode_211 : memref<25xi8>, memref<40xi8> -> record<"_": memref<25xi8>, "_": memref<40xi8>>
    %uniq_209 = sdql.unique %recnode_212 : record<"_" : memref<25xi8>, "_" : memref<40xi8>> -> record<"_" : memref<25xi8>, "_" : memref<40xi8>>
    %consti_213 = "arith.constant"() <{value = 1 : i32}> : () -> i32
    %dict_208 = sdql.create_dictionary %uniq_209, %consti_213 : record<"_" : memref<25xi8>, "_" : memref<40xi8>>, i32 -> dictionary<record<"_" : memref<25xi8>, "_" : memref<40xi8>>, i32>
    "scf.yield"(%dict_208) : (dictionary<record<"_" : memref<25xi8>, "_" : memref<40xi8>>, i32>) -> ()
  }, {
    %dict_214 = sdql.empty_dictionary : dictionary<record<"_" : memref<25xi8>, "_" : memref<40xi8>>, i32>
    "scf.yield"(%dict_214) : (dictionary<record<"_" : memref<25xi8>, "_" : memref<40xi8>>, i32>) -> ()
  }) : (i1) -> dictionary<record<"_" : memref<25xi8>, "_" : memref<40xi8>>, i32>
  sdql.yield %if_207 : dictionary<record<"_" : memref<25xi8>, "_" : memref<40xi8>>, i32>
}
