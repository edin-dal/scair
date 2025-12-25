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
%part = sdql.load "datasets/tpch/part.tbl" : record<"p_partkey" : dictionary<i32, i32>, "p_name" : dictionary<i32, memref<55xi8>>, "p_mfgr" : dictionary<i32, memref<25xi8>>, "p_brand" : dictionary<i32, memref<10xi8>>, "p_type" : dictionary<i32, memref<25xi8>>, "p_size" : dictionary<i32, i32>, "p_container" : dictionary<i32, memref<10xi8>>, "p_retailprice" : dictionary<i32, f64>, "p_comment" : dictionary<i32, memref<23xi8>>, "size" : i32>
%supplier = sdql.load "datasets/tpch/supplier.tbl" : record<"s_suppkey" : dictionary<i32, i32>, "s_name" : dictionary<i32, memref<25xi8>>, "s_address" : dictionary<i32, memref<40xi8>>, "s_nationkey" : dictionary<i32, i32>, "s_phone" : dictionary<i32, memref<15xi8>>, "s_acctbal" : dictionary<i32, f64>, "s_comment" : dictionary<i32, memref<101xi8>>, "size" : i32>
%partsupp = sdql.load "datasets/tpch/partsupp.tbl" : record<"ps_partkey" : dictionary<i32, i32>, "ps_suppkey" : dictionary<i32, i32>, "ps_availqty" : dictionary<i32, f64>, "ps_supplycost" : dictionary<i32, f64>, "ps_comment" : dictionary<i32, memref<199xi8>>, "size" : i32>
%nation = sdql.load "datasets/tpch/nation.tbl" : record<"n_nationkey" : dictionary<i32, i32>, "n_name" : dictionary<i32, memref<25xi8>>, "n_regionkey" : dictionary<i32, i32>, "n_comment" : dictionary<i32, memref<152xi8>>, "size" : i32>
%region = sdql.load "datasets/tpch/region.tbl" : record<"r_regionkey" : dictionary<i32, i32>, "r_name" : dictionary<i32, memref<25xi8>>, "r_comment" : dictionary<i32, memref<152xi8>>, "size" : i32>
// BRASS
%constds_0 = "arith.constant"() <{value = dense<[66, 82, 65, 83, 83]> : memref<?xi8>}> : () -> memref<?xi8>
%consti_1 = "arith.constant"() <{value = 6 : i32}> : () -> i32
%brass = sdql.external "ConstantString", %constds_0, %consti_1 : memref<?xi8>, i32 -> memref<5xi8>
// EUROPE
%constds_2 = "arith.constant"() <{value = dense<[69, 85, 82, 79, 80, 69]> : memref<?xi8>}> : () -> memref<?xi8>
%consti_3 = "arith.constant"() <{value = 7 : i32}> : () -> i32
%europe = sdql.external "ConstantString", %constds_2, %consti_3 : memref<?xi8>, i32 -> memref<6xi8>
%fieldnode_4 = sdql.access_record %region "size" : record<"r_regionkey": dictionary<i32, i32>, "r_name": dictionary<i32, memref<25xi8>>, "r_comment": dictionary<i32, memref<152xi8>>, "size": i32> -> i32
%rangenode_5 = "func.call"(%fieldnode_4) <{callee = @range_builtin}> : (i32) -> dictionary<i32, i32>
%l_h = sdql.sum %rangenode_5 : dictionary<i32, i32> -> dictionary<i32, record<"_" : i32>> {
^bb_6(%i: i32, %_: i32):
  %fieldnode_8 = sdql.access_record %region "r_regionkey" : record<"r_regionkey": dictionary<i32, i32>, "r_name": dictionary<i32, memref<25xi8>>, "r_comment": dictionary<i32, memref<152xi8>>, "size": i32> -> dictionary<i32, i32>
  %get_7 = sdql.lookup_dictionary %fieldnode_8 [%i : i32] : dictionary<i32, i32> -> i32
  %fieldnode_10 = sdql.access_record %region "r_name" : record<"r_regionkey": dictionary<i32, i32>, "r_name": dictionary<i32, memref<25xi8>>, "r_comment": dictionary<i32, memref<152xi8>>, "size": i32> -> dictionary<i32, memref<25xi8>>
  %get_9 = sdql.lookup_dictionary %fieldnode_10 [%i : i32] : dictionary<i32, memref<25xi8>> -> memref<25xi8>
  %fieldnode_12 = sdql.access_record %region "r_comment" : record<"r_regionkey": dictionary<i32, i32>, "r_name": dictionary<i32, memref<25xi8>>, "r_comment": dictionary<i32, memref<152xi8>>, "size": i32> -> dictionary<i32, memref<152xi8>>
  %get_11 = sdql.lookup_dictionary %fieldnode_12 [%i : i32] : dictionary<i32, memref<152xi8>> -> memref<152xi8>
  %r = sdql.create_record {fields = ["r_regionkey", "r_name", "r_comment"]} %get_7, %get_9, %get_11 : i32, memref<25xi8>, memref<152xi8> -> record<"r_regionkey": i32, "r_name": memref<25xi8>, "r_comment": memref<152xi8>>
  %fieldnode_13 = sdql.access_record %r "r_name" : record<"r_regionkey": i32, "r_name": memref<25xi8>, "r_comment": memref<152xi8>> -> memref<25xi8>
  %cmp_14 = sdql.cmp %fieldnode_13, %europe : memref<25xi8>, memref<6xi8> -> i1
  %if_15 = "scf.if"(%cmp_14) ({
    %fieldnode_18 = sdql.access_record %r "r_regionkey" : record<"r_regionkey": i32, "r_name": memref<25xi8>, "r_comment": memref<152xi8>> -> i32
    %uniq_17 = sdql.unique %fieldnode_18 : i32 -> i32
    %fieldnode_19 = sdql.access_record %r "r_regionkey" : record<"r_regionkey": i32, "r_name": memref<25xi8>, "r_comment": memref<152xi8>> -> i32
    %recnode_20 = sdql.create_record {fields = ["_"]} %fieldnode_19 : i32 -> record<"_": i32>
    %dict_16 = sdql.create_dictionary %uniq_17, %recnode_20 : i32, record<"_" : i32> -> dictionary<i32, record<"_" : i32>>
    "scf.yield"(%dict_16) : (dictionary<i32, record<"_" : i32>>) -> ()
  }, {
    %dict_21 = sdql.empty_dictionary : dictionary<i32, record<"_" : i32>>
    "scf.yield"(%dict_21) : (dictionary<i32, record<"_" : i32>>) -> ()
  }) : (i1) -> dictionary<i32, record<"_" : i32>>
  sdql.yield %if_15 : dictionary<i32, record<"_" : i32>>
}
%fieldnode_22 = sdql.access_record %nation "size" : record<"n_nationkey": dictionary<i32, i32>, "n_name": dictionary<i32, memref<25xi8>>, "n_regionkey": dictionary<i32, i32>, "n_comment": dictionary<i32, memref<152xi8>>, "size": i32> -> i32
%rangenode_23 = "func.call"(%fieldnode_22) <{callee = @range_builtin}> : (i32) -> dictionary<i32, i32>
%n_h = sdql.sum %rangenode_23 : dictionary<i32, i32> -> dictionary<i32, memref<25xi8>> {
^bb_24(%i: i32, %_: i32):
  %fieldnode_26 = sdql.access_record %nation "n_nationkey" : record<"n_nationkey": dictionary<i32, i32>, "n_name": dictionary<i32, memref<25xi8>>, "n_regionkey": dictionary<i32, i32>, "n_comment": dictionary<i32, memref<152xi8>>, "size": i32> -> dictionary<i32, i32>
  %get_25 = sdql.lookup_dictionary %fieldnode_26 [%i : i32] : dictionary<i32, i32> -> i32
  %fieldnode_28 = sdql.access_record %nation "n_name" : record<"n_nationkey": dictionary<i32, i32>, "n_name": dictionary<i32, memref<25xi8>>, "n_regionkey": dictionary<i32, i32>, "n_comment": dictionary<i32, memref<152xi8>>, "size": i32> -> dictionary<i32, memref<25xi8>>
  %get_27 = sdql.lookup_dictionary %fieldnode_28 [%i : i32] : dictionary<i32, memref<25xi8>> -> memref<25xi8>
  %fieldnode_30 = sdql.access_record %nation "n_regionkey" : record<"n_nationkey": dictionary<i32, i32>, "n_name": dictionary<i32, memref<25xi8>>, "n_regionkey": dictionary<i32, i32>, "n_comment": dictionary<i32, memref<152xi8>>, "size": i32> -> dictionary<i32, i32>
  %get_29 = sdql.lookup_dictionary %fieldnode_30 [%i : i32] : dictionary<i32, i32> -> i32
  %fieldnode_32 = sdql.access_record %nation "n_comment" : record<"n_nationkey": dictionary<i32, i32>, "n_name": dictionary<i32, memref<25xi8>>, "n_regionkey": dictionary<i32, i32>, "n_comment": dictionary<i32, memref<152xi8>>, "size": i32> -> dictionary<i32, memref<152xi8>>
  %get_31 = sdql.lookup_dictionary %fieldnode_32 [%i : i32] : dictionary<i32, memref<152xi8>> -> memref<152xi8>
  %n = sdql.create_record {fields = ["n_nationkey", "n_name", "n_regionkey", "n_comment"]} %get_25, %get_27, %get_29, %get_31 : i32, memref<25xi8>, i32, memref<152xi8> -> record<"n_nationkey": i32, "n_name": memref<25xi8>, "n_regionkey": i32, "n_comment": memref<152xi8>>
  %fieldnode_36 = sdql.access_record %n "n_regionkey" : record<"n_nationkey": i32, "n_name": memref<25xi8>, "n_regionkey": i32, "n_comment": memref<152xi8>> -> i32
  %get_35 = sdql.lookup_dictionary %l_h [%fieldnode_36 : i32] : dictionary<i32, record<"_" : i32>> -> record<"_" : i32>
  %dict_37 = sdql.empty_dictionary : record<"_" : i32>
  %cmp_38 = sdql.cmp %get_35, %dict_37 : record<"_" : i32>, record<"_" : i32> -> i1
  %zero_33 = "arith.constant"() <{value = 0 : i1}> : () -> i1
  %neg_34 = "arith.cmpi"(%cmp_38, %zero_33) <{predicate = 0}> : (i1, i1) -> i1
  %if_39 = "scf.if"(%neg_34) ({
    %fieldnode_42 = sdql.access_record %n "n_nationkey" : record<"n_nationkey": i32, "n_name": memref<25xi8>, "n_regionkey": i32, "n_comment": memref<152xi8>> -> i32
    %uniq_41 = sdql.unique %fieldnode_42 : i32 -> i32
    %fieldnode_43 = sdql.access_record %n "n_name" : record<"n_nationkey": i32, "n_name": memref<25xi8>, "n_regionkey": i32, "n_comment": memref<152xi8>> -> memref<25xi8>
    %dict_40 = sdql.create_dictionary %uniq_41, %fieldnode_43 : i32, memref<25xi8> -> dictionary<i32, memref<25xi8>>
    "scf.yield"(%dict_40) : (dictionary<i32, memref<25xi8>>) -> ()
  }, {
    %dict_44 = sdql.empty_dictionary : dictionary<i32, memref<25xi8>>
    "scf.yield"(%dict_44) : (dictionary<i32, memref<25xi8>>) -> ()
  }) : (i1) -> dictionary<i32, memref<25xi8>>
  sdql.yield %if_39 : dictionary<i32, memref<25xi8>>
}
%fieldnode_45 = sdql.access_record %supplier "size" : record<"s_suppkey": dictionary<i32, i32>, "s_name": dictionary<i32, memref<25xi8>>, "s_address": dictionary<i32, memref<40xi8>>, "s_nationkey": dictionary<i32, i32>, "s_phone": dictionary<i32, memref<15xi8>>, "s_acctbal": dictionary<i32, f64>, "s_comment": dictionary<i32, memref<101xi8>>, "size": i32> -> i32
%rangenode_46 = "func.call"(%fieldnode_45) <{callee = @range_builtin}> : (i32) -> dictionary<i32, i32>
%s_h = sdql.sum %rangenode_46 : dictionary<i32, i32> -> dictionary<i32, record<"_" : f64, "_" : memref<25xi8>, "_" : memref<25xi8>, "_" : memref<40xi8>, "_" : memref<15xi8>, "_" : memref<101xi8>>> {
^bb_47(%i: i32, %_: i32):
  %fieldnode_49 = sdql.access_record %supplier "s_suppkey" : record<"s_suppkey": dictionary<i32, i32>, "s_name": dictionary<i32, memref<25xi8>>, "s_address": dictionary<i32, memref<40xi8>>, "s_nationkey": dictionary<i32, i32>, "s_phone": dictionary<i32, memref<15xi8>>, "s_acctbal": dictionary<i32, f64>, "s_comment": dictionary<i32, memref<101xi8>>, "size": i32> -> dictionary<i32, i32>
  %get_48 = sdql.lookup_dictionary %fieldnode_49 [%i : i32] : dictionary<i32, i32> -> i32
  %fieldnode_51 = sdql.access_record %supplier "s_name" : record<"s_suppkey": dictionary<i32, i32>, "s_name": dictionary<i32, memref<25xi8>>, "s_address": dictionary<i32, memref<40xi8>>, "s_nationkey": dictionary<i32, i32>, "s_phone": dictionary<i32, memref<15xi8>>, "s_acctbal": dictionary<i32, f64>, "s_comment": dictionary<i32, memref<101xi8>>, "size": i32> -> dictionary<i32, memref<25xi8>>
  %get_50 = sdql.lookup_dictionary %fieldnode_51 [%i : i32] : dictionary<i32, memref<25xi8>> -> memref<25xi8>
  %fieldnode_53 = sdql.access_record %supplier "s_address" : record<"s_suppkey": dictionary<i32, i32>, "s_name": dictionary<i32, memref<25xi8>>, "s_address": dictionary<i32, memref<40xi8>>, "s_nationkey": dictionary<i32, i32>, "s_phone": dictionary<i32, memref<15xi8>>, "s_acctbal": dictionary<i32, f64>, "s_comment": dictionary<i32, memref<101xi8>>, "size": i32> -> dictionary<i32, memref<40xi8>>
  %get_52 = sdql.lookup_dictionary %fieldnode_53 [%i : i32] : dictionary<i32, memref<40xi8>> -> memref<40xi8>
  %fieldnode_55 = sdql.access_record %supplier "s_nationkey" : record<"s_suppkey": dictionary<i32, i32>, "s_name": dictionary<i32, memref<25xi8>>, "s_address": dictionary<i32, memref<40xi8>>, "s_nationkey": dictionary<i32, i32>, "s_phone": dictionary<i32, memref<15xi8>>, "s_acctbal": dictionary<i32, f64>, "s_comment": dictionary<i32, memref<101xi8>>, "size": i32> -> dictionary<i32, i32>
  %get_54 = sdql.lookup_dictionary %fieldnode_55 [%i : i32] : dictionary<i32, i32> -> i32
  %fieldnode_57 = sdql.access_record %supplier "s_phone" : record<"s_suppkey": dictionary<i32, i32>, "s_name": dictionary<i32, memref<25xi8>>, "s_address": dictionary<i32, memref<40xi8>>, "s_nationkey": dictionary<i32, i32>, "s_phone": dictionary<i32, memref<15xi8>>, "s_acctbal": dictionary<i32, f64>, "s_comment": dictionary<i32, memref<101xi8>>, "size": i32> -> dictionary<i32, memref<15xi8>>
  %get_56 = sdql.lookup_dictionary %fieldnode_57 [%i : i32] : dictionary<i32, memref<15xi8>> -> memref<15xi8>
  %fieldnode_59 = sdql.access_record %supplier "s_acctbal" : record<"s_suppkey": dictionary<i32, i32>, "s_name": dictionary<i32, memref<25xi8>>, "s_address": dictionary<i32, memref<40xi8>>, "s_nationkey": dictionary<i32, i32>, "s_phone": dictionary<i32, memref<15xi8>>, "s_acctbal": dictionary<i32, f64>, "s_comment": dictionary<i32, memref<101xi8>>, "size": i32> -> dictionary<i32, f64>
  %get_58 = sdql.lookup_dictionary %fieldnode_59 [%i : i32] : dictionary<i32, f64> -> f64
  %fieldnode_61 = sdql.access_record %supplier "s_comment" : record<"s_suppkey": dictionary<i32, i32>, "s_name": dictionary<i32, memref<25xi8>>, "s_address": dictionary<i32, memref<40xi8>>, "s_nationkey": dictionary<i32, i32>, "s_phone": dictionary<i32, memref<15xi8>>, "s_acctbal": dictionary<i32, f64>, "s_comment": dictionary<i32, memref<101xi8>>, "size": i32> -> dictionary<i32, memref<101xi8>>
  %get_60 = sdql.lookup_dictionary %fieldnode_61 [%i : i32] : dictionary<i32, memref<101xi8>> -> memref<101xi8>
  %s = sdql.create_record {fields = ["s_suppkey", "s_name", "s_address", "s_nationkey", "s_phone", "s_acctbal", "s_comment"]} %get_48, %get_50, %get_52, %get_54, %get_56, %get_58, %get_60 : i32, memref<25xi8>, memref<40xi8>, i32, memref<15xi8>, f64, memref<101xi8> -> record<"s_suppkey": i32, "s_name": memref<25xi8>, "s_address": memref<40xi8>, "s_nationkey": i32, "s_phone": memref<15xi8>, "s_acctbal": f64, "s_comment": memref<101xi8>>
  %fieldnode_65 = sdql.access_record %s "s_nationkey" : record<"s_suppkey": i32, "s_name": memref<25xi8>, "s_address": memref<40xi8>, "s_nationkey": i32, "s_phone": memref<15xi8>, "s_acctbal": f64, "s_comment": memref<101xi8>> -> i32
  %get_64 = sdql.lookup_dictionary %n_h [%fieldnode_65 : i32] : dictionary<i32, memref<25xi8>> -> memref<25xi8>
  %dict_66 = sdql.empty_dictionary : memref<25xi8>
  %cmp_67 = sdql.cmp %get_64, %dict_66 : memref<25xi8>, memref<25xi8> -> i1
  %zero_62 = "arith.constant"() <{value = 0 : i1}> : () -> i1
  %neg_63 = "arith.cmpi"(%cmp_67, %zero_62) <{predicate = 0}> : (i1, i1) -> i1
  %if_68 = "scf.if"(%neg_63) ({
    %fieldnode_71 = sdql.access_record %s "s_suppkey" : record<"s_suppkey": i32, "s_name": memref<25xi8>, "s_address": memref<40xi8>, "s_nationkey": i32, "s_phone": memref<15xi8>, "s_acctbal": f64, "s_comment": memref<101xi8>> -> i32
    %uniq_70 = sdql.unique %fieldnode_71 : i32 -> i32
    %fieldnode_72 = sdql.access_record %s "s_acctbal" : record<"s_suppkey": i32, "s_name": memref<25xi8>, "s_address": memref<40xi8>, "s_nationkey": i32, "s_phone": memref<15xi8>, "s_acctbal": f64, "s_comment": memref<101xi8>> -> f64
    %fieldnode_73 = sdql.access_record %s "s_name" : record<"s_suppkey": i32, "s_name": memref<25xi8>, "s_address": memref<40xi8>, "s_nationkey": i32, "s_phone": memref<15xi8>, "s_acctbal": f64, "s_comment": memref<101xi8>> -> memref<25xi8>
    %fieldnode_75 = sdql.access_record %s "s_nationkey" : record<"s_suppkey": i32, "s_name": memref<25xi8>, "s_address": memref<40xi8>, "s_nationkey": i32, "s_phone": memref<15xi8>, "s_acctbal": f64, "s_comment": memref<101xi8>> -> i32
    %get_74 = sdql.lookup_dictionary %n_h [%fieldnode_75 : i32] : dictionary<i32, memref<25xi8>> -> memref<25xi8>
    %fieldnode_76 = sdql.access_record %s "s_address" : record<"s_suppkey": i32, "s_name": memref<25xi8>, "s_address": memref<40xi8>, "s_nationkey": i32, "s_phone": memref<15xi8>, "s_acctbal": f64, "s_comment": memref<101xi8>> -> memref<40xi8>
    %fieldnode_77 = sdql.access_record %s "s_phone" : record<"s_suppkey": i32, "s_name": memref<25xi8>, "s_address": memref<40xi8>, "s_nationkey": i32, "s_phone": memref<15xi8>, "s_acctbal": f64, "s_comment": memref<101xi8>> -> memref<15xi8>
    %fieldnode_78 = sdql.access_record %s "s_comment" : record<"s_suppkey": i32, "s_name": memref<25xi8>, "s_address": memref<40xi8>, "s_nationkey": i32, "s_phone": memref<15xi8>, "s_acctbal": f64, "s_comment": memref<101xi8>> -> memref<101xi8>
    %recnode_79 = sdql.create_record {fields = ["_", "_", "_", "_", "_", "_"]} %fieldnode_72, %fieldnode_73, %get_74, %fieldnode_76, %fieldnode_77, %fieldnode_78 : f64, memref<25xi8>, memref<25xi8>, memref<40xi8>, memref<15xi8>, memref<101xi8> -> record<"_": f64, "_": memref<25xi8>, "_": memref<25xi8>, "_": memref<40xi8>, "_": memref<15xi8>, "_": memref<101xi8>>
    %dict_69 = sdql.create_dictionary %uniq_70, %recnode_79 : i32, record<"_" : f64, "_" : memref<25xi8>, "_" : memref<25xi8>, "_" : memref<40xi8>, "_" : memref<15xi8>, "_" : memref<101xi8>> -> dictionary<i32, record<"_" : f64, "_" : memref<25xi8>, "_" : memref<25xi8>, "_" : memref<40xi8>, "_" : memref<15xi8>, "_" : memref<101xi8>>>
    "scf.yield"(%dict_69) : (dictionary<i32, record<"_" : f64, "_" : memref<25xi8>, "_" : memref<25xi8>, "_" : memref<40xi8>, "_" : memref<15xi8>, "_" : memref<101xi8>>>) -> ()
  }, {
    %dict_80 = sdql.empty_dictionary : dictionary<i32, record<"_" : f64, "_" : memref<25xi8>, "_" : memref<25xi8>, "_" : memref<40xi8>, "_" : memref<15xi8>, "_" : memref<101xi8>>>
    "scf.yield"(%dict_80) : (dictionary<i32, record<"_" : f64, "_" : memref<25xi8>, "_" : memref<25xi8>, "_" : memref<40xi8>, "_" : memref<15xi8>, "_" : memref<101xi8>>>) -> ()
  }) : (i1) -> dictionary<i32, record<"_" : f64, "_" : memref<25xi8>, "_" : memref<25xi8>, "_" : memref<40xi8>, "_" : memref<15xi8>, "_" : memref<101xi8>>>
  sdql.yield %if_68 : dictionary<i32, record<"_" : f64, "_" : memref<25xi8>, "_" : memref<25xi8>, "_" : memref<40xi8>, "_" : memref<15xi8>, "_" : memref<101xi8>>>
}
%fieldnode_81 = sdql.access_record %part "size" : record<"p_partkey": dictionary<i32, i32>, "p_name": dictionary<i32, memref<55xi8>>, "p_mfgr": dictionary<i32, memref<25xi8>>, "p_brand": dictionary<i32, memref<10xi8>>, "p_type": dictionary<i32, memref<25xi8>>, "p_size": dictionary<i32, i32>, "p_container": dictionary<i32, memref<10xi8>>, "p_retailprice": dictionary<i32, f64>, "p_comment": dictionary<i32, memref<23xi8>>, "size": i32> -> i32
%rangenode_82 = "func.call"(%fieldnode_81) <{callee = @range_builtin}> : (i32) -> dictionary<i32, i32>
%p_h = sdql.sum %rangenode_82 : dictionary<i32, i32> -> dictionary<i32, record<"_" : memref<25xi8>>> {
^bb_83(%i: i32, %_: i32):
  %fieldnode_85 = sdql.access_record %part "p_partkey" : record<"p_partkey": dictionary<i32, i32>, "p_name": dictionary<i32, memref<55xi8>>, "p_mfgr": dictionary<i32, memref<25xi8>>, "p_brand": dictionary<i32, memref<10xi8>>, "p_type": dictionary<i32, memref<25xi8>>, "p_size": dictionary<i32, i32>, "p_container": dictionary<i32, memref<10xi8>>, "p_retailprice": dictionary<i32, f64>, "p_comment": dictionary<i32, memref<23xi8>>, "size": i32> -> dictionary<i32, i32>
  %get_84 = sdql.lookup_dictionary %fieldnode_85 [%i : i32] : dictionary<i32, i32> -> i32
  %fieldnode_87 = sdql.access_record %part "p_name" : record<"p_partkey": dictionary<i32, i32>, "p_name": dictionary<i32, memref<55xi8>>, "p_mfgr": dictionary<i32, memref<25xi8>>, "p_brand": dictionary<i32, memref<10xi8>>, "p_type": dictionary<i32, memref<25xi8>>, "p_size": dictionary<i32, i32>, "p_container": dictionary<i32, memref<10xi8>>, "p_retailprice": dictionary<i32, f64>, "p_comment": dictionary<i32, memref<23xi8>>, "size": i32> -> dictionary<i32, memref<55xi8>>
  %get_86 = sdql.lookup_dictionary %fieldnode_87 [%i : i32] : dictionary<i32, memref<55xi8>> -> memref<55xi8>
  %fieldnode_89 = sdql.access_record %part "p_mfgr" : record<"p_partkey": dictionary<i32, i32>, "p_name": dictionary<i32, memref<55xi8>>, "p_mfgr": dictionary<i32, memref<25xi8>>, "p_brand": dictionary<i32, memref<10xi8>>, "p_type": dictionary<i32, memref<25xi8>>, "p_size": dictionary<i32, i32>, "p_container": dictionary<i32, memref<10xi8>>, "p_retailprice": dictionary<i32, f64>, "p_comment": dictionary<i32, memref<23xi8>>, "size": i32> -> dictionary<i32, memref<25xi8>>
  %get_88 = sdql.lookup_dictionary %fieldnode_89 [%i : i32] : dictionary<i32, memref<25xi8>> -> memref<25xi8>
  %fieldnode_91 = sdql.access_record %part "p_brand" : record<"p_partkey": dictionary<i32, i32>, "p_name": dictionary<i32, memref<55xi8>>, "p_mfgr": dictionary<i32, memref<25xi8>>, "p_brand": dictionary<i32, memref<10xi8>>, "p_type": dictionary<i32, memref<25xi8>>, "p_size": dictionary<i32, i32>, "p_container": dictionary<i32, memref<10xi8>>, "p_retailprice": dictionary<i32, f64>, "p_comment": dictionary<i32, memref<23xi8>>, "size": i32> -> dictionary<i32, memref<10xi8>>
  %get_90 = sdql.lookup_dictionary %fieldnode_91 [%i : i32] : dictionary<i32, memref<10xi8>> -> memref<10xi8>
  %fieldnode_93 = sdql.access_record %part "p_type" : record<"p_partkey": dictionary<i32, i32>, "p_name": dictionary<i32, memref<55xi8>>, "p_mfgr": dictionary<i32, memref<25xi8>>, "p_brand": dictionary<i32, memref<10xi8>>, "p_type": dictionary<i32, memref<25xi8>>, "p_size": dictionary<i32, i32>, "p_container": dictionary<i32, memref<10xi8>>, "p_retailprice": dictionary<i32, f64>, "p_comment": dictionary<i32, memref<23xi8>>, "size": i32> -> dictionary<i32, memref<25xi8>>
  %get_92 = sdql.lookup_dictionary %fieldnode_93 [%i : i32] : dictionary<i32, memref<25xi8>> -> memref<25xi8>
  %fieldnode_95 = sdql.access_record %part "p_size" : record<"p_partkey": dictionary<i32, i32>, "p_name": dictionary<i32, memref<55xi8>>, "p_mfgr": dictionary<i32, memref<25xi8>>, "p_brand": dictionary<i32, memref<10xi8>>, "p_type": dictionary<i32, memref<25xi8>>, "p_size": dictionary<i32, i32>, "p_container": dictionary<i32, memref<10xi8>>, "p_retailprice": dictionary<i32, f64>, "p_comment": dictionary<i32, memref<23xi8>>, "size": i32> -> dictionary<i32, i32>
  %get_94 = sdql.lookup_dictionary %fieldnode_95 [%i : i32] : dictionary<i32, i32> -> i32
  %fieldnode_97 = sdql.access_record %part "p_container" : record<"p_partkey": dictionary<i32, i32>, "p_name": dictionary<i32, memref<55xi8>>, "p_mfgr": dictionary<i32, memref<25xi8>>, "p_brand": dictionary<i32, memref<10xi8>>, "p_type": dictionary<i32, memref<25xi8>>, "p_size": dictionary<i32, i32>, "p_container": dictionary<i32, memref<10xi8>>, "p_retailprice": dictionary<i32, f64>, "p_comment": dictionary<i32, memref<23xi8>>, "size": i32> -> dictionary<i32, memref<10xi8>>
  %get_96 = sdql.lookup_dictionary %fieldnode_97 [%i : i32] : dictionary<i32, memref<10xi8>> -> memref<10xi8>
  %fieldnode_99 = sdql.access_record %part "p_retailprice" : record<"p_partkey": dictionary<i32, i32>, "p_name": dictionary<i32, memref<55xi8>>, "p_mfgr": dictionary<i32, memref<25xi8>>, "p_brand": dictionary<i32, memref<10xi8>>, "p_type": dictionary<i32, memref<25xi8>>, "p_size": dictionary<i32, i32>, "p_container": dictionary<i32, memref<10xi8>>, "p_retailprice": dictionary<i32, f64>, "p_comment": dictionary<i32, memref<23xi8>>, "size": i32> -> dictionary<i32, f64>
  %get_98 = sdql.lookup_dictionary %fieldnode_99 [%i : i32] : dictionary<i32, f64> -> f64
  %fieldnode_101 = sdql.access_record %part "p_comment" : record<"p_partkey": dictionary<i32, i32>, "p_name": dictionary<i32, memref<55xi8>>, "p_mfgr": dictionary<i32, memref<25xi8>>, "p_brand": dictionary<i32, memref<10xi8>>, "p_type": dictionary<i32, memref<25xi8>>, "p_size": dictionary<i32, i32>, "p_container": dictionary<i32, memref<10xi8>>, "p_retailprice": dictionary<i32, f64>, "p_comment": dictionary<i32, memref<23xi8>>, "size": i32> -> dictionary<i32, memref<23xi8>>
  %get_100 = sdql.lookup_dictionary %fieldnode_101 [%i : i32] : dictionary<i32, memref<23xi8>> -> memref<23xi8>
  %p = sdql.create_record {fields = ["p_partkey", "p_name", "p_mfgr", "p_brand", "p_type", "p_size", "p_container", "p_retailprice", "p_comment"]} %get_84, %get_86, %get_88, %get_90, %get_92, %get_94, %get_96, %get_98, %get_100 : i32, memref<55xi8>, memref<25xi8>, memref<10xi8>, memref<25xi8>, i32, memref<10xi8>, f64, memref<23xi8> -> record<"p_partkey": i32, "p_name": memref<55xi8>, "p_mfgr": memref<25xi8>, "p_brand": memref<10xi8>, "p_type": memref<25xi8>, "p_size": i32, "p_container": memref<10xi8>, "p_retailprice": f64, "p_comment": memref<23xi8>>
  %fieldnode_102 = sdql.access_record %p "p_size" : record<"p_partkey": i32, "p_name": memref<55xi8>, "p_mfgr": memref<25xi8>, "p_brand": memref<10xi8>, "p_type": memref<25xi8>, "p_size": i32, "p_container": memref<10xi8>, "p_retailprice": f64, "p_comment": memref<23xi8>> -> i32
  %consti_103 = "arith.constant"() <{value = 15 : i32}> : () -> i32
  %cmpi_104 = "arith.cmpi"(%fieldnode_102, %consti_103) <{predicate = 0}> : (i32, i32) -> i1
  %if_105 = "scf.if"(%cmpi_104) ({
    %fieldnode_106 = sdql.access_record %p "p_type" : record<"p_partkey": i32, "p_name": memref<55xi8>, "p_mfgr": memref<25xi8>, "p_brand": memref<10xi8>, "p_type": memref<25xi8>, "p_size": i32, "p_container": memref<10xi8>, "p_retailprice": f64, "p_comment": memref<23xi8>> -> memref<25xi8>
    %external_107 = sdql.external "StrEndsWith", %fieldnode_106, %brass : memref<25xi8>, memref<5xi8> -> i1
    "scf.yield"(%external_107) : (i1) -> ()
  }, {
    %constv_108 = "arith.constant"() <{value = 0 : i1}> : () -> i1
    "scf.yield"(%constv_108) : (i1) -> ()
  }) : (i1) -> i1
  %if_109 = "scf.if"(%if_105) ({
    %fieldnode_112 = sdql.access_record %p "p_partkey" : record<"p_partkey": i32, "p_name": memref<55xi8>, "p_mfgr": memref<25xi8>, "p_brand": memref<10xi8>, "p_type": memref<25xi8>, "p_size": i32, "p_container": memref<10xi8>, "p_retailprice": f64, "p_comment": memref<23xi8>> -> i32
    %uniq_111 = sdql.unique %fieldnode_112 : i32 -> i32
    %fieldnode_113 = sdql.access_record %p "p_mfgr" : record<"p_partkey": i32, "p_name": memref<55xi8>, "p_mfgr": memref<25xi8>, "p_brand": memref<10xi8>, "p_type": memref<25xi8>, "p_size": i32, "p_container": memref<10xi8>, "p_retailprice": f64, "p_comment": memref<23xi8>> -> memref<25xi8>
    %recnode_114 = sdql.create_record {fields = ["_"]} %fieldnode_113 : memref<25xi8> -> record<"_": memref<25xi8>>
    %dict_110 = sdql.create_dictionary %uniq_111, %recnode_114 : i32, record<"_" : memref<25xi8>> -> dictionary<i32, record<"_" : memref<25xi8>>>
    "scf.yield"(%dict_110) : (dictionary<i32, record<"_" : memref<25xi8>>>) -> ()
  }, {
    %dict_115 = sdql.empty_dictionary : dictionary<i32, record<"_" : memref<25xi8>>>
    "scf.yield"(%dict_115) : (dictionary<i32, record<"_" : memref<25xi8>>>) -> ()
  }) : (i1) -> dictionary<i32, record<"_" : memref<25xi8>>>
  sdql.yield %if_109 : dictionary<i32, record<"_" : memref<25xi8>>>
}
%fieldnode_116 = sdql.access_record %partsupp "size" : record<"ps_partkey": dictionary<i32, i32>, "ps_suppkey": dictionary<i32, i32>, "ps_availqty": dictionary<i32, f64>, "ps_supplycost": dictionary<i32, f64>, "ps_comment": dictionary<i32, memref<199xi8>>, "size": i32> -> i32
%rangenode_117 = "func.call"(%fieldnode_116) <{callee = @range_builtin}> : (i32) -> dictionary<i32, i32>
%ps_h = sdql.sum %rangenode_117 : dictionary<i32, i32> -> dictionary<i32, f64> {
^bb_118(%i: i32, %_: i32):
  %fieldnode_120 = sdql.access_record %partsupp "ps_partkey" : record<"ps_partkey": dictionary<i32, i32>, "ps_suppkey": dictionary<i32, i32>, "ps_availqty": dictionary<i32, f64>, "ps_supplycost": dictionary<i32, f64>, "ps_comment": dictionary<i32, memref<199xi8>>, "size": i32> -> dictionary<i32, i32>
  %get_119 = sdql.lookup_dictionary %fieldnode_120 [%i : i32] : dictionary<i32, i32> -> i32
  %fieldnode_122 = sdql.access_record %partsupp "ps_suppkey" : record<"ps_partkey": dictionary<i32, i32>, "ps_suppkey": dictionary<i32, i32>, "ps_availqty": dictionary<i32, f64>, "ps_supplycost": dictionary<i32, f64>, "ps_comment": dictionary<i32, memref<199xi8>>, "size": i32> -> dictionary<i32, i32>
  %get_121 = sdql.lookup_dictionary %fieldnode_122 [%i : i32] : dictionary<i32, i32> -> i32
  %fieldnode_124 = sdql.access_record %partsupp "ps_availqty" : record<"ps_partkey": dictionary<i32, i32>, "ps_suppkey": dictionary<i32, i32>, "ps_availqty": dictionary<i32, f64>, "ps_supplycost": dictionary<i32, f64>, "ps_comment": dictionary<i32, memref<199xi8>>, "size": i32> -> dictionary<i32, f64>
  %get_123 = sdql.lookup_dictionary %fieldnode_124 [%i : i32] : dictionary<i32, f64> -> f64
  %fieldnode_126 = sdql.access_record %partsupp "ps_supplycost" : record<"ps_partkey": dictionary<i32, i32>, "ps_suppkey": dictionary<i32, i32>, "ps_availqty": dictionary<i32, f64>, "ps_supplycost": dictionary<i32, f64>, "ps_comment": dictionary<i32, memref<199xi8>>, "size": i32> -> dictionary<i32, f64>
  %get_125 = sdql.lookup_dictionary %fieldnode_126 [%i : i32] : dictionary<i32, f64> -> f64
  %fieldnode_128 = sdql.access_record %partsupp "ps_comment" : record<"ps_partkey": dictionary<i32, i32>, "ps_suppkey": dictionary<i32, i32>, "ps_availqty": dictionary<i32, f64>, "ps_supplycost": dictionary<i32, f64>, "ps_comment": dictionary<i32, memref<199xi8>>, "size": i32> -> dictionary<i32, memref<199xi8>>
  %get_127 = sdql.lookup_dictionary %fieldnode_128 [%i : i32] : dictionary<i32, memref<199xi8>> -> memref<199xi8>
  %ps = sdql.create_record {fields = ["ps_partkey", "ps_suppkey", "ps_availqty", "ps_supplycost", "ps_comment"]} %get_119, %get_121, %get_123, %get_125, %get_127 : i32, i32, f64, f64, memref<199xi8> -> record<"ps_partkey": i32, "ps_suppkey": i32, "ps_availqty": f64, "ps_supplycost": f64, "ps_comment": memref<199xi8>>
  %fieldnode_132 = sdql.access_record %ps "ps_partkey" : record<"ps_partkey": i32, "ps_suppkey": i32, "ps_availqty": f64, "ps_supplycost": f64, "ps_comment": memref<199xi8>> -> i32
  %get_131 = sdql.lookup_dictionary %p_h [%fieldnode_132 : i32] : dictionary<i32, record<"_" : memref<25xi8>>> -> record<"_" : memref<25xi8>>
  %dict_133 = sdql.empty_dictionary : record<"_" : memref<25xi8>>
  %cmp_134 = sdql.cmp %get_131, %dict_133 : record<"_" : memref<25xi8>>, record<"_" : memref<25xi8>> -> i1
  %zero_129 = "arith.constant"() <{value = 0 : i1}> : () -> i1
  %neg_130 = "arith.cmpi"(%cmp_134, %zero_129) <{predicate = 0}> : (i1, i1) -> i1
  %if_135 = "scf.if"(%neg_130) ({
    %fieldnode_139 = sdql.access_record %ps "ps_suppkey" : record<"ps_partkey": i32, "ps_suppkey": i32, "ps_availqty": f64, "ps_supplycost": f64, "ps_comment": memref<199xi8>> -> i32
    %get_138 = sdql.lookup_dictionary %s_h [%fieldnode_139 : i32] : dictionary<i32, record<"_" : f64, "_" : memref<25xi8>, "_" : memref<25xi8>, "_" : memref<40xi8>, "_" : memref<15xi8>, "_" : memref<101xi8>>> -> record<"_" : f64, "_" : memref<25xi8>, "_" : memref<25xi8>, "_" : memref<40xi8>, "_" : memref<15xi8>, "_" : memref<101xi8>>
    %dict_140 = sdql.empty_dictionary : record<"_" : f64, "_" : memref<25xi8>, "_" : memref<25xi8>, "_" : memref<40xi8>, "_" : memref<15xi8>, "_" : memref<101xi8>>
    %cmp_141 = sdql.cmp %get_138, %dict_140 : record<"_" : f64, "_" : memref<25xi8>, "_" : memref<25xi8>, "_" : memref<40xi8>, "_" : memref<15xi8>, "_" : memref<101xi8>>, record<"_" : f64, "_" : memref<25xi8>, "_" : memref<25xi8>, "_" : memref<40xi8>, "_" : memref<15xi8>, "_" : memref<101xi8>> -> i1
    %zero_136 = "arith.constant"() <{value = 0 : i1}> : () -> i1
    %neg_137 = "arith.cmpi"(%cmp_141, %zero_136) <{predicate = 0}> : (i1, i1) -> i1
    "scf.yield"(%neg_137) : (i1) -> ()
  }, {
    %constv_142 = "arith.constant"() <{value = 0 : i1}> : () -> i1
    "scf.yield"(%constv_142) : (i1) -> ()
  }) : (i1) -> i1
  %if_143 = "scf.if"(%if_135) ({
    %fieldnode_145 = sdql.access_record %ps "ps_partkey" : record<"ps_partkey": i32, "ps_suppkey": i32, "ps_availqty": f64, "ps_supplycost": f64, "ps_comment": memref<199xi8>> -> i32
    %fieldnode_146 = sdql.access_record %ps "ps_supplycost" : record<"ps_partkey": i32, "ps_suppkey": i32, "ps_availqty": f64, "ps_supplycost": f64, "ps_comment": memref<199xi8>> -> f64
    %dict_144 = sdql.create_dictionary %fieldnode_145, %fieldnode_146 : i32, f64 -> dictionary<i32, f64>
    "scf.yield"(%dict_144) : (dictionary<i32, f64>) -> ()
  }, {
    %dict_147 = sdql.empty_dictionary : dictionary<i32, f64>
    "scf.yield"(%dict_147) : (dictionary<i32, f64>) -> ()
  }) : (i1) -> dictionary<i32, f64>
  sdql.yield %if_143 : dictionary<i32, f64>
}
%fieldnode_148 = sdql.access_record %partsupp "size" : record<"ps_partkey": dictionary<i32, i32>, "ps_suppkey": dictionary<i32, i32>, "ps_availqty": dictionary<i32, f64>, "ps_supplycost": dictionary<i32, f64>, "ps_comment": dictionary<i32, memref<199xi8>>, "size": i32> -> i32
%rangenode_149 = "func.call"(%fieldnode_148) <{callee = @range_builtin}> : (i32) -> dictionary<i32, i32>
%sum_150 = sdql.sum %rangenode_149 : dictionary<i32, i32> -> dictionary<record<"_" : f64, "_" : memref<25xi8>, "_" : memref<25xi8>, "_" : i32, "_" : memref<25xi8>, "_" : memref<15xi8>, "_" : memref<40xi8>, "_" : memref<101xi8>>, i32> {
^bb_151(%i: i32, %_: i32):
  %fieldnode_153 = sdql.access_record %partsupp "ps_partkey" : record<"ps_partkey": dictionary<i32, i32>, "ps_suppkey": dictionary<i32, i32>, "ps_availqty": dictionary<i32, f64>, "ps_supplycost": dictionary<i32, f64>, "ps_comment": dictionary<i32, memref<199xi8>>, "size": i32> -> dictionary<i32, i32>
  %get_152 = sdql.lookup_dictionary %fieldnode_153 [%i : i32] : dictionary<i32, i32> -> i32
  %fieldnode_155 = sdql.access_record %partsupp "ps_suppkey" : record<"ps_partkey": dictionary<i32, i32>, "ps_suppkey": dictionary<i32, i32>, "ps_availqty": dictionary<i32, f64>, "ps_supplycost": dictionary<i32, f64>, "ps_comment": dictionary<i32, memref<199xi8>>, "size": i32> -> dictionary<i32, i32>
  %get_154 = sdql.lookup_dictionary %fieldnode_155 [%i : i32] : dictionary<i32, i32> -> i32
  %fieldnode_157 = sdql.access_record %partsupp "ps_availqty" : record<"ps_partkey": dictionary<i32, i32>, "ps_suppkey": dictionary<i32, i32>, "ps_availqty": dictionary<i32, f64>, "ps_supplycost": dictionary<i32, f64>, "ps_comment": dictionary<i32, memref<199xi8>>, "size": i32> -> dictionary<i32, f64>
  %get_156 = sdql.lookup_dictionary %fieldnode_157 [%i : i32] : dictionary<i32, f64> -> f64
  %fieldnode_159 = sdql.access_record %partsupp "ps_supplycost" : record<"ps_partkey": dictionary<i32, i32>, "ps_suppkey": dictionary<i32, i32>, "ps_availqty": dictionary<i32, f64>, "ps_supplycost": dictionary<i32, f64>, "ps_comment": dictionary<i32, memref<199xi8>>, "size": i32> -> dictionary<i32, f64>
  %get_158 = sdql.lookup_dictionary %fieldnode_159 [%i : i32] : dictionary<i32, f64> -> f64
  %fieldnode_161 = sdql.access_record %partsupp "ps_comment" : record<"ps_partkey": dictionary<i32, i32>, "ps_suppkey": dictionary<i32, i32>, "ps_availqty": dictionary<i32, f64>, "ps_supplycost": dictionary<i32, f64>, "ps_comment": dictionary<i32, memref<199xi8>>, "size": i32> -> dictionary<i32, memref<199xi8>>
  %get_160 = sdql.lookup_dictionary %fieldnode_161 [%i : i32] : dictionary<i32, memref<199xi8>> -> memref<199xi8>
  %ps = sdql.create_record {fields = ["ps_partkey", "ps_suppkey", "ps_availqty", "ps_supplycost", "ps_comment"]} %get_152, %get_154, %get_156, %get_158, %get_160 : i32, i32, f64, f64, memref<199xi8> -> record<"ps_partkey": i32, "ps_suppkey": i32, "ps_availqty": f64, "ps_supplycost": f64, "ps_comment": memref<199xi8>>
  %fieldnode_163 = sdql.access_record %ps "ps_partkey" : record<"ps_partkey": i32, "ps_suppkey": i32, "ps_availqty": f64, "ps_supplycost": f64, "ps_comment": memref<199xi8>> -> i32
  %get_162 = sdql.lookup_dictionary %ps_h [%fieldnode_163 : i32] : dictionary<i32, f64> -> f64
  %dict_164 = sdql.empty_dictionary : f64
  %cmpf_165 = "arith.cmpf"(%get_162, %dict_164) <{fastmath = #arith.fastmath<none>, predicate = 6}> : (f64, f64) -> i1
  %if_166 = "scf.if"(%cmpf_165) ({
    %fieldnode_168 = sdql.access_record %ps "ps_partkey" : record<"ps_partkey": i32, "ps_suppkey": i32, "ps_availqty": f64, "ps_supplycost": f64, "ps_comment": memref<199xi8>> -> i32
    %get_167 = sdql.lookup_dictionary %ps_h [%fieldnode_168 : i32] : dictionary<i32, f64> -> f64
    %fieldnode_169 = sdql.access_record %ps "ps_supplycost" : record<"ps_partkey": i32, "ps_suppkey": i32, "ps_availqty": f64, "ps_supplycost": f64, "ps_comment": memref<199xi8>> -> f64
    %cmpf_170 = "arith.cmpf"(%get_167, %fieldnode_169) <{fastmath = #arith.fastmath<none>, predicate = 1}> : (f64, f64) -> i1
    "scf.yield"(%cmpf_170) : (i1) -> ()
  }, {
    %constv_171 = "arith.constant"() <{value = 0 : i1}> : () -> i1
    "scf.yield"(%constv_171) : (i1) -> ()
  }) : (i1) -> i1
  %if_172 = "scf.if"(%if_166) ({
    %fieldnode_176 = sdql.access_record %ps "ps_suppkey" : record<"ps_partkey": i32, "ps_suppkey": i32, "ps_availqty": f64, "ps_supplycost": f64, "ps_comment": memref<199xi8>> -> i32
    %get_175 = sdql.lookup_dictionary %s_h [%fieldnode_176 : i32] : dictionary<i32, record<"_" : f64, "_" : memref<25xi8>, "_" : memref<25xi8>, "_" : memref<40xi8>, "_" : memref<15xi8>, "_" : memref<101xi8>>> -> record<"_" : f64, "_" : memref<25xi8>, "_" : memref<25xi8>, "_" : memref<40xi8>, "_" : memref<15xi8>, "_" : memref<101xi8>>
    %dict_177 = sdql.empty_dictionary : record<"_" : f64, "_" : memref<25xi8>, "_" : memref<25xi8>, "_" : memref<40xi8>, "_" : memref<15xi8>, "_" : memref<101xi8>>
    %cmp_178 = sdql.cmp %get_175, %dict_177 : record<"_" : f64, "_" : memref<25xi8>, "_" : memref<25xi8>, "_" : memref<40xi8>, "_" : memref<15xi8>, "_" : memref<101xi8>>, record<"_" : f64, "_" : memref<25xi8>, "_" : memref<25xi8>, "_" : memref<40xi8>, "_" : memref<15xi8>, "_" : memref<101xi8>> -> i1
    %zero_173 = "arith.constant"() <{value = 0 : i1}> : () -> i1
    %neg_174 = "arith.cmpi"(%cmp_178, %zero_173) <{predicate = 0}> : (i1, i1) -> i1
    "scf.yield"(%neg_174) : (i1) -> ()
  }, {
    %constv_179 = "arith.constant"() <{value = 0 : i1}> : () -> i1
    "scf.yield"(%constv_179) : (i1) -> ()
  }) : (i1) -> i1
  %if_180 = "scf.if"(%if_172) ({
    %fieldnode_185 = sdql.access_record %ps "ps_suppkey" : record<"ps_partkey": i32, "ps_suppkey": i32, "ps_availqty": f64, "ps_supplycost": f64, "ps_comment": memref<199xi8>> -> i32
    %get_184 = sdql.lookup_dictionary %s_h [%fieldnode_185 : i32] : dictionary<i32, record<"_" : f64, "_" : memref<25xi8>, "_" : memref<25xi8>, "_" : memref<40xi8>, "_" : memref<15xi8>, "_" : memref<101xi8>>> -> record<"_" : f64, "_" : memref<25xi8>, "_" : memref<25xi8>, "_" : memref<40xi8>, "_" : memref<15xi8>, "_" : memref<101xi8>>
    %consti_186 = "arith.constant"() <{value = 0 : i32}> : () -> i32
    %get_183 = sdql.access_record %get_184 "%consti_186" : record<"_" : f64, "_" : memref<25xi8>, "_" : memref<25xi8>, "_" : memref<40xi8>, "_" : memref<15xi8>, "_" : memref<101xi8>> -> f64
    %fieldnode_189 = sdql.access_record %ps "ps_suppkey" : record<"ps_partkey": i32, "ps_suppkey": i32, "ps_availqty": f64, "ps_supplycost": f64, "ps_comment": memref<199xi8>> -> i32
    %get_188 = sdql.lookup_dictionary %s_h [%fieldnode_189 : i32] : dictionary<i32, record<"_" : f64, "_" : memref<25xi8>, "_" : memref<25xi8>, "_" : memref<40xi8>, "_" : memref<15xi8>, "_" : memref<101xi8>>> -> record<"_" : f64, "_" : memref<25xi8>, "_" : memref<25xi8>, "_" : memref<40xi8>, "_" : memref<15xi8>, "_" : memref<101xi8>>
    %consti_190 = "arith.constant"() <{value = 1 : i32}> : () -> i32
    %get_187 = sdql.access_record %get_188 "%consti_190" : record<"_" : f64, "_" : memref<25xi8>, "_" : memref<25xi8>, "_" : memref<40xi8>, "_" : memref<15xi8>, "_" : memref<101xi8>> -> memref<25xi8>
    %fieldnode_193 = sdql.access_record %ps "ps_suppkey" : record<"ps_partkey": i32, "ps_suppkey": i32, "ps_availqty": f64, "ps_supplycost": f64, "ps_comment": memref<199xi8>> -> i32
    %get_192 = sdql.lookup_dictionary %s_h [%fieldnode_193 : i32] : dictionary<i32, record<"_" : f64, "_" : memref<25xi8>, "_" : memref<25xi8>, "_" : memref<40xi8>, "_" : memref<15xi8>, "_" : memref<101xi8>>> -> record<"_" : f64, "_" : memref<25xi8>, "_" : memref<25xi8>, "_" : memref<40xi8>, "_" : memref<15xi8>, "_" : memref<101xi8>>
    %consti_194 = "arith.constant"() <{value = 2 : i32}> : () -> i32
    %get_191 = sdql.access_record %get_192 "%consti_194" : record<"_" : f64, "_" : memref<25xi8>, "_" : memref<25xi8>, "_" : memref<40xi8>, "_" : memref<15xi8>, "_" : memref<101xi8>> -> memref<25xi8>
    %fieldnode_195 = sdql.access_record %ps "ps_partkey" : record<"ps_partkey": i32, "ps_suppkey": i32, "ps_availqty": f64, "ps_supplycost": f64, "ps_comment": memref<199xi8>> -> i32
    %fieldnode_198 = sdql.access_record %ps "ps_partkey" : record<"ps_partkey": i32, "ps_suppkey": i32, "ps_availqty": f64, "ps_supplycost": f64, "ps_comment": memref<199xi8>> -> i32
    %get_197 = sdql.lookup_dictionary %p_h [%fieldnode_198 : i32] : dictionary<i32, record<"_" : memref<25xi8>>> -> record<"_" : memref<25xi8>>
    %consti_199 = "arith.constant"() <{value = 0 : i32}> : () -> i32
    %get_196 = sdql.access_record %get_197 "%consti_199" : record<"_" : memref<25xi8>> -> memref<25xi8>
    %fieldnode_202 = sdql.access_record %ps "ps_suppkey" : record<"ps_partkey": i32, "ps_suppkey": i32, "ps_availqty": f64, "ps_supplycost": f64, "ps_comment": memref<199xi8>> -> i32
    %get_201 = sdql.lookup_dictionary %s_h [%fieldnode_202 : i32] : dictionary<i32, record<"_" : f64, "_" : memref<25xi8>, "_" : memref<25xi8>, "_" : memref<40xi8>, "_" : memref<15xi8>, "_" : memref<101xi8>>> -> record<"_" : f64, "_" : memref<25xi8>, "_" : memref<25xi8>, "_" : memref<40xi8>, "_" : memref<15xi8>, "_" : memref<101xi8>>
    %consti_203 = "arith.constant"() <{value = 4 : i32}> : () -> i32
    %get_200 = sdql.access_record %get_201 "%consti_203" : record<"_" : f64, "_" : memref<25xi8>, "_" : memref<25xi8>, "_" : memref<40xi8>, "_" : memref<15xi8>, "_" : memref<101xi8>> -> memref<15xi8>
    %fieldnode_206 = sdql.access_record %ps "ps_suppkey" : record<"ps_partkey": i32, "ps_suppkey": i32, "ps_availqty": f64, "ps_supplycost": f64, "ps_comment": memref<199xi8>> -> i32
    %get_205 = sdql.lookup_dictionary %s_h [%fieldnode_206 : i32] : dictionary<i32, record<"_" : f64, "_" : memref<25xi8>, "_" : memref<25xi8>, "_" : memref<40xi8>, "_" : memref<15xi8>, "_" : memref<101xi8>>> -> record<"_" : f64, "_" : memref<25xi8>, "_" : memref<25xi8>, "_" : memref<40xi8>, "_" : memref<15xi8>, "_" : memref<101xi8>>
    %consti_207 = "arith.constant"() <{value = 3 : i32}> : () -> i32
    %get_204 = sdql.access_record %get_205 "%consti_207" : record<"_" : f64, "_" : memref<25xi8>, "_" : memref<25xi8>, "_" : memref<40xi8>, "_" : memref<15xi8>, "_" : memref<101xi8>> -> memref<40xi8>
    %fieldnode_210 = sdql.access_record %ps "ps_suppkey" : record<"ps_partkey": i32, "ps_suppkey": i32, "ps_availqty": f64, "ps_supplycost": f64, "ps_comment": memref<199xi8>> -> i32
    %get_209 = sdql.lookup_dictionary %s_h [%fieldnode_210 : i32] : dictionary<i32, record<"_" : f64, "_" : memref<25xi8>, "_" : memref<25xi8>, "_" : memref<40xi8>, "_" : memref<15xi8>, "_" : memref<101xi8>>> -> record<"_" : f64, "_" : memref<25xi8>, "_" : memref<25xi8>, "_" : memref<40xi8>, "_" : memref<15xi8>, "_" : memref<101xi8>>
    %consti_211 = "arith.constant"() <{value = 5 : i32}> : () -> i32
    %get_208 = sdql.access_record %get_209 "%consti_211" : record<"_" : f64, "_" : memref<25xi8>, "_" : memref<25xi8>, "_" : memref<40xi8>, "_" : memref<15xi8>, "_" : memref<101xi8>> -> memref<101xi8>
    %recnode_212 = sdql.create_record {fields = ["_", "_", "_", "_", "_", "_", "_", "_"]} %get_183, %get_187, %get_191, %fieldnode_195, %get_196, %get_200, %get_204, %get_208 : f64, memref<25xi8>, memref<25xi8>, i32, memref<25xi8>, memref<15xi8>, memref<40xi8>, memref<101xi8> -> record<"_": f64, "_": memref<25xi8>, "_": memref<25xi8>, "_": i32, "_": memref<25xi8>, "_": memref<15xi8>, "_": memref<40xi8>, "_": memref<101xi8>>
    %uniq_182 = sdql.unique %recnode_212 : record<"_" : f64, "_" : memref<25xi8>, "_" : memref<25xi8>, "_" : i32, "_" : memref<25xi8>, "_" : memref<15xi8>, "_" : memref<40xi8>, "_" : memref<101xi8>> -> record<"_" : f64, "_" : memref<25xi8>, "_" : memref<25xi8>, "_" : i32, "_" : memref<25xi8>, "_" : memref<15xi8>, "_" : memref<40xi8>, "_" : memref<101xi8>>
    %consti_213 = "arith.constant"() <{value = 1 : i32}> : () -> i32
    %dict_181 = sdql.create_dictionary %uniq_182, %consti_213 : record<"_" : f64, "_" : memref<25xi8>, "_" : memref<25xi8>, "_" : i32, "_" : memref<25xi8>, "_" : memref<15xi8>, "_" : memref<40xi8>, "_" : memref<101xi8>>, i32 -> dictionary<record<"_" : f64, "_" : memref<25xi8>, "_" : memref<25xi8>, "_" : i32, "_" : memref<25xi8>, "_" : memref<15xi8>, "_" : memref<40xi8>, "_" : memref<101xi8>>, i32>
    "scf.yield"(%dict_181) : (dictionary<record<"_" : f64, "_" : memref<25xi8>, "_" : memref<25xi8>, "_" : i32, "_" : memref<25xi8>, "_" : memref<15xi8>, "_" : memref<40xi8>, "_" : memref<101xi8>>, i32>) -> ()
  }, {
    %dict_214 = sdql.empty_dictionary : dictionary<record<"_" : f64, "_" : memref<25xi8>, "_" : memref<25xi8>, "_" : i32, "_" : memref<25xi8>, "_" : memref<15xi8>, "_" : memref<40xi8>, "_" : memref<101xi8>>, i32>
    "scf.yield"(%dict_214) : (dictionary<record<"_" : f64, "_" : memref<25xi8>, "_" : memref<25xi8>, "_" : i32, "_" : memref<25xi8>, "_" : memref<15xi8>, "_" : memref<40xi8>, "_" : memref<101xi8>>, i32>) -> ()
  }) : (i1) -> dictionary<record<"_" : f64, "_" : memref<25xi8>, "_" : memref<25xi8>, "_" : i32, "_" : memref<25xi8>, "_" : memref<15xi8>, "_" : memref<40xi8>, "_" : memref<101xi8>>, i32>
  sdql.yield %if_180 : dictionary<record<"_" : f64, "_" : memref<25xi8>, "_" : memref<25xi8>, "_" : i32, "_" : memref<25xi8>, "_" : memref<15xi8>, "_" : memref<40xi8>, "_" : memref<101xi8>>, i32>
}
