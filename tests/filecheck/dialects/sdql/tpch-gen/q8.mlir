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
%lineitem = sdql.load "datasets/tpch/lineitem.tbl" : record<"l_orderkey" : dictionary<i32, i32>, "l_partkey" : dictionary<i32, i32>, "l_suppkey" : dictionary<i32, i32>, "l_linenumber" : dictionary<i32, i32>, "l_quantity" : dictionary<i32, f64>, "l_extendedprice" : dictionary<i32, f64>, "l_discount" : dictionary<i32, f64>, "l_tax" : dictionary<i32, f64>, "l_returnflag" : dictionary<i32, memref<1xi8>>, "l_linestatus" : dictionary<i32, memref<1xi8>>, "l_shipdate" : dictionary<i32, i32>, "l_commitdate" : dictionary<i32, i32>, "l_receiptdate" : dictionary<i32, i32>, "l_shipinstruct" : dictionary<i32, memref<25xi8>>, "l_shipmode" : dictionary<i32, memref<10xi8>>, "l_comment" : dictionary<i32, memref<44xi8>>, "size" : i32>
%orders = sdql.load "datasets/tpch/orders.tbl" : record<"o_orderkey" : dictionary<i32, i32>, "o_custkey" : dictionary<i32, i32>, "o_orderstatus" : dictionary<i32, memref<1xi8>>, "o_totalprice" : dictionary<i32, f64>, "o_orderdate" : dictionary<i32, i32>, "o_orderpriority" : dictionary<i32, memref<15xi8>>, "o_clerk" : dictionary<i32, memref<15xi8>>, "o_shippriority" : dictionary<i32, i32>, "o_comment" : dictionary<i32, memref<79xi8>>, "size" : i32>
%customer = sdql.load "datasets/tpch/customer.tbl" : record<"c_custkey" : dictionary<i32, i32>, "c_name" : dictionary<i32, memref<25xi8>>, "c_address" : dictionary<i32, memref<40xi8>>, "c_nationkey" : dictionary<i32, i32>, "c_phone" : dictionary<i32, memref<15xi8>>, "c_acctbal" : dictionary<i32, f64>, "c_mktsegment" : dictionary<i32, memref<10xi8>>, "c_comment" : dictionary<i32, memref<117xi8>>, "size" : i32>
%nation = sdql.load "datasets/tpch/nation.tbl" : record<"n_nationkey" : dictionary<i32, i32>, "n_name" : dictionary<i32, memref<25xi8>>, "n_regionkey" : dictionary<i32, i32>, "n_comment" : dictionary<i32, memref<152xi8>>, "size" : i32>
%region = sdql.load "datasets/tpch/region.tbl" : record<"r_regionkey" : dictionary<i32, i32>, "r_name" : dictionary<i32, memref<25xi8>>, "r_comment" : dictionary<i32, memref<152xi8>>, "size" : i32>
// ECONOMY ANODIZED STEEL
%constds_0 = "arith.constant"() <{value = dense<[69, 67, 79, 78, 79, 77, 89, 32, 65, 78, 79, 68, 73, 90, 69, 68, 32, 83, 84, 69, 69, 76]> : memref<?xi8>}> : () -> memref<?xi8>
%consti_1 = "arith.constant"() <{value = 23 : i32}> : () -> i32
%steel = sdql.external "ConstantString", %constds_0, %consti_1 : memref<?xi8>, i32 -> memref<22xi8>
// AMERICA
%constds_2 = "arith.constant"() <{value = dense<[65, 77, 69, 82, 73, 67, 65]> : memref<?xi8>}> : () -> memref<?xi8>
%consti_3 = "arith.constant"() <{value = 8 : i32}> : () -> i32
%america = sdql.external "ConstantString", %constds_2, %consti_3 : memref<?xi8>, i32 -> memref<7xi8>
// BRAZIL
%constds_4 = "arith.constant"() <{value = dense<[66, 82, 65, 90, 73, 76]> : memref<?xi8>}> : () -> memref<?xi8>
%consti_5 = "arith.constant"() <{value = 7 : i32}> : () -> i32
%brazil = sdql.external "ConstantString", %constds_4, %consti_5 : memref<?xi8>, i32 -> memref<6xi8>
%fieldnode_6 = sdql.access_record %region "size" : record<"r_regionkey": dictionary<i32, i32>, "r_name": dictionary<i32, memref<25xi8>>, "r_comment": dictionary<i32, memref<152xi8>>, "size": i32> -> i32
%rangenode_7 = "func.call"(%fieldnode_6) <{callee = @range_builtin}> : (i32) -> dictionary<i32, i32>
%r_h = sdql.sum %rangenode_7 : dictionary<i32, i32> -> dictionary<i32, record<"_" : i32>> {
^bb_8(%i: i32, %_: i32):
  %fieldnode_10 = sdql.access_record %region "r_regionkey" : record<"r_regionkey": dictionary<i32, i32>, "r_name": dictionary<i32, memref<25xi8>>, "r_comment": dictionary<i32, memref<152xi8>>, "size": i32> -> dictionary<i32, i32>
  %get_9 = sdql.lookup_dictionary %fieldnode_10 [%i : i32] : dictionary<i32, i32> -> i32
  %fieldnode_12 = sdql.access_record %region "r_name" : record<"r_regionkey": dictionary<i32, i32>, "r_name": dictionary<i32, memref<25xi8>>, "r_comment": dictionary<i32, memref<152xi8>>, "size": i32> -> dictionary<i32, memref<25xi8>>
  %get_11 = sdql.lookup_dictionary %fieldnode_12 [%i : i32] : dictionary<i32, memref<25xi8>> -> memref<25xi8>
  %fieldnode_14 = sdql.access_record %region "r_comment" : record<"r_regionkey": dictionary<i32, i32>, "r_name": dictionary<i32, memref<25xi8>>, "r_comment": dictionary<i32, memref<152xi8>>, "size": i32> -> dictionary<i32, memref<152xi8>>
  %get_13 = sdql.lookup_dictionary %fieldnode_14 [%i : i32] : dictionary<i32, memref<152xi8>> -> memref<152xi8>
  %r = sdql.create_record {fields = ["r_regionkey", "r_name", "r_comment"]} %get_9, %get_11, %get_13 : i32, memref<25xi8>, memref<152xi8> -> record<"r_regionkey": i32, "r_name": memref<25xi8>, "r_comment": memref<152xi8>>
  %fieldnode_15 = sdql.access_record %r "r_name" : record<"r_regionkey": i32, "r_name": memref<25xi8>, "r_comment": memref<152xi8>> -> memref<25xi8>
  %cmp_16 = sdql.cmp %fieldnode_15, %america : memref<25xi8>, memref<7xi8> -> i1
  %if_17 = "scf.if"(%cmp_16) ({
    %fieldnode_20 = sdql.access_record %r "r_regionkey" : record<"r_regionkey": i32, "r_name": memref<25xi8>, "r_comment": memref<152xi8>> -> i32
    %uniq_19 = sdql.unique %fieldnode_20 : i32 -> i32
    %fieldnode_21 = sdql.access_record %r "r_regionkey" : record<"r_regionkey": i32, "r_name": memref<25xi8>, "r_comment": memref<152xi8>> -> i32
    %recnode_22 = sdql.create_record {fields = ["_"]} %fieldnode_21 : i32 -> record<"_": i32>
    %dict_18 = sdql.create_dictionary %uniq_19, %recnode_22 : i32, record<"_" : i32> -> dictionary<i32, record<"_" : i32>>
    "scf.yield"(%dict_18) : (dictionary<i32, record<"_" : i32>>) -> ()
  }, {
    %dict_23 = sdql.empty_dictionary : dictionary<i32, record<"_" : i32>>
    "scf.yield"(%dict_23) : (dictionary<i32, record<"_" : i32>>) -> ()
  }) : (i1) -> dictionary<i32, record<"_" : i32>>
  sdql.yield %if_17 : dictionary<i32, record<"_" : i32>>
}
%fieldnode_24 = sdql.access_record %nation "size" : record<"n_nationkey": dictionary<i32, i32>, "n_name": dictionary<i32, memref<25xi8>>, "n_regionkey": dictionary<i32, i32>, "n_comment": dictionary<i32, memref<152xi8>>, "size": i32> -> i32
%rangenode_25 = "func.call"(%fieldnode_24) <{callee = @range_builtin}> : (i32) -> dictionary<i32, i32>
%n_h = sdql.sum %rangenode_25 : dictionary<i32, i32> -> dictionary<i32, i32> {
^bb_26(%i: i32, %_: i32):
  %fieldnode_28 = sdql.access_record %nation "n_nationkey" : record<"n_nationkey": dictionary<i32, i32>, "n_name": dictionary<i32, memref<25xi8>>, "n_regionkey": dictionary<i32, i32>, "n_comment": dictionary<i32, memref<152xi8>>, "size": i32> -> dictionary<i32, i32>
  %get_27 = sdql.lookup_dictionary %fieldnode_28 [%i : i32] : dictionary<i32, i32> -> i32
  %fieldnode_30 = sdql.access_record %nation "n_name" : record<"n_nationkey": dictionary<i32, i32>, "n_name": dictionary<i32, memref<25xi8>>, "n_regionkey": dictionary<i32, i32>, "n_comment": dictionary<i32, memref<152xi8>>, "size": i32> -> dictionary<i32, memref<25xi8>>
  %get_29 = sdql.lookup_dictionary %fieldnode_30 [%i : i32] : dictionary<i32, memref<25xi8>> -> memref<25xi8>
  %fieldnode_32 = sdql.access_record %nation "n_regionkey" : record<"n_nationkey": dictionary<i32, i32>, "n_name": dictionary<i32, memref<25xi8>>, "n_regionkey": dictionary<i32, i32>, "n_comment": dictionary<i32, memref<152xi8>>, "size": i32> -> dictionary<i32, i32>
  %get_31 = sdql.lookup_dictionary %fieldnode_32 [%i : i32] : dictionary<i32, i32> -> i32
  %fieldnode_34 = sdql.access_record %nation "n_comment" : record<"n_nationkey": dictionary<i32, i32>, "n_name": dictionary<i32, memref<25xi8>>, "n_regionkey": dictionary<i32, i32>, "n_comment": dictionary<i32, memref<152xi8>>, "size": i32> -> dictionary<i32, memref<152xi8>>
  %get_33 = sdql.lookup_dictionary %fieldnode_34 [%i : i32] : dictionary<i32, memref<152xi8>> -> memref<152xi8>
  %n = sdql.create_record {fields = ["n_nationkey", "n_name", "n_regionkey", "n_comment"]} %get_27, %get_29, %get_31, %get_33 : i32, memref<25xi8>, i32, memref<152xi8> -> record<"n_nationkey": i32, "n_name": memref<25xi8>, "n_regionkey": i32, "n_comment": memref<152xi8>>
  %fieldnode_38 = sdql.access_record %n "n_regionkey" : record<"n_nationkey": i32, "n_name": memref<25xi8>, "n_regionkey": i32, "n_comment": memref<152xi8>> -> i32
  %get_37 = sdql.lookup_dictionary %r_h [%fieldnode_38 : i32] : dictionary<i32, record<"_" : i32>> -> record<"_" : i32>
  %dict_39 = sdql.empty_dictionary : record<"_" : i32>
  %cmp_40 = sdql.cmp %get_37, %dict_39 : record<"_" : i32>, record<"_" : i32> -> i1
  %zero_35 = "arith.constant"() <{value = 0 : i1}> : () -> i1
  %neg_36 = "arith.cmpi"(%cmp_40, %zero_35) <{predicate = 0}> : (i1, i1) -> i1
  %if_41 = "scf.if"(%neg_36) ({
    %fieldnode_44 = sdql.access_record %n "n_nationkey" : record<"n_nationkey": i32, "n_name": memref<25xi8>, "n_regionkey": i32, "n_comment": memref<152xi8>> -> i32
    %uniq_43 = sdql.unique %fieldnode_44 : i32 -> i32
    %consti_45 = "arith.constant"() <{value = 1 : i32}> : () -> i32
    %dict_42 = sdql.create_dictionary %uniq_43, %consti_45 : i32, i32 -> dictionary<i32, i32>
    "scf.yield"(%dict_42) : (dictionary<i32, i32>) -> ()
  }, {
    %dict_46 = sdql.empty_dictionary : dictionary<i32, i32>
    "scf.yield"(%dict_46) : (dictionary<i32, i32>) -> ()
  }) : (i1) -> dictionary<i32, i32>
  sdql.yield %if_41 : dictionary<i32, i32>
}
%fieldnode_47 = sdql.access_record %nation "size" : record<"n_nationkey": dictionary<i32, i32>, "n_name": dictionary<i32, memref<25xi8>>, "n_regionkey": dictionary<i32, i32>, "n_comment": dictionary<i32, memref<152xi8>>, "size": i32> -> i32
%rangenode_48 = "func.call"(%fieldnode_47) <{callee = @range_builtin}> : (i32) -> dictionary<i32, i32>
%nationkey_to_name = sdql.sum %rangenode_48 : dictionary<i32, i32> -> dictionary<i32, record<"_" : memref<25xi8>>> {
^bb_49(%i: i32, %_: i32):
  %fieldnode_51 = sdql.access_record %nation "n_nationkey" : record<"n_nationkey": dictionary<i32, i32>, "n_name": dictionary<i32, memref<25xi8>>, "n_regionkey": dictionary<i32, i32>, "n_comment": dictionary<i32, memref<152xi8>>, "size": i32> -> dictionary<i32, i32>
  %get_50 = sdql.lookup_dictionary %fieldnode_51 [%i : i32] : dictionary<i32, i32> -> i32
  %fieldnode_53 = sdql.access_record %nation "n_name" : record<"n_nationkey": dictionary<i32, i32>, "n_name": dictionary<i32, memref<25xi8>>, "n_regionkey": dictionary<i32, i32>, "n_comment": dictionary<i32, memref<152xi8>>, "size": i32> -> dictionary<i32, memref<25xi8>>
  %get_52 = sdql.lookup_dictionary %fieldnode_53 [%i : i32] : dictionary<i32, memref<25xi8>> -> memref<25xi8>
  %fieldnode_55 = sdql.access_record %nation "n_regionkey" : record<"n_nationkey": dictionary<i32, i32>, "n_name": dictionary<i32, memref<25xi8>>, "n_regionkey": dictionary<i32, i32>, "n_comment": dictionary<i32, memref<152xi8>>, "size": i32> -> dictionary<i32, i32>
  %get_54 = sdql.lookup_dictionary %fieldnode_55 [%i : i32] : dictionary<i32, i32> -> i32
  %fieldnode_57 = sdql.access_record %nation "n_comment" : record<"n_nationkey": dictionary<i32, i32>, "n_name": dictionary<i32, memref<25xi8>>, "n_regionkey": dictionary<i32, i32>, "n_comment": dictionary<i32, memref<152xi8>>, "size": i32> -> dictionary<i32, memref<152xi8>>
  %get_56 = sdql.lookup_dictionary %fieldnode_57 [%i : i32] : dictionary<i32, memref<152xi8>> -> memref<152xi8>
  %n = sdql.create_record {fields = ["n_nationkey", "n_name", "n_regionkey", "n_comment"]} %get_50, %get_52, %get_54, %get_56 : i32, memref<25xi8>, i32, memref<152xi8> -> record<"n_nationkey": i32, "n_name": memref<25xi8>, "n_regionkey": i32, "n_comment": memref<152xi8>>
  %fieldnode_60 = sdql.access_record %n "n_nationkey" : record<"n_nationkey": i32, "n_name": memref<25xi8>, "n_regionkey": i32, "n_comment": memref<152xi8>> -> i32
  %uniq_59 = sdql.unique %fieldnode_60 : i32 -> i32
  %fieldnode_61 = sdql.access_record %n "n_name" : record<"n_nationkey": i32, "n_name": memref<25xi8>, "n_regionkey": i32, "n_comment": memref<152xi8>> -> memref<25xi8>
  %recnode_62 = sdql.create_record {fields = ["_"]} %fieldnode_61 : memref<25xi8> -> record<"_": memref<25xi8>>
  %dict_58 = sdql.create_dictionary %uniq_59, %recnode_62 : i32, record<"_" : memref<25xi8>> -> dictionary<i32, record<"_" : memref<25xi8>>>
  sdql.yield %dict_58 : dictionary<i32, record<"_" : memref<25xi8>>>
}
%fieldnode_63 = sdql.access_record %supplier "size" : record<"s_suppkey": dictionary<i32, i32>, "s_name": dictionary<i32, memref<25xi8>>, "s_address": dictionary<i32, memref<40xi8>>, "s_nationkey": dictionary<i32, i32>, "s_phone": dictionary<i32, memref<15xi8>>, "s_acctbal": dictionary<i32, f64>, "s_comment": dictionary<i32, memref<101xi8>>, "size": i32> -> i32
%rangenode_64 = "func.call"(%fieldnode_63) <{callee = @range_builtin}> : (i32) -> dictionary<i32, i32>
%s_h = sdql.sum %rangenode_64 : dictionary<i32, i32> -> dictionary<i32, record<"_" : i32>> {
^bb_65(%i: i32, %_: i32):
  %fieldnode_67 = sdql.access_record %supplier "s_suppkey" : record<"s_suppkey": dictionary<i32, i32>, "s_name": dictionary<i32, memref<25xi8>>, "s_address": dictionary<i32, memref<40xi8>>, "s_nationkey": dictionary<i32, i32>, "s_phone": dictionary<i32, memref<15xi8>>, "s_acctbal": dictionary<i32, f64>, "s_comment": dictionary<i32, memref<101xi8>>, "size": i32> -> dictionary<i32, i32>
  %get_66 = sdql.lookup_dictionary %fieldnode_67 [%i : i32] : dictionary<i32, i32> -> i32
  %fieldnode_69 = sdql.access_record %supplier "s_name" : record<"s_suppkey": dictionary<i32, i32>, "s_name": dictionary<i32, memref<25xi8>>, "s_address": dictionary<i32, memref<40xi8>>, "s_nationkey": dictionary<i32, i32>, "s_phone": dictionary<i32, memref<15xi8>>, "s_acctbal": dictionary<i32, f64>, "s_comment": dictionary<i32, memref<101xi8>>, "size": i32> -> dictionary<i32, memref<25xi8>>
  %get_68 = sdql.lookup_dictionary %fieldnode_69 [%i : i32] : dictionary<i32, memref<25xi8>> -> memref<25xi8>
  %fieldnode_71 = sdql.access_record %supplier "s_address" : record<"s_suppkey": dictionary<i32, i32>, "s_name": dictionary<i32, memref<25xi8>>, "s_address": dictionary<i32, memref<40xi8>>, "s_nationkey": dictionary<i32, i32>, "s_phone": dictionary<i32, memref<15xi8>>, "s_acctbal": dictionary<i32, f64>, "s_comment": dictionary<i32, memref<101xi8>>, "size": i32> -> dictionary<i32, memref<40xi8>>
  %get_70 = sdql.lookup_dictionary %fieldnode_71 [%i : i32] : dictionary<i32, memref<40xi8>> -> memref<40xi8>
  %fieldnode_73 = sdql.access_record %supplier "s_nationkey" : record<"s_suppkey": dictionary<i32, i32>, "s_name": dictionary<i32, memref<25xi8>>, "s_address": dictionary<i32, memref<40xi8>>, "s_nationkey": dictionary<i32, i32>, "s_phone": dictionary<i32, memref<15xi8>>, "s_acctbal": dictionary<i32, f64>, "s_comment": dictionary<i32, memref<101xi8>>, "size": i32> -> dictionary<i32, i32>
  %get_72 = sdql.lookup_dictionary %fieldnode_73 [%i : i32] : dictionary<i32, i32> -> i32
  %fieldnode_75 = sdql.access_record %supplier "s_phone" : record<"s_suppkey": dictionary<i32, i32>, "s_name": dictionary<i32, memref<25xi8>>, "s_address": dictionary<i32, memref<40xi8>>, "s_nationkey": dictionary<i32, i32>, "s_phone": dictionary<i32, memref<15xi8>>, "s_acctbal": dictionary<i32, f64>, "s_comment": dictionary<i32, memref<101xi8>>, "size": i32> -> dictionary<i32, memref<15xi8>>
  %get_74 = sdql.lookup_dictionary %fieldnode_75 [%i : i32] : dictionary<i32, memref<15xi8>> -> memref<15xi8>
  %fieldnode_77 = sdql.access_record %supplier "s_acctbal" : record<"s_suppkey": dictionary<i32, i32>, "s_name": dictionary<i32, memref<25xi8>>, "s_address": dictionary<i32, memref<40xi8>>, "s_nationkey": dictionary<i32, i32>, "s_phone": dictionary<i32, memref<15xi8>>, "s_acctbal": dictionary<i32, f64>, "s_comment": dictionary<i32, memref<101xi8>>, "size": i32> -> dictionary<i32, f64>
  %get_76 = sdql.lookup_dictionary %fieldnode_77 [%i : i32] : dictionary<i32, f64> -> f64
  %fieldnode_79 = sdql.access_record %supplier "s_comment" : record<"s_suppkey": dictionary<i32, i32>, "s_name": dictionary<i32, memref<25xi8>>, "s_address": dictionary<i32, memref<40xi8>>, "s_nationkey": dictionary<i32, i32>, "s_phone": dictionary<i32, memref<15xi8>>, "s_acctbal": dictionary<i32, f64>, "s_comment": dictionary<i32, memref<101xi8>>, "size": i32> -> dictionary<i32, memref<101xi8>>
  %get_78 = sdql.lookup_dictionary %fieldnode_79 [%i : i32] : dictionary<i32, memref<101xi8>> -> memref<101xi8>
  %s = sdql.create_record {fields = ["s_suppkey", "s_name", "s_address", "s_nationkey", "s_phone", "s_acctbal", "s_comment"]} %get_66, %get_68, %get_70, %get_72, %get_74, %get_76, %get_78 : i32, memref<25xi8>, memref<40xi8>, i32, memref<15xi8>, f64, memref<101xi8> -> record<"s_suppkey": i32, "s_name": memref<25xi8>, "s_address": memref<40xi8>, "s_nationkey": i32, "s_phone": memref<15xi8>, "s_acctbal": f64, "s_comment": memref<101xi8>>
  %fieldnode_82 = sdql.access_record %s "s_suppkey" : record<"s_suppkey": i32, "s_name": memref<25xi8>, "s_address": memref<40xi8>, "s_nationkey": i32, "s_phone": memref<15xi8>, "s_acctbal": f64, "s_comment": memref<101xi8>> -> i32
  %uniq_81 = sdql.unique %fieldnode_82 : i32 -> i32
  %fieldnode_83 = sdql.access_record %s "s_nationkey" : record<"s_suppkey": i32, "s_name": memref<25xi8>, "s_address": memref<40xi8>, "s_nationkey": i32, "s_phone": memref<15xi8>, "s_acctbal": f64, "s_comment": memref<101xi8>> -> i32
  %recnode_84 = sdql.create_record {fields = ["_"]} %fieldnode_83 : i32 -> record<"_": i32>
  %dict_80 = sdql.create_dictionary %uniq_81, %recnode_84 : i32, record<"_" : i32> -> dictionary<i32, record<"_" : i32>>
  sdql.yield %dict_80 : dictionary<i32, record<"_" : i32>>
}
%fieldnode_85 = sdql.access_record %customer "size" : record<"c_custkey": dictionary<i32, i32>, "c_name": dictionary<i32, memref<25xi8>>, "c_address": dictionary<i32, memref<40xi8>>, "c_nationkey": dictionary<i32, i32>, "c_phone": dictionary<i32, memref<15xi8>>, "c_acctbal": dictionary<i32, f64>, "c_mktsegment": dictionary<i32, memref<10xi8>>, "c_comment": dictionary<i32, memref<117xi8>>, "size": i32> -> i32
%rangenode_86 = "func.call"(%fieldnode_85) <{callee = @range_builtin}> : (i32) -> dictionary<i32, i32>
%c_h = sdql.sum %rangenode_86 : dictionary<i32, i32> -> dictionary<i32, i32> {
^bb_87(%i: i32, %_: i32):
  %fieldnode_89 = sdql.access_record %customer "c_custkey" : record<"c_custkey": dictionary<i32, i32>, "c_name": dictionary<i32, memref<25xi8>>, "c_address": dictionary<i32, memref<40xi8>>, "c_nationkey": dictionary<i32, i32>, "c_phone": dictionary<i32, memref<15xi8>>, "c_acctbal": dictionary<i32, f64>, "c_mktsegment": dictionary<i32, memref<10xi8>>, "c_comment": dictionary<i32, memref<117xi8>>, "size": i32> -> dictionary<i32, i32>
  %get_88 = sdql.lookup_dictionary %fieldnode_89 [%i : i32] : dictionary<i32, i32> -> i32
  %fieldnode_91 = sdql.access_record %customer "c_name" : record<"c_custkey": dictionary<i32, i32>, "c_name": dictionary<i32, memref<25xi8>>, "c_address": dictionary<i32, memref<40xi8>>, "c_nationkey": dictionary<i32, i32>, "c_phone": dictionary<i32, memref<15xi8>>, "c_acctbal": dictionary<i32, f64>, "c_mktsegment": dictionary<i32, memref<10xi8>>, "c_comment": dictionary<i32, memref<117xi8>>, "size": i32> -> dictionary<i32, memref<25xi8>>
  %get_90 = sdql.lookup_dictionary %fieldnode_91 [%i : i32] : dictionary<i32, memref<25xi8>> -> memref<25xi8>
  %fieldnode_93 = sdql.access_record %customer "c_address" : record<"c_custkey": dictionary<i32, i32>, "c_name": dictionary<i32, memref<25xi8>>, "c_address": dictionary<i32, memref<40xi8>>, "c_nationkey": dictionary<i32, i32>, "c_phone": dictionary<i32, memref<15xi8>>, "c_acctbal": dictionary<i32, f64>, "c_mktsegment": dictionary<i32, memref<10xi8>>, "c_comment": dictionary<i32, memref<117xi8>>, "size": i32> -> dictionary<i32, memref<40xi8>>
  %get_92 = sdql.lookup_dictionary %fieldnode_93 [%i : i32] : dictionary<i32, memref<40xi8>> -> memref<40xi8>
  %fieldnode_95 = sdql.access_record %customer "c_nationkey" : record<"c_custkey": dictionary<i32, i32>, "c_name": dictionary<i32, memref<25xi8>>, "c_address": dictionary<i32, memref<40xi8>>, "c_nationkey": dictionary<i32, i32>, "c_phone": dictionary<i32, memref<15xi8>>, "c_acctbal": dictionary<i32, f64>, "c_mktsegment": dictionary<i32, memref<10xi8>>, "c_comment": dictionary<i32, memref<117xi8>>, "size": i32> -> dictionary<i32, i32>
  %get_94 = sdql.lookup_dictionary %fieldnode_95 [%i : i32] : dictionary<i32, i32> -> i32
  %fieldnode_97 = sdql.access_record %customer "c_phone" : record<"c_custkey": dictionary<i32, i32>, "c_name": dictionary<i32, memref<25xi8>>, "c_address": dictionary<i32, memref<40xi8>>, "c_nationkey": dictionary<i32, i32>, "c_phone": dictionary<i32, memref<15xi8>>, "c_acctbal": dictionary<i32, f64>, "c_mktsegment": dictionary<i32, memref<10xi8>>, "c_comment": dictionary<i32, memref<117xi8>>, "size": i32> -> dictionary<i32, memref<15xi8>>
  %get_96 = sdql.lookup_dictionary %fieldnode_97 [%i : i32] : dictionary<i32, memref<15xi8>> -> memref<15xi8>
  %fieldnode_99 = sdql.access_record %customer "c_acctbal" : record<"c_custkey": dictionary<i32, i32>, "c_name": dictionary<i32, memref<25xi8>>, "c_address": dictionary<i32, memref<40xi8>>, "c_nationkey": dictionary<i32, i32>, "c_phone": dictionary<i32, memref<15xi8>>, "c_acctbal": dictionary<i32, f64>, "c_mktsegment": dictionary<i32, memref<10xi8>>, "c_comment": dictionary<i32, memref<117xi8>>, "size": i32> -> dictionary<i32, f64>
  %get_98 = sdql.lookup_dictionary %fieldnode_99 [%i : i32] : dictionary<i32, f64> -> f64
  %fieldnode_101 = sdql.access_record %customer "c_mktsegment" : record<"c_custkey": dictionary<i32, i32>, "c_name": dictionary<i32, memref<25xi8>>, "c_address": dictionary<i32, memref<40xi8>>, "c_nationkey": dictionary<i32, i32>, "c_phone": dictionary<i32, memref<15xi8>>, "c_acctbal": dictionary<i32, f64>, "c_mktsegment": dictionary<i32, memref<10xi8>>, "c_comment": dictionary<i32, memref<117xi8>>, "size": i32> -> dictionary<i32, memref<10xi8>>
  %get_100 = sdql.lookup_dictionary %fieldnode_101 [%i : i32] : dictionary<i32, memref<10xi8>> -> memref<10xi8>
  %fieldnode_103 = sdql.access_record %customer "c_comment" : record<"c_custkey": dictionary<i32, i32>, "c_name": dictionary<i32, memref<25xi8>>, "c_address": dictionary<i32, memref<40xi8>>, "c_nationkey": dictionary<i32, i32>, "c_phone": dictionary<i32, memref<15xi8>>, "c_acctbal": dictionary<i32, f64>, "c_mktsegment": dictionary<i32, memref<10xi8>>, "c_comment": dictionary<i32, memref<117xi8>>, "size": i32> -> dictionary<i32, memref<117xi8>>
  %get_102 = sdql.lookup_dictionary %fieldnode_103 [%i : i32] : dictionary<i32, memref<117xi8>> -> memref<117xi8>
  %c = sdql.create_record {fields = ["c_custkey", "c_name", "c_address", "c_nationkey", "c_phone", "c_acctbal", "c_mktsegment", "c_comment"]} %get_88, %get_90, %get_92, %get_94, %get_96, %get_98, %get_100, %get_102 : i32, memref<25xi8>, memref<40xi8>, i32, memref<15xi8>, f64, memref<10xi8>, memref<117xi8> -> record<"c_custkey": i32, "c_name": memref<25xi8>, "c_address": memref<40xi8>, "c_nationkey": i32, "c_phone": memref<15xi8>, "c_acctbal": f64, "c_mktsegment": memref<10xi8>, "c_comment": memref<117xi8>>
  %fieldnode_105 = sdql.access_record %c "c_custkey" : record<"c_custkey": i32, "c_name": memref<25xi8>, "c_address": memref<40xi8>, "c_nationkey": i32, "c_phone": memref<15xi8>, "c_acctbal": f64, "c_mktsegment": memref<10xi8>, "c_comment": memref<117xi8>> -> i32
  %fieldnode_106 = sdql.access_record %c "c_nationkey" : record<"c_custkey": i32, "c_name": memref<25xi8>, "c_address": memref<40xi8>, "c_nationkey": i32, "c_phone": memref<15xi8>, "c_acctbal": f64, "c_mktsegment": memref<10xi8>, "c_comment": memref<117xi8>> -> i32
  %dict_104 = sdql.create_dictionary %fieldnode_105, %fieldnode_106 : i32, i32 -> dictionary<i32, i32>
  sdql.yield %dict_104 : dictionary<i32, i32>
}
%fieldnode_107 = sdql.access_record %part "size" : record<"p_partkey": dictionary<i32, i32>, "p_name": dictionary<i32, memref<55xi8>>, "p_mfgr": dictionary<i32, memref<25xi8>>, "p_brand": dictionary<i32, memref<10xi8>>, "p_type": dictionary<i32, memref<25xi8>>, "p_size": dictionary<i32, i32>, "p_container": dictionary<i32, memref<10xi8>>, "p_retailprice": dictionary<i32, f64>, "p_comment": dictionary<i32, memref<23xi8>>, "size": i32> -> i32
%rangenode_108 = "func.call"(%fieldnode_107) <{callee = @range_builtin}> : (i32) -> dictionary<i32, i32>
%p_h = sdql.sum %rangenode_108 : dictionary<i32, i32> -> dictionary<i32, record<"_" : i32>> {
^bb_109(%i: i32, %_: i32):
  %fieldnode_111 = sdql.access_record %part "p_partkey" : record<"p_partkey": dictionary<i32, i32>, "p_name": dictionary<i32, memref<55xi8>>, "p_mfgr": dictionary<i32, memref<25xi8>>, "p_brand": dictionary<i32, memref<10xi8>>, "p_type": dictionary<i32, memref<25xi8>>, "p_size": dictionary<i32, i32>, "p_container": dictionary<i32, memref<10xi8>>, "p_retailprice": dictionary<i32, f64>, "p_comment": dictionary<i32, memref<23xi8>>, "size": i32> -> dictionary<i32, i32>
  %get_110 = sdql.lookup_dictionary %fieldnode_111 [%i : i32] : dictionary<i32, i32> -> i32
  %fieldnode_113 = sdql.access_record %part "p_name" : record<"p_partkey": dictionary<i32, i32>, "p_name": dictionary<i32, memref<55xi8>>, "p_mfgr": dictionary<i32, memref<25xi8>>, "p_brand": dictionary<i32, memref<10xi8>>, "p_type": dictionary<i32, memref<25xi8>>, "p_size": dictionary<i32, i32>, "p_container": dictionary<i32, memref<10xi8>>, "p_retailprice": dictionary<i32, f64>, "p_comment": dictionary<i32, memref<23xi8>>, "size": i32> -> dictionary<i32, memref<55xi8>>
  %get_112 = sdql.lookup_dictionary %fieldnode_113 [%i : i32] : dictionary<i32, memref<55xi8>> -> memref<55xi8>
  %fieldnode_115 = sdql.access_record %part "p_mfgr" : record<"p_partkey": dictionary<i32, i32>, "p_name": dictionary<i32, memref<55xi8>>, "p_mfgr": dictionary<i32, memref<25xi8>>, "p_brand": dictionary<i32, memref<10xi8>>, "p_type": dictionary<i32, memref<25xi8>>, "p_size": dictionary<i32, i32>, "p_container": dictionary<i32, memref<10xi8>>, "p_retailprice": dictionary<i32, f64>, "p_comment": dictionary<i32, memref<23xi8>>, "size": i32> -> dictionary<i32, memref<25xi8>>
  %get_114 = sdql.lookup_dictionary %fieldnode_115 [%i : i32] : dictionary<i32, memref<25xi8>> -> memref<25xi8>
  %fieldnode_117 = sdql.access_record %part "p_brand" : record<"p_partkey": dictionary<i32, i32>, "p_name": dictionary<i32, memref<55xi8>>, "p_mfgr": dictionary<i32, memref<25xi8>>, "p_brand": dictionary<i32, memref<10xi8>>, "p_type": dictionary<i32, memref<25xi8>>, "p_size": dictionary<i32, i32>, "p_container": dictionary<i32, memref<10xi8>>, "p_retailprice": dictionary<i32, f64>, "p_comment": dictionary<i32, memref<23xi8>>, "size": i32> -> dictionary<i32, memref<10xi8>>
  %get_116 = sdql.lookup_dictionary %fieldnode_117 [%i : i32] : dictionary<i32, memref<10xi8>> -> memref<10xi8>
  %fieldnode_119 = sdql.access_record %part "p_type" : record<"p_partkey": dictionary<i32, i32>, "p_name": dictionary<i32, memref<55xi8>>, "p_mfgr": dictionary<i32, memref<25xi8>>, "p_brand": dictionary<i32, memref<10xi8>>, "p_type": dictionary<i32, memref<25xi8>>, "p_size": dictionary<i32, i32>, "p_container": dictionary<i32, memref<10xi8>>, "p_retailprice": dictionary<i32, f64>, "p_comment": dictionary<i32, memref<23xi8>>, "size": i32> -> dictionary<i32, memref<25xi8>>
  %get_118 = sdql.lookup_dictionary %fieldnode_119 [%i : i32] : dictionary<i32, memref<25xi8>> -> memref<25xi8>
  %fieldnode_121 = sdql.access_record %part "p_size" : record<"p_partkey": dictionary<i32, i32>, "p_name": dictionary<i32, memref<55xi8>>, "p_mfgr": dictionary<i32, memref<25xi8>>, "p_brand": dictionary<i32, memref<10xi8>>, "p_type": dictionary<i32, memref<25xi8>>, "p_size": dictionary<i32, i32>, "p_container": dictionary<i32, memref<10xi8>>, "p_retailprice": dictionary<i32, f64>, "p_comment": dictionary<i32, memref<23xi8>>, "size": i32> -> dictionary<i32, i32>
  %get_120 = sdql.lookup_dictionary %fieldnode_121 [%i : i32] : dictionary<i32, i32> -> i32
  %fieldnode_123 = sdql.access_record %part "p_container" : record<"p_partkey": dictionary<i32, i32>, "p_name": dictionary<i32, memref<55xi8>>, "p_mfgr": dictionary<i32, memref<25xi8>>, "p_brand": dictionary<i32, memref<10xi8>>, "p_type": dictionary<i32, memref<25xi8>>, "p_size": dictionary<i32, i32>, "p_container": dictionary<i32, memref<10xi8>>, "p_retailprice": dictionary<i32, f64>, "p_comment": dictionary<i32, memref<23xi8>>, "size": i32> -> dictionary<i32, memref<10xi8>>
  %get_122 = sdql.lookup_dictionary %fieldnode_123 [%i : i32] : dictionary<i32, memref<10xi8>> -> memref<10xi8>
  %fieldnode_125 = sdql.access_record %part "p_retailprice" : record<"p_partkey": dictionary<i32, i32>, "p_name": dictionary<i32, memref<55xi8>>, "p_mfgr": dictionary<i32, memref<25xi8>>, "p_brand": dictionary<i32, memref<10xi8>>, "p_type": dictionary<i32, memref<25xi8>>, "p_size": dictionary<i32, i32>, "p_container": dictionary<i32, memref<10xi8>>, "p_retailprice": dictionary<i32, f64>, "p_comment": dictionary<i32, memref<23xi8>>, "size": i32> -> dictionary<i32, f64>
  %get_124 = sdql.lookup_dictionary %fieldnode_125 [%i : i32] : dictionary<i32, f64> -> f64
  %fieldnode_127 = sdql.access_record %part "p_comment" : record<"p_partkey": dictionary<i32, i32>, "p_name": dictionary<i32, memref<55xi8>>, "p_mfgr": dictionary<i32, memref<25xi8>>, "p_brand": dictionary<i32, memref<10xi8>>, "p_type": dictionary<i32, memref<25xi8>>, "p_size": dictionary<i32, i32>, "p_container": dictionary<i32, memref<10xi8>>, "p_retailprice": dictionary<i32, f64>, "p_comment": dictionary<i32, memref<23xi8>>, "size": i32> -> dictionary<i32, memref<23xi8>>
  %get_126 = sdql.lookup_dictionary %fieldnode_127 [%i : i32] : dictionary<i32, memref<23xi8>> -> memref<23xi8>
  %p = sdql.create_record {fields = ["p_partkey", "p_name", "p_mfgr", "p_brand", "p_type", "p_size", "p_container", "p_retailprice", "p_comment"]} %get_110, %get_112, %get_114, %get_116, %get_118, %get_120, %get_122, %get_124, %get_126 : i32, memref<55xi8>, memref<25xi8>, memref<10xi8>, memref<25xi8>, i32, memref<10xi8>, f64, memref<23xi8> -> record<"p_partkey": i32, "p_name": memref<55xi8>, "p_mfgr": memref<25xi8>, "p_brand": memref<10xi8>, "p_type": memref<25xi8>, "p_size": i32, "p_container": memref<10xi8>, "p_retailprice": f64, "p_comment": memref<23xi8>>
  %fieldnode_128 = sdql.access_record %p "p_type" : record<"p_partkey": i32, "p_name": memref<55xi8>, "p_mfgr": memref<25xi8>, "p_brand": memref<10xi8>, "p_type": memref<25xi8>, "p_size": i32, "p_container": memref<10xi8>, "p_retailprice": f64, "p_comment": memref<23xi8>> -> memref<25xi8>
  %cmp_129 = sdql.cmp %fieldnode_128, %steel : memref<25xi8>, memref<22xi8> -> i1
  %if_130 = "scf.if"(%cmp_129) ({
    %fieldnode_133 = sdql.access_record %p "p_partkey" : record<"p_partkey": i32, "p_name": memref<55xi8>, "p_mfgr": memref<25xi8>, "p_brand": memref<10xi8>, "p_type": memref<25xi8>, "p_size": i32, "p_container": memref<10xi8>, "p_retailprice": f64, "p_comment": memref<23xi8>> -> i32
    %uniq_132 = sdql.unique %fieldnode_133 : i32 -> i32
    %fieldnode_134 = sdql.access_record %p "p_partkey" : record<"p_partkey": i32, "p_name": memref<55xi8>, "p_mfgr": memref<25xi8>, "p_brand": memref<10xi8>, "p_type": memref<25xi8>, "p_size": i32, "p_container": memref<10xi8>, "p_retailprice": f64, "p_comment": memref<23xi8>> -> i32
    %recnode_135 = sdql.create_record {fields = ["_"]} %fieldnode_134 : i32 -> record<"_": i32>
    %dict_131 = sdql.create_dictionary %uniq_132, %recnode_135 : i32, record<"_" : i32> -> dictionary<i32, record<"_" : i32>>
    "scf.yield"(%dict_131) : (dictionary<i32, record<"_" : i32>>) -> ()
  }, {
    %dict_136 = sdql.empty_dictionary : dictionary<i32, record<"_" : i32>>
    "scf.yield"(%dict_136) : (dictionary<i32, record<"_" : i32>>) -> ()
  }) : (i1) -> dictionary<i32, record<"_" : i32>>
  sdql.yield %if_130 : dictionary<i32, record<"_" : i32>>
}
%fieldnode_137 = sdql.access_record %orders "size" : record<"o_orderkey": dictionary<i32, i32>, "o_custkey": dictionary<i32, i32>, "o_orderstatus": dictionary<i32, memref<1xi8>>, "o_totalprice": dictionary<i32, f64>, "o_orderdate": dictionary<i32, i32>, "o_orderpriority": dictionary<i32, memref<15xi8>>, "o_clerk": dictionary<i32, memref<15xi8>>, "o_shippriority": dictionary<i32, i32>, "o_comment": dictionary<i32, memref<79xi8>>, "size": i32> -> i32
%rangenode_138 = "func.call"(%fieldnode_137) <{callee = @range_builtin}> : (i32) -> dictionary<i32, i32>
%o_h = sdql.sum %rangenode_138 : dictionary<i32, i32> -> dictionary<i32, record<"_" : i32, "_" : i32>> {
^bb_139(%i: i32, %_: i32):
  %fieldnode_141 = sdql.access_record %orders "o_orderkey" : record<"o_orderkey": dictionary<i32, i32>, "o_custkey": dictionary<i32, i32>, "o_orderstatus": dictionary<i32, memref<1xi8>>, "o_totalprice": dictionary<i32, f64>, "o_orderdate": dictionary<i32, i32>, "o_orderpriority": dictionary<i32, memref<15xi8>>, "o_clerk": dictionary<i32, memref<15xi8>>, "o_shippriority": dictionary<i32, i32>, "o_comment": dictionary<i32, memref<79xi8>>, "size": i32> -> dictionary<i32, i32>
  %get_140 = sdql.lookup_dictionary %fieldnode_141 [%i : i32] : dictionary<i32, i32> -> i32
  %fieldnode_143 = sdql.access_record %orders "o_custkey" : record<"o_orderkey": dictionary<i32, i32>, "o_custkey": dictionary<i32, i32>, "o_orderstatus": dictionary<i32, memref<1xi8>>, "o_totalprice": dictionary<i32, f64>, "o_orderdate": dictionary<i32, i32>, "o_orderpriority": dictionary<i32, memref<15xi8>>, "o_clerk": dictionary<i32, memref<15xi8>>, "o_shippriority": dictionary<i32, i32>, "o_comment": dictionary<i32, memref<79xi8>>, "size": i32> -> dictionary<i32, i32>
  %get_142 = sdql.lookup_dictionary %fieldnode_143 [%i : i32] : dictionary<i32, i32> -> i32
  %fieldnode_145 = sdql.access_record %orders "o_orderstatus" : record<"o_orderkey": dictionary<i32, i32>, "o_custkey": dictionary<i32, i32>, "o_orderstatus": dictionary<i32, memref<1xi8>>, "o_totalprice": dictionary<i32, f64>, "o_orderdate": dictionary<i32, i32>, "o_orderpriority": dictionary<i32, memref<15xi8>>, "o_clerk": dictionary<i32, memref<15xi8>>, "o_shippriority": dictionary<i32, i32>, "o_comment": dictionary<i32, memref<79xi8>>, "size": i32> -> dictionary<i32, memref<1xi8>>
  %get_144 = sdql.lookup_dictionary %fieldnode_145 [%i : i32] : dictionary<i32, memref<1xi8>> -> memref<1xi8>
  %fieldnode_147 = sdql.access_record %orders "o_totalprice" : record<"o_orderkey": dictionary<i32, i32>, "o_custkey": dictionary<i32, i32>, "o_orderstatus": dictionary<i32, memref<1xi8>>, "o_totalprice": dictionary<i32, f64>, "o_orderdate": dictionary<i32, i32>, "o_orderpriority": dictionary<i32, memref<15xi8>>, "o_clerk": dictionary<i32, memref<15xi8>>, "o_shippriority": dictionary<i32, i32>, "o_comment": dictionary<i32, memref<79xi8>>, "size": i32> -> dictionary<i32, f64>
  %get_146 = sdql.lookup_dictionary %fieldnode_147 [%i : i32] : dictionary<i32, f64> -> f64
  %fieldnode_149 = sdql.access_record %orders "o_orderdate" : record<"o_orderkey": dictionary<i32, i32>, "o_custkey": dictionary<i32, i32>, "o_orderstatus": dictionary<i32, memref<1xi8>>, "o_totalprice": dictionary<i32, f64>, "o_orderdate": dictionary<i32, i32>, "o_orderpriority": dictionary<i32, memref<15xi8>>, "o_clerk": dictionary<i32, memref<15xi8>>, "o_shippriority": dictionary<i32, i32>, "o_comment": dictionary<i32, memref<79xi8>>, "size": i32> -> dictionary<i32, i32>
  %get_148 = sdql.lookup_dictionary %fieldnode_149 [%i : i32] : dictionary<i32, i32> -> i32
  %fieldnode_151 = sdql.access_record %orders "o_orderpriority" : record<"o_orderkey": dictionary<i32, i32>, "o_custkey": dictionary<i32, i32>, "o_orderstatus": dictionary<i32, memref<1xi8>>, "o_totalprice": dictionary<i32, f64>, "o_orderdate": dictionary<i32, i32>, "o_orderpriority": dictionary<i32, memref<15xi8>>, "o_clerk": dictionary<i32, memref<15xi8>>, "o_shippriority": dictionary<i32, i32>, "o_comment": dictionary<i32, memref<79xi8>>, "size": i32> -> dictionary<i32, memref<15xi8>>
  %get_150 = sdql.lookup_dictionary %fieldnode_151 [%i : i32] : dictionary<i32, memref<15xi8>> -> memref<15xi8>
  %fieldnode_153 = sdql.access_record %orders "o_clerk" : record<"o_orderkey": dictionary<i32, i32>, "o_custkey": dictionary<i32, i32>, "o_orderstatus": dictionary<i32, memref<1xi8>>, "o_totalprice": dictionary<i32, f64>, "o_orderdate": dictionary<i32, i32>, "o_orderpriority": dictionary<i32, memref<15xi8>>, "o_clerk": dictionary<i32, memref<15xi8>>, "o_shippriority": dictionary<i32, i32>, "o_comment": dictionary<i32, memref<79xi8>>, "size": i32> -> dictionary<i32, memref<15xi8>>
  %get_152 = sdql.lookup_dictionary %fieldnode_153 [%i : i32] : dictionary<i32, memref<15xi8>> -> memref<15xi8>
  %fieldnode_155 = sdql.access_record %orders "o_shippriority" : record<"o_orderkey": dictionary<i32, i32>, "o_custkey": dictionary<i32, i32>, "o_orderstatus": dictionary<i32, memref<1xi8>>, "o_totalprice": dictionary<i32, f64>, "o_orderdate": dictionary<i32, i32>, "o_orderpriority": dictionary<i32, memref<15xi8>>, "o_clerk": dictionary<i32, memref<15xi8>>, "o_shippriority": dictionary<i32, i32>, "o_comment": dictionary<i32, memref<79xi8>>, "size": i32> -> dictionary<i32, i32>
  %get_154 = sdql.lookup_dictionary %fieldnode_155 [%i : i32] : dictionary<i32, i32> -> i32
  %fieldnode_157 = sdql.access_record %orders "o_comment" : record<"o_orderkey": dictionary<i32, i32>, "o_custkey": dictionary<i32, i32>, "o_orderstatus": dictionary<i32, memref<1xi8>>, "o_totalprice": dictionary<i32, f64>, "o_orderdate": dictionary<i32, i32>, "o_orderpriority": dictionary<i32, memref<15xi8>>, "o_clerk": dictionary<i32, memref<15xi8>>, "o_shippriority": dictionary<i32, i32>, "o_comment": dictionary<i32, memref<79xi8>>, "size": i32> -> dictionary<i32, memref<79xi8>>
  %get_156 = sdql.lookup_dictionary %fieldnode_157 [%i : i32] : dictionary<i32, memref<79xi8>> -> memref<79xi8>
  %o = sdql.create_record {fields = ["o_orderkey", "o_custkey", "o_orderstatus", "o_totalprice", "o_orderdate", "o_orderpriority", "o_clerk", "o_shippriority", "o_comment"]} %get_140, %get_142, %get_144, %get_146, %get_148, %get_150, %get_152, %get_154, %get_156 : i32, i32, memref<1xi8>, f64, i32, memref<15xi8>, memref<15xi8>, i32, memref<79xi8> -> record<"o_orderkey": i32, "o_custkey": i32, "o_orderstatus": memref<1xi8>, "o_totalprice": f64, "o_orderdate": i32, "o_orderpriority": memref<15xi8>, "o_clerk": memref<15xi8>, "o_shippriority": i32, "o_comment": memref<79xi8>>
  %constda_158 = "arith.constant"() <{value = 19950101 : i32}> : () -> i32
  %fieldnode_159 = sdql.access_record %o "o_orderdate" : record<"o_orderkey": i32, "o_custkey": i32, "o_orderstatus": memref<1xi8>, "o_totalprice": f64, "o_orderdate": i32, "o_orderpriority": memref<15xi8>, "o_clerk": memref<15xi8>, "o_shippriority": i32, "o_comment": memref<79xi8>> -> i32
  %cmpi_160 = "arith.cmpi"(%constda_158, %fieldnode_159) <{predicate = 3}> : (i32, i32) -> i1
  %if_161 = "scf.if"(%cmpi_160) ({
    %fieldnode_162 = sdql.access_record %o "o_orderdate" : record<"o_orderkey": i32, "o_custkey": i32, "o_orderstatus": memref<1xi8>, "o_totalprice": f64, "o_orderdate": i32, "o_orderpriority": memref<15xi8>, "o_clerk": memref<15xi8>, "o_shippriority": i32, "o_comment": memref<79xi8>> -> i32
    %constda_163 = "arith.constant"() <{value = 19961231 : i32}> : () -> i32
    %cmpi_164 = "arith.cmpi"(%fieldnode_162, %constda_163) <{predicate = 3}> : (i32, i32) -> i1
    "scf.yield"(%cmpi_164) : (i1) -> ()
  }, {
    %constv_165 = "arith.constant"() <{value = 0 : i1}> : () -> i1
    "scf.yield"(%constv_165) : (i1) -> ()
  }) : (i1) -> i1
  %if_166 = "scf.if"(%if_161) ({
    %fieldnode_169 = sdql.access_record %o "o_orderkey" : record<"o_orderkey": i32, "o_custkey": i32, "o_orderstatus": memref<1xi8>, "o_totalprice": f64, "o_orderdate": i32, "o_orderpriority": memref<15xi8>, "o_clerk": memref<15xi8>, "o_shippriority": i32, "o_comment": memref<79xi8>> -> i32
    %uniq_168 = sdql.unique %fieldnode_169 : i32 -> i32
    %fieldnode_170 = sdql.access_record %o "o_custkey" : record<"o_orderkey": i32, "o_custkey": i32, "o_orderstatus": memref<1xi8>, "o_totalprice": f64, "o_orderdate": i32, "o_orderpriority": memref<15xi8>, "o_clerk": memref<15xi8>, "o_shippriority": i32, "o_comment": memref<79xi8>> -> i32
    %fieldnode_171 = sdql.access_record %o "o_orderdate" : record<"o_orderkey": i32, "o_custkey": i32, "o_orderstatus": memref<1xi8>, "o_totalprice": f64, "o_orderdate": i32, "o_orderpriority": memref<15xi8>, "o_clerk": memref<15xi8>, "o_shippriority": i32, "o_comment": memref<79xi8>> -> i32
    %recnode_172 = sdql.create_record {fields = ["_", "_"]} %fieldnode_170, %fieldnode_171 : i32, i32 -> record<"_": i32, "_": i32>
    %dict_167 = sdql.create_dictionary %uniq_168, %recnode_172 : i32, record<"_" : i32, "_" : i32> -> dictionary<i32, record<"_" : i32, "_" : i32>>
    "scf.yield"(%dict_167) : (dictionary<i32, record<"_" : i32, "_" : i32>>) -> ()
  }, {
    %dict_173 = sdql.empty_dictionary : dictionary<i32, record<"_" : i32, "_" : i32>>
    "scf.yield"(%dict_173) : (dictionary<i32, record<"_" : i32, "_" : i32>>) -> ()
  }) : (i1) -> dictionary<i32, record<"_" : i32, "_" : i32>>
  sdql.yield %if_166 : dictionary<i32, record<"_" : i32, "_" : i32>>
}
%fieldnode_174 = sdql.access_record %lineitem "size" : record<"l_orderkey": dictionary<i32, i32>, "l_partkey": dictionary<i32, i32>, "l_suppkey": dictionary<i32, i32>, "l_linenumber": dictionary<i32, i32>, "l_quantity": dictionary<i32, f64>, "l_extendedprice": dictionary<i32, f64>, "l_discount": dictionary<i32, f64>, "l_tax": dictionary<i32, f64>, "l_returnflag": dictionary<i32, memref<1xi8>>, "l_linestatus": dictionary<i32, memref<1xi8>>, "l_shipdate": dictionary<i32, i32>, "l_commitdate": dictionary<i32, i32>, "l_receiptdate": dictionary<i32, i32>, "l_shipinstruct": dictionary<i32, memref<25xi8>>, "l_shipmode": dictionary<i32, memref<10xi8>>, "l_comment": dictionary<i32, memref<44xi8>>, "size": i32> -> i32
%rangenode_175 = "func.call"(%fieldnode_174) <{callee = @range_builtin}> : (i32) -> dictionary<i32, i32>
%l_h = sdql.sum %rangenode_175 : dictionary<i32, i32> -> dictionary<i32, record<"_" : f64, "_" : f64>> {
^bb_176(%i: i32, %_: i32):
  %fieldnode_178 = sdql.access_record %lineitem "l_orderkey" : record<"l_orderkey": dictionary<i32, i32>, "l_partkey": dictionary<i32, i32>, "l_suppkey": dictionary<i32, i32>, "l_linenumber": dictionary<i32, i32>, "l_quantity": dictionary<i32, f64>, "l_extendedprice": dictionary<i32, f64>, "l_discount": dictionary<i32, f64>, "l_tax": dictionary<i32, f64>, "l_returnflag": dictionary<i32, memref<1xi8>>, "l_linestatus": dictionary<i32, memref<1xi8>>, "l_shipdate": dictionary<i32, i32>, "l_commitdate": dictionary<i32, i32>, "l_receiptdate": dictionary<i32, i32>, "l_shipinstruct": dictionary<i32, memref<25xi8>>, "l_shipmode": dictionary<i32, memref<10xi8>>, "l_comment": dictionary<i32, memref<44xi8>>, "size": i32> -> dictionary<i32, i32>
  %get_177 = sdql.lookup_dictionary %fieldnode_178 [%i : i32] : dictionary<i32, i32> -> i32
  %fieldnode_180 = sdql.access_record %lineitem "l_partkey" : record<"l_orderkey": dictionary<i32, i32>, "l_partkey": dictionary<i32, i32>, "l_suppkey": dictionary<i32, i32>, "l_linenumber": dictionary<i32, i32>, "l_quantity": dictionary<i32, f64>, "l_extendedprice": dictionary<i32, f64>, "l_discount": dictionary<i32, f64>, "l_tax": dictionary<i32, f64>, "l_returnflag": dictionary<i32, memref<1xi8>>, "l_linestatus": dictionary<i32, memref<1xi8>>, "l_shipdate": dictionary<i32, i32>, "l_commitdate": dictionary<i32, i32>, "l_receiptdate": dictionary<i32, i32>, "l_shipinstruct": dictionary<i32, memref<25xi8>>, "l_shipmode": dictionary<i32, memref<10xi8>>, "l_comment": dictionary<i32, memref<44xi8>>, "size": i32> -> dictionary<i32, i32>
  %get_179 = sdql.lookup_dictionary %fieldnode_180 [%i : i32] : dictionary<i32, i32> -> i32
  %fieldnode_182 = sdql.access_record %lineitem "l_suppkey" : record<"l_orderkey": dictionary<i32, i32>, "l_partkey": dictionary<i32, i32>, "l_suppkey": dictionary<i32, i32>, "l_linenumber": dictionary<i32, i32>, "l_quantity": dictionary<i32, f64>, "l_extendedprice": dictionary<i32, f64>, "l_discount": dictionary<i32, f64>, "l_tax": dictionary<i32, f64>, "l_returnflag": dictionary<i32, memref<1xi8>>, "l_linestatus": dictionary<i32, memref<1xi8>>, "l_shipdate": dictionary<i32, i32>, "l_commitdate": dictionary<i32, i32>, "l_receiptdate": dictionary<i32, i32>, "l_shipinstruct": dictionary<i32, memref<25xi8>>, "l_shipmode": dictionary<i32, memref<10xi8>>, "l_comment": dictionary<i32, memref<44xi8>>, "size": i32> -> dictionary<i32, i32>
  %get_181 = sdql.lookup_dictionary %fieldnode_182 [%i : i32] : dictionary<i32, i32> -> i32
  %fieldnode_184 = sdql.access_record %lineitem "l_linenumber" : record<"l_orderkey": dictionary<i32, i32>, "l_partkey": dictionary<i32, i32>, "l_suppkey": dictionary<i32, i32>, "l_linenumber": dictionary<i32, i32>, "l_quantity": dictionary<i32, f64>, "l_extendedprice": dictionary<i32, f64>, "l_discount": dictionary<i32, f64>, "l_tax": dictionary<i32, f64>, "l_returnflag": dictionary<i32, memref<1xi8>>, "l_linestatus": dictionary<i32, memref<1xi8>>, "l_shipdate": dictionary<i32, i32>, "l_commitdate": dictionary<i32, i32>, "l_receiptdate": dictionary<i32, i32>, "l_shipinstruct": dictionary<i32, memref<25xi8>>, "l_shipmode": dictionary<i32, memref<10xi8>>, "l_comment": dictionary<i32, memref<44xi8>>, "size": i32> -> dictionary<i32, i32>
  %get_183 = sdql.lookup_dictionary %fieldnode_184 [%i : i32] : dictionary<i32, i32> -> i32
  %fieldnode_186 = sdql.access_record %lineitem "l_quantity" : record<"l_orderkey": dictionary<i32, i32>, "l_partkey": dictionary<i32, i32>, "l_suppkey": dictionary<i32, i32>, "l_linenumber": dictionary<i32, i32>, "l_quantity": dictionary<i32, f64>, "l_extendedprice": dictionary<i32, f64>, "l_discount": dictionary<i32, f64>, "l_tax": dictionary<i32, f64>, "l_returnflag": dictionary<i32, memref<1xi8>>, "l_linestatus": dictionary<i32, memref<1xi8>>, "l_shipdate": dictionary<i32, i32>, "l_commitdate": dictionary<i32, i32>, "l_receiptdate": dictionary<i32, i32>, "l_shipinstruct": dictionary<i32, memref<25xi8>>, "l_shipmode": dictionary<i32, memref<10xi8>>, "l_comment": dictionary<i32, memref<44xi8>>, "size": i32> -> dictionary<i32, f64>
  %get_185 = sdql.lookup_dictionary %fieldnode_186 [%i : i32] : dictionary<i32, f64> -> f64
  %fieldnode_188 = sdql.access_record %lineitem "l_extendedprice" : record<"l_orderkey": dictionary<i32, i32>, "l_partkey": dictionary<i32, i32>, "l_suppkey": dictionary<i32, i32>, "l_linenumber": dictionary<i32, i32>, "l_quantity": dictionary<i32, f64>, "l_extendedprice": dictionary<i32, f64>, "l_discount": dictionary<i32, f64>, "l_tax": dictionary<i32, f64>, "l_returnflag": dictionary<i32, memref<1xi8>>, "l_linestatus": dictionary<i32, memref<1xi8>>, "l_shipdate": dictionary<i32, i32>, "l_commitdate": dictionary<i32, i32>, "l_receiptdate": dictionary<i32, i32>, "l_shipinstruct": dictionary<i32, memref<25xi8>>, "l_shipmode": dictionary<i32, memref<10xi8>>, "l_comment": dictionary<i32, memref<44xi8>>, "size": i32> -> dictionary<i32, f64>
  %get_187 = sdql.lookup_dictionary %fieldnode_188 [%i : i32] : dictionary<i32, f64> -> f64
  %fieldnode_190 = sdql.access_record %lineitem "l_discount" : record<"l_orderkey": dictionary<i32, i32>, "l_partkey": dictionary<i32, i32>, "l_suppkey": dictionary<i32, i32>, "l_linenumber": dictionary<i32, i32>, "l_quantity": dictionary<i32, f64>, "l_extendedprice": dictionary<i32, f64>, "l_discount": dictionary<i32, f64>, "l_tax": dictionary<i32, f64>, "l_returnflag": dictionary<i32, memref<1xi8>>, "l_linestatus": dictionary<i32, memref<1xi8>>, "l_shipdate": dictionary<i32, i32>, "l_commitdate": dictionary<i32, i32>, "l_receiptdate": dictionary<i32, i32>, "l_shipinstruct": dictionary<i32, memref<25xi8>>, "l_shipmode": dictionary<i32, memref<10xi8>>, "l_comment": dictionary<i32, memref<44xi8>>, "size": i32> -> dictionary<i32, f64>
  %get_189 = sdql.lookup_dictionary %fieldnode_190 [%i : i32] : dictionary<i32, f64> -> f64
  %fieldnode_192 = sdql.access_record %lineitem "l_tax" : record<"l_orderkey": dictionary<i32, i32>, "l_partkey": dictionary<i32, i32>, "l_suppkey": dictionary<i32, i32>, "l_linenumber": dictionary<i32, i32>, "l_quantity": dictionary<i32, f64>, "l_extendedprice": dictionary<i32, f64>, "l_discount": dictionary<i32, f64>, "l_tax": dictionary<i32, f64>, "l_returnflag": dictionary<i32, memref<1xi8>>, "l_linestatus": dictionary<i32, memref<1xi8>>, "l_shipdate": dictionary<i32, i32>, "l_commitdate": dictionary<i32, i32>, "l_receiptdate": dictionary<i32, i32>, "l_shipinstruct": dictionary<i32, memref<25xi8>>, "l_shipmode": dictionary<i32, memref<10xi8>>, "l_comment": dictionary<i32, memref<44xi8>>, "size": i32> -> dictionary<i32, f64>
  %get_191 = sdql.lookup_dictionary %fieldnode_192 [%i : i32] : dictionary<i32, f64> -> f64
  %fieldnode_194 = sdql.access_record %lineitem "l_returnflag" : record<"l_orderkey": dictionary<i32, i32>, "l_partkey": dictionary<i32, i32>, "l_suppkey": dictionary<i32, i32>, "l_linenumber": dictionary<i32, i32>, "l_quantity": dictionary<i32, f64>, "l_extendedprice": dictionary<i32, f64>, "l_discount": dictionary<i32, f64>, "l_tax": dictionary<i32, f64>, "l_returnflag": dictionary<i32, memref<1xi8>>, "l_linestatus": dictionary<i32, memref<1xi8>>, "l_shipdate": dictionary<i32, i32>, "l_commitdate": dictionary<i32, i32>, "l_receiptdate": dictionary<i32, i32>, "l_shipinstruct": dictionary<i32, memref<25xi8>>, "l_shipmode": dictionary<i32, memref<10xi8>>, "l_comment": dictionary<i32, memref<44xi8>>, "size": i32> -> dictionary<i32, memref<1xi8>>
  %get_193 = sdql.lookup_dictionary %fieldnode_194 [%i : i32] : dictionary<i32, memref<1xi8>> -> memref<1xi8>
  %fieldnode_196 = sdql.access_record %lineitem "l_linestatus" : record<"l_orderkey": dictionary<i32, i32>, "l_partkey": dictionary<i32, i32>, "l_suppkey": dictionary<i32, i32>, "l_linenumber": dictionary<i32, i32>, "l_quantity": dictionary<i32, f64>, "l_extendedprice": dictionary<i32, f64>, "l_discount": dictionary<i32, f64>, "l_tax": dictionary<i32, f64>, "l_returnflag": dictionary<i32, memref<1xi8>>, "l_linestatus": dictionary<i32, memref<1xi8>>, "l_shipdate": dictionary<i32, i32>, "l_commitdate": dictionary<i32, i32>, "l_receiptdate": dictionary<i32, i32>, "l_shipinstruct": dictionary<i32, memref<25xi8>>, "l_shipmode": dictionary<i32, memref<10xi8>>, "l_comment": dictionary<i32, memref<44xi8>>, "size": i32> -> dictionary<i32, memref<1xi8>>
  %get_195 = sdql.lookup_dictionary %fieldnode_196 [%i : i32] : dictionary<i32, memref<1xi8>> -> memref<1xi8>
  %fieldnode_198 = sdql.access_record %lineitem "l_shipdate" : record<"l_orderkey": dictionary<i32, i32>, "l_partkey": dictionary<i32, i32>, "l_suppkey": dictionary<i32, i32>, "l_linenumber": dictionary<i32, i32>, "l_quantity": dictionary<i32, f64>, "l_extendedprice": dictionary<i32, f64>, "l_discount": dictionary<i32, f64>, "l_tax": dictionary<i32, f64>, "l_returnflag": dictionary<i32, memref<1xi8>>, "l_linestatus": dictionary<i32, memref<1xi8>>, "l_shipdate": dictionary<i32, i32>, "l_commitdate": dictionary<i32, i32>, "l_receiptdate": dictionary<i32, i32>, "l_shipinstruct": dictionary<i32, memref<25xi8>>, "l_shipmode": dictionary<i32, memref<10xi8>>, "l_comment": dictionary<i32, memref<44xi8>>, "size": i32> -> dictionary<i32, i32>
  %get_197 = sdql.lookup_dictionary %fieldnode_198 [%i : i32] : dictionary<i32, i32> -> i32
  %fieldnode_200 = sdql.access_record %lineitem "l_commitdate" : record<"l_orderkey": dictionary<i32, i32>, "l_partkey": dictionary<i32, i32>, "l_suppkey": dictionary<i32, i32>, "l_linenumber": dictionary<i32, i32>, "l_quantity": dictionary<i32, f64>, "l_extendedprice": dictionary<i32, f64>, "l_discount": dictionary<i32, f64>, "l_tax": dictionary<i32, f64>, "l_returnflag": dictionary<i32, memref<1xi8>>, "l_linestatus": dictionary<i32, memref<1xi8>>, "l_shipdate": dictionary<i32, i32>, "l_commitdate": dictionary<i32, i32>, "l_receiptdate": dictionary<i32, i32>, "l_shipinstruct": dictionary<i32, memref<25xi8>>, "l_shipmode": dictionary<i32, memref<10xi8>>, "l_comment": dictionary<i32, memref<44xi8>>, "size": i32> -> dictionary<i32, i32>
  %get_199 = sdql.lookup_dictionary %fieldnode_200 [%i : i32] : dictionary<i32, i32> -> i32
  %fieldnode_202 = sdql.access_record %lineitem "l_receiptdate" : record<"l_orderkey": dictionary<i32, i32>, "l_partkey": dictionary<i32, i32>, "l_suppkey": dictionary<i32, i32>, "l_linenumber": dictionary<i32, i32>, "l_quantity": dictionary<i32, f64>, "l_extendedprice": dictionary<i32, f64>, "l_discount": dictionary<i32, f64>, "l_tax": dictionary<i32, f64>, "l_returnflag": dictionary<i32, memref<1xi8>>, "l_linestatus": dictionary<i32, memref<1xi8>>, "l_shipdate": dictionary<i32, i32>, "l_commitdate": dictionary<i32, i32>, "l_receiptdate": dictionary<i32, i32>, "l_shipinstruct": dictionary<i32, memref<25xi8>>, "l_shipmode": dictionary<i32, memref<10xi8>>, "l_comment": dictionary<i32, memref<44xi8>>, "size": i32> -> dictionary<i32, i32>
  %get_201 = sdql.lookup_dictionary %fieldnode_202 [%i : i32] : dictionary<i32, i32> -> i32
  %fieldnode_204 = sdql.access_record %lineitem "l_shipinstruct" : record<"l_orderkey": dictionary<i32, i32>, "l_partkey": dictionary<i32, i32>, "l_suppkey": dictionary<i32, i32>, "l_linenumber": dictionary<i32, i32>, "l_quantity": dictionary<i32, f64>, "l_extendedprice": dictionary<i32, f64>, "l_discount": dictionary<i32, f64>, "l_tax": dictionary<i32, f64>, "l_returnflag": dictionary<i32, memref<1xi8>>, "l_linestatus": dictionary<i32, memref<1xi8>>, "l_shipdate": dictionary<i32, i32>, "l_commitdate": dictionary<i32, i32>, "l_receiptdate": dictionary<i32, i32>, "l_shipinstruct": dictionary<i32, memref<25xi8>>, "l_shipmode": dictionary<i32, memref<10xi8>>, "l_comment": dictionary<i32, memref<44xi8>>, "size": i32> -> dictionary<i32, memref<25xi8>>
  %get_203 = sdql.lookup_dictionary %fieldnode_204 [%i : i32] : dictionary<i32, memref<25xi8>> -> memref<25xi8>
  %fieldnode_206 = sdql.access_record %lineitem "l_shipmode" : record<"l_orderkey": dictionary<i32, i32>, "l_partkey": dictionary<i32, i32>, "l_suppkey": dictionary<i32, i32>, "l_linenumber": dictionary<i32, i32>, "l_quantity": dictionary<i32, f64>, "l_extendedprice": dictionary<i32, f64>, "l_discount": dictionary<i32, f64>, "l_tax": dictionary<i32, f64>, "l_returnflag": dictionary<i32, memref<1xi8>>, "l_linestatus": dictionary<i32, memref<1xi8>>, "l_shipdate": dictionary<i32, i32>, "l_commitdate": dictionary<i32, i32>, "l_receiptdate": dictionary<i32, i32>, "l_shipinstruct": dictionary<i32, memref<25xi8>>, "l_shipmode": dictionary<i32, memref<10xi8>>, "l_comment": dictionary<i32, memref<44xi8>>, "size": i32> -> dictionary<i32, memref<10xi8>>
  %get_205 = sdql.lookup_dictionary %fieldnode_206 [%i : i32] : dictionary<i32, memref<10xi8>> -> memref<10xi8>
  %fieldnode_208 = sdql.access_record %lineitem "l_comment" : record<"l_orderkey": dictionary<i32, i32>, "l_partkey": dictionary<i32, i32>, "l_suppkey": dictionary<i32, i32>, "l_linenumber": dictionary<i32, i32>, "l_quantity": dictionary<i32, f64>, "l_extendedprice": dictionary<i32, f64>, "l_discount": dictionary<i32, f64>, "l_tax": dictionary<i32, f64>, "l_returnflag": dictionary<i32, memref<1xi8>>, "l_linestatus": dictionary<i32, memref<1xi8>>, "l_shipdate": dictionary<i32, i32>, "l_commitdate": dictionary<i32, i32>, "l_receiptdate": dictionary<i32, i32>, "l_shipinstruct": dictionary<i32, memref<25xi8>>, "l_shipmode": dictionary<i32, memref<10xi8>>, "l_comment": dictionary<i32, memref<44xi8>>, "size": i32> -> dictionary<i32, memref<44xi8>>
  %get_207 = sdql.lookup_dictionary %fieldnode_208 [%i : i32] : dictionary<i32, memref<44xi8>> -> memref<44xi8>
  %l = sdql.create_record {fields = ["l_orderkey", "l_partkey", "l_suppkey", "l_linenumber", "l_quantity", "l_extendedprice", "l_discount", "l_tax", "l_returnflag", "l_linestatus", "l_shipdate", "l_commitdate", "l_receiptdate", "l_shipinstruct", "l_shipmode", "l_comment"]} %get_177, %get_179, %get_181, %get_183, %get_185, %get_187, %get_189, %get_191, %get_193, %get_195, %get_197, %get_199, %get_201, %get_203, %get_205, %get_207 : i32, i32, i32, i32, f64, f64, f64, f64, memref<1xi8>, memref<1xi8>, i32, i32, i32, memref<25xi8>, memref<10xi8>, memref<44xi8> -> record<"l_orderkey": i32, "l_partkey": i32, "l_suppkey": i32, "l_linenumber": i32, "l_quantity": f64, "l_extendedprice": f64, "l_discount": f64, "l_tax": f64, "l_returnflag": memref<1xi8>, "l_linestatus": memref<1xi8>, "l_shipdate": i32, "l_commitdate": i32, "l_receiptdate": i32, "l_shipinstruct": memref<25xi8>, "l_shipmode": memref<10xi8>, "l_comment": memref<44xi8>>
  %fieldnode_212 = sdql.access_record %l "l_partkey" : record<"l_orderkey": i32, "l_partkey": i32, "l_suppkey": i32, "l_linenumber": i32, "l_quantity": f64, "l_extendedprice": f64, "l_discount": f64, "l_tax": f64, "l_returnflag": memref<1xi8>, "l_linestatus": memref<1xi8>, "l_shipdate": i32, "l_commitdate": i32, "l_receiptdate": i32, "l_shipinstruct": memref<25xi8>, "l_shipmode": memref<10xi8>, "l_comment": memref<44xi8>> -> i32
  %get_211 = sdql.lookup_dictionary %p_h [%fieldnode_212 : i32] : dictionary<i32, record<"_" : i32>> -> record<"_" : i32>
  %dict_213 = sdql.empty_dictionary : record<"_" : i32>
  %cmp_214 = sdql.cmp %get_211, %dict_213 : record<"_" : i32>, record<"_" : i32> -> i1
  %zero_209 = "arith.constant"() <{value = 0 : i1}> : () -> i1
  %neg_210 = "arith.cmpi"(%cmp_214, %zero_209) <{predicate = 0}> : (i1, i1) -> i1
  %if_215 = "scf.if"(%neg_210) ({
    %fieldnode_219 = sdql.access_record %l "l_orderkey" : record<"l_orderkey": i32, "l_partkey": i32, "l_suppkey": i32, "l_linenumber": i32, "l_quantity": f64, "l_extendedprice": f64, "l_discount": f64, "l_tax": f64, "l_returnflag": memref<1xi8>, "l_linestatus": memref<1xi8>, "l_shipdate": i32, "l_commitdate": i32, "l_receiptdate": i32, "l_shipinstruct": memref<25xi8>, "l_shipmode": memref<10xi8>, "l_comment": memref<44xi8>> -> i32
    %get_218 = sdql.lookup_dictionary %o_h [%fieldnode_219 : i32] : dictionary<i32, record<"_" : i32, "_" : i32>> -> record<"_" : i32, "_" : i32>
    %dict_220 = sdql.empty_dictionary : record<"_" : i32, "_" : i32>
    %cmp_221 = sdql.cmp %get_218, %dict_220 : record<"_" : i32, "_" : i32>, record<"_" : i32, "_" : i32> -> i1
    %zero_216 = "arith.constant"() <{value = 0 : i1}> : () -> i1
    %neg_217 = "arith.cmpi"(%cmp_221, %zero_216) <{predicate = 0}> : (i1, i1) -> i1
    "scf.yield"(%neg_217) : (i1) -> ()
  }, {
    %constv_222 = "arith.constant"() <{value = 0 : i1}> : () -> i1
    "scf.yield"(%constv_222) : (i1) -> ()
  }) : (i1) -> i1
  %if_223 = "scf.if"(%if_215) ({
    %fieldnode_228 = sdql.access_record %l "l_orderkey" : record<"l_orderkey": i32, "l_partkey": i32, "l_suppkey": i32, "l_linenumber": i32, "l_quantity": f64, "l_extendedprice": f64, "l_discount": f64, "l_tax": f64, "l_returnflag": memref<1xi8>, "l_linestatus": memref<1xi8>, "l_shipdate": i32, "l_commitdate": i32, "l_receiptdate": i32, "l_shipinstruct": memref<25xi8>, "l_shipmode": memref<10xi8>, "l_comment": memref<44xi8>> -> i32
    %get_227 = sdql.lookup_dictionary %o_h [%fieldnode_228 : i32] : dictionary<i32, record<"_" : i32, "_" : i32>> -> record<"_" : i32, "_" : i32>
    %consti_229 = "arith.constant"() <{value = 0 : i32}> : () -> i32
    %get_226 = sdql.access_record %get_227 "%consti_229" : record<"_" : i32, "_" : i32> -> i32
    %get_225 = sdql.lookup_dictionary %c_h [%get_226 : i32] : dictionary<i32, i32> -> i32
    %get_224 = sdql.lookup_dictionary %n_h [%get_225 : i32] : dictionary<i32, i32> -> i32
    %dict_230 = sdql.empty_dictionary : i32
    %cmpi_231 = "arith.cmpi"(%get_224, %dict_230) <{predicate = 1}> : (i32, i32) -> i1
    "scf.yield"(%cmpi_231) : (i1) -> ()
  }, {
    %constv_232 = "arith.constant"() <{value = 0 : i1}> : () -> i1
    "scf.yield"(%constv_232) : (i1) -> ()
  }) : (i1) -> i1
  %if_233 = "scf.if"(%if_223) ({
    %fieldnode_235 = sdql.access_record %l "l_orderkey" : record<"l_orderkey": i32, "l_partkey": i32, "l_suppkey": i32, "l_linenumber": i32, "l_quantity": f64, "l_extendedprice": f64, "l_discount": f64, "l_tax": f64, "l_returnflag": memref<1xi8>, "l_linestatus": memref<1xi8>, "l_shipdate": i32, "l_commitdate": i32, "l_receiptdate": i32, "l_shipinstruct": memref<25xi8>, "l_shipmode": memref<10xi8>, "l_comment": memref<44xi8>> -> i32
    %get_234 = sdql.lookup_dictionary %o_h [%fieldnode_235 : i32] : dictionary<i32, record<"_" : i32, "_" : i32>> -> record<"_" : i32, "_" : i32>
    %consti_236 = "arith.constant"() <{value = 1 : i32}> : () -> i32
    %orderdate = sdql.access_record %get_234 "%consti_236" : record<"_" : i32, "_" : i32> -> i32
    %consti_237 = "arith.constant"() <{value = 10000 : i32}> : () -> i32
    %external_238 = sdql.external "inv", %consti_237 : i32 -> i32
    %orderyear = "arith.muli"(%orderdate, %external_238) : (i32, i32) -> i32
    %fieldnode_239 = sdql.access_record %l "l_extendedprice" : record<"l_orderkey": i32, "l_partkey": i32, "l_suppkey": i32, "l_linenumber": i32, "l_quantity": f64, "l_extendedprice": f64, "l_discount": f64, "l_tax": f64, "l_returnflag": memref<1xi8>, "l_linestatus": memref<1xi8>, "l_shipdate": i32, "l_commitdate": i32, "l_receiptdate": i32, "l_shipinstruct": memref<25xi8>, "l_shipmode": memref<10xi8>, "l_comment": memref<44xi8>> -> f64
    %constd_241 = "arith.constant"() <{value = 1.0  : f64}> : () -> f64
    %fieldnode_243 = sdql.access_record %l "l_discount" : record<"l_orderkey": i32, "l_partkey": i32, "l_suppkey": i32, "l_linenumber": i32, "l_quantity": f64, "l_extendedprice": f64, "l_discount": f64, "l_tax": f64, "l_returnflag": memref<1xi8>, "l_linestatus": memref<1xi8>, "l_shipdate": i32, "l_commitdate": i32, "l_receiptdate": i32, "l_shipinstruct": memref<25xi8>, "l_shipmode": memref<10xi8>, "l_comment": memref<44xi8>> -> f64
    %zero_244 = "arith.constant"() <{value = 0.0 : f64}> : () -> f64
    %negf_242 = "arith.subf"(%zero_244, %fieldnode_243) <{fastmath = #arith.fastmath<none>}> : (f64, f64) -> f64
    %addf_240 = "arith.addf"(%constd_241, %negf_242) <{fastmath = #arith.fastmath<none>}> : (f64, f64) -> f64
    %volume = "arith.mulf"(%fieldnode_239, %addf_240) <{fastmath = #arith.fastmath<none>}> : (f64, f64) -> f64
    %fieldnode_249 = sdql.access_record %l "l_suppkey" : record<"l_orderkey": i32, "l_partkey": i32, "l_suppkey": i32, "l_linenumber": i32, "l_quantity": f64, "l_extendedprice": f64, "l_discount": f64, "l_tax": f64, "l_returnflag": memref<1xi8>, "l_linestatus": memref<1xi8>, "l_shipdate": i32, "l_commitdate": i32, "l_receiptdate": i32, "l_shipinstruct": memref<25xi8>, "l_shipmode": memref<10xi8>, "l_comment": memref<44xi8>> -> i32
    %get_248 = sdql.lookup_dictionary %s_h [%fieldnode_249 : i32] : dictionary<i32, record<"_" : i32>> -> record<"_" : i32>
    %consti_250 = "arith.constant"() <{value = 0 : i32}> : () -> i32
    %get_247 = sdql.access_record %get_248 "%consti_250" : record<"_" : i32> -> i32
    %get_246 = sdql.lookup_dictionary %nationkey_to_name [%get_247 : i32] : dictionary<i32, record<"_" : memref<25xi8>>> -> record<"_" : memref<25xi8>>
    %consti_251 = "arith.constant"() <{value = 0 : i32}> : () -> i32
    %get_245 = sdql.access_record %get_246 "%consti_251" : record<"_" : memref<25xi8>> -> memref<25xi8>
    %cmp_252 = sdql.cmp %get_245, %brazil : memref<25xi8>, memref<6xi8> -> i1
    %brazil_volume = "scf.if"(%cmp_252) ({
      "scf.yield"(%volume) : (f64) -> ()
    }, {
      %constd_253 = "arith.constant"() <{value = 0.0  : f64}> : () -> f64
      "scf.yield"(%constd_253) : (f64) -> ()
    }) : (i1) -> f64
    %recnode_255 = sdql.create_record {fields = ["_", "_"]} %brazil_volume, %volume : f64, f64 -> record<"_": f64, "_": f64>
    %dict_254 = sdql.create_dictionary %orderyear, %recnode_255 : i32, record<"_" : f64, "_" : f64> -> dictionary<i32, record<"_" : f64, "_" : f64>>
    "scf.yield"(%dict_254) : (dictionary<i32, record<"_" : f64, "_" : f64>>) -> ()
  }, {
    %dict_256 = sdql.empty_dictionary : dictionary<i32, record<"_" : f64, "_" : f64>>
    "scf.yield"(%dict_256) : (dictionary<i32, record<"_" : f64, "_" : f64>>) -> ()
  }) : (i1) -> dictionary<i32, record<"_" : f64, "_" : f64>>
  sdql.yield %if_233 : dictionary<i32, record<"_" : f64, "_" : f64>>
}
%sum_257 = sdql.sum %l_h : dictionary<i32, record<"_" : f64, "_" : f64>> -> dictionary<record<"_" : i32, "_" : f64>, i32> {
^bb_258(%k: i32, %v: record<"_" : f64, "_" : f64>):
  %consti_263 = "arith.constant"() <{value = 0 : i32}> : () -> i32
  %get_262 = sdql.access_record %v "%consti_263" : record<"_" : f64, "_" : f64> -> f64
  %consti_265 = "arith.constant"() <{value = 1 : i32}> : () -> i32
  %get_264 = sdql.access_record %v "%consti_265" : record<"_" : f64, "_" : f64> -> f64
  %external_266 = sdql.external "inv", %get_264 : f64 -> f64
  %multf_261 = "arith.mulf"(%get_262, %external_266) <{fastmath = #arith.fastmath<none>}> : (f64, f64) -> f64
  %recnode_267 = sdql.create_record {fields = ["_", "_"]} %k, %multf_261 : i32, f64 -> record<"_": i32, "_": f64>
  %uniq_260 = sdql.unique %recnode_267 : record<"_" : i32, "_" : f64> -> record<"_" : i32, "_" : f64>
  %consti_268 = "arith.constant"() <{value = 1 : i32}> : () -> i32
  %dict_259 = sdql.create_dictionary %uniq_260, %consti_268 : record<"_" : i32, "_" : f64>, i32 -> dictionary<record<"_" : i32, "_" : f64>, i32>
  sdql.yield %dict_259 : dictionary<record<"_" : i32, "_" : f64>, i32>
}
