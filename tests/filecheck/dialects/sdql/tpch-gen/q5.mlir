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
%customer = sdql.load "datasets/tpch/customer.tbl" : record<"c_custkey" : dictionary<i32, i32>, "c_name" : dictionary<i32, memref<25xi8>>, "c_address" : dictionary<i32, memref<40xi8>>, "c_nationkey" : dictionary<i32, i32>, "c_phone" : dictionary<i32, memref<15xi8>>, "c_acctbal" : dictionary<i32, f64>, "c_mktsegment" : dictionary<i32, memref<10xi8>>, "c_comment" : dictionary<i32, memref<117xi8>>, "size" : i32>
%orders = sdql.load "datasets/tpch/orders.tbl" : record<"o_orderkey" : dictionary<i32, i32>, "o_custkey" : dictionary<i32, i32>, "o_orderstatus" : dictionary<i32, memref<1xi8>>, "o_totalprice" : dictionary<i32, f64>, "o_orderdate" : dictionary<i32, i32>, "o_orderpriority" : dictionary<i32, memref<15xi8>>, "o_clerk" : dictionary<i32, memref<15xi8>>, "o_shippriority" : dictionary<i32, i32>, "o_comment" : dictionary<i32, memref<79xi8>>, "size" : i32>
%lineitem = sdql.load "datasets/tpch/lineitem.tbl" : record<"l_orderkey" : dictionary<i32, i32>, "l_partkey" : dictionary<i32, i32>, "l_suppkey" : dictionary<i32, i32>, "l_linenumber" : dictionary<i32, i32>, "l_quantity" : dictionary<i32, f64>, "l_extendedprice" : dictionary<i32, f64>, "l_discount" : dictionary<i32, f64>, "l_tax" : dictionary<i32, f64>, "l_returnflag" : dictionary<i32, memref<1xi8>>, "l_linestatus" : dictionary<i32, memref<1xi8>>, "l_shipdate" : dictionary<i32, i32>, "l_commitdate" : dictionary<i32, i32>, "l_receiptdate" : dictionary<i32, i32>, "l_shipinstruct" : dictionary<i32, memref<25xi8>>, "l_shipmode" : dictionary<i32, memref<10xi8>>, "l_comment" : dictionary<i32, memref<44xi8>>, "size" : i32>
%supplier = sdql.load "datasets/tpch/supplier.tbl" : record<"s_suppkey" : dictionary<i32, i32>, "s_name" : dictionary<i32, memref<25xi8>>, "s_address" : dictionary<i32, memref<40xi8>>, "s_nationkey" : dictionary<i32, i32>, "s_phone" : dictionary<i32, memref<15xi8>>, "s_acctbal" : dictionary<i32, f64>, "s_comment" : dictionary<i32, memref<101xi8>>, "size" : i32>
%nation = sdql.load "datasets/tpch/nation.tbl" : record<"n_nationkey" : dictionary<i32, i32>, "n_name" : dictionary<i32, memref<25xi8>>, "n_regionkey" : dictionary<i32, i32>, "n_comment" : dictionary<i32, memref<152xi8>>, "size" : i32>
%region = sdql.load "datasets/tpch/region.tbl" : record<"r_regionkey" : dictionary<i32, i32>, "r_name" : dictionary<i32, memref<25xi8>>, "r_comment" : dictionary<i32, memref<152xi8>>, "size" : i32>
// ASIA
%constds_0 = "arith.constant"() <{value = dense<[65, 83, 73, 65]> : memref<?xi8>}> : () -> memref<?xi8>
%consti_1 = "arith.constant"() <{value = 5 : i32}> : () -> i32
%asia = sdql.external "ConstantString", %constds_0, %consti_1 : memref<?xi8>, i32 -> memref<4xi8>
%fieldnode_2 = sdql.access_record %region "size" : record<"r_regionkey": dictionary<i32, i32>, "r_name": dictionary<i32, memref<25xi8>>, "r_comment": dictionary<i32, memref<152xi8>>, "size": i32> -> i32
%rangenode_3 = "func.call"(%fieldnode_2) <{callee = @range_builtin}> : (i32) -> dictionary<i32, i32>
%r_h = sdql.sum %rangenode_3 : dictionary<i32, i32> -> dictionary<i32, record<"_" : i32>> {
^bb_4(%i: i32, %_: i32):
  %fieldnode_6 = sdql.access_record %region "r_regionkey" : record<"r_regionkey": dictionary<i32, i32>, "r_name": dictionary<i32, memref<25xi8>>, "r_comment": dictionary<i32, memref<152xi8>>, "size": i32> -> dictionary<i32, i32>
  %get_5 = sdql.lookup_dictionary %fieldnode_6 [%i : i32] : dictionary<i32, i32> -> i32
  %fieldnode_8 = sdql.access_record %region "r_name" : record<"r_regionkey": dictionary<i32, i32>, "r_name": dictionary<i32, memref<25xi8>>, "r_comment": dictionary<i32, memref<152xi8>>, "size": i32> -> dictionary<i32, memref<25xi8>>
  %get_7 = sdql.lookup_dictionary %fieldnode_8 [%i : i32] : dictionary<i32, memref<25xi8>> -> memref<25xi8>
  %fieldnode_10 = sdql.access_record %region "r_comment" : record<"r_regionkey": dictionary<i32, i32>, "r_name": dictionary<i32, memref<25xi8>>, "r_comment": dictionary<i32, memref<152xi8>>, "size": i32> -> dictionary<i32, memref<152xi8>>
  %get_9 = sdql.lookup_dictionary %fieldnode_10 [%i : i32] : dictionary<i32, memref<152xi8>> -> memref<152xi8>
  %r = sdql.create_record {fields = ["r_regionkey", "r_name", "r_comment"]} %get_5, %get_7, %get_9 : i32, memref<25xi8>, memref<152xi8> -> record<"r_regionkey": i32, "r_name": memref<25xi8>, "r_comment": memref<152xi8>>
  %fieldnode_11 = sdql.access_record %r "r_name" : record<"r_regionkey": i32, "r_name": memref<25xi8>, "r_comment": memref<152xi8>> -> memref<25xi8>
  %cmp_12 = sdql.cmp %fieldnode_11, %asia : memref<25xi8>, memref<4xi8> -> i1
  %if_13 = "scf.if"(%cmp_12) ({
    %fieldnode_16 = sdql.access_record %r "r_regionkey" : record<"r_regionkey": i32, "r_name": memref<25xi8>, "r_comment": memref<152xi8>> -> i32
    %uniq_15 = sdql.unique %fieldnode_16 : i32 -> i32
    %fieldnode_17 = sdql.access_record %r "r_regionkey" : record<"r_regionkey": i32, "r_name": memref<25xi8>, "r_comment": memref<152xi8>> -> i32
    %recnode_18 = sdql.create_record {fields = ["_"]} %fieldnode_17 : i32 -> record<"_": i32>
    %dict_14 = sdql.create_dictionary %uniq_15, %recnode_18 : i32, record<"_" : i32> -> dictionary<i32, record<"_" : i32>>
    "scf.yield"(%dict_14) : (dictionary<i32, record<"_" : i32>>) -> ()
  }, {
    %dict_19 = sdql.empty_dictionary : dictionary<i32, record<"_" : i32>>
    "scf.yield"(%dict_19) : (dictionary<i32, record<"_" : i32>>) -> ()
  }) : (i1) -> dictionary<i32, record<"_" : i32>>
  sdql.yield %if_13 : dictionary<i32, record<"_" : i32>>
}
%fieldnode_20 = sdql.access_record %nation "size" : record<"n_nationkey": dictionary<i32, i32>, "n_name": dictionary<i32, memref<25xi8>>, "n_regionkey": dictionary<i32, i32>, "n_comment": dictionary<i32, memref<152xi8>>, "size": i32> -> i32
%rangenode_21 = "func.call"(%fieldnode_20) <{callee = @range_builtin}> : (i32) -> dictionary<i32, i32>
%n_h = sdql.sum %rangenode_21 : dictionary<i32, i32> -> dictionary<i32, memref<25xi8>> {
^bb_22(%i: i32, %_: i32):
  %fieldnode_24 = sdql.access_record %nation "n_nationkey" : record<"n_nationkey": dictionary<i32, i32>, "n_name": dictionary<i32, memref<25xi8>>, "n_regionkey": dictionary<i32, i32>, "n_comment": dictionary<i32, memref<152xi8>>, "size": i32> -> dictionary<i32, i32>
  %get_23 = sdql.lookup_dictionary %fieldnode_24 [%i : i32] : dictionary<i32, i32> -> i32
  %fieldnode_26 = sdql.access_record %nation "n_name" : record<"n_nationkey": dictionary<i32, i32>, "n_name": dictionary<i32, memref<25xi8>>, "n_regionkey": dictionary<i32, i32>, "n_comment": dictionary<i32, memref<152xi8>>, "size": i32> -> dictionary<i32, memref<25xi8>>
  %get_25 = sdql.lookup_dictionary %fieldnode_26 [%i : i32] : dictionary<i32, memref<25xi8>> -> memref<25xi8>
  %fieldnode_28 = sdql.access_record %nation "n_regionkey" : record<"n_nationkey": dictionary<i32, i32>, "n_name": dictionary<i32, memref<25xi8>>, "n_regionkey": dictionary<i32, i32>, "n_comment": dictionary<i32, memref<152xi8>>, "size": i32> -> dictionary<i32, i32>
  %get_27 = sdql.lookup_dictionary %fieldnode_28 [%i : i32] : dictionary<i32, i32> -> i32
  %fieldnode_30 = sdql.access_record %nation "n_comment" : record<"n_nationkey": dictionary<i32, i32>, "n_name": dictionary<i32, memref<25xi8>>, "n_regionkey": dictionary<i32, i32>, "n_comment": dictionary<i32, memref<152xi8>>, "size": i32> -> dictionary<i32, memref<152xi8>>
  %get_29 = sdql.lookup_dictionary %fieldnode_30 [%i : i32] : dictionary<i32, memref<152xi8>> -> memref<152xi8>
  %n = sdql.create_record {fields = ["n_nationkey", "n_name", "n_regionkey", "n_comment"]} %get_23, %get_25, %get_27, %get_29 : i32, memref<25xi8>, i32, memref<152xi8> -> record<"n_nationkey": i32, "n_name": memref<25xi8>, "n_regionkey": i32, "n_comment": memref<152xi8>>
  %fieldnode_34 = sdql.access_record %n "n_regionkey" : record<"n_nationkey": i32, "n_name": memref<25xi8>, "n_regionkey": i32, "n_comment": memref<152xi8>> -> i32
  %get_33 = sdql.lookup_dictionary %r_h [%fieldnode_34 : i32] : dictionary<i32, record<"_" : i32>> -> record<"_" : i32>
  %dict_35 = sdql.empty_dictionary : record<"_" : i32>
  %cmp_36 = sdql.cmp %get_33, %dict_35 : record<"_" : i32>, record<"_" : i32> -> i1
  %zero_31 = "arith.constant"() <{value = 0 : i1}> : () -> i1
  %neg_32 = "arith.cmpi"(%cmp_36, %zero_31) <{predicate = 0}> : (i1, i1) -> i1
  %if_37 = "scf.if"(%neg_32) ({
    %fieldnode_40 = sdql.access_record %n "n_nationkey" : record<"n_nationkey": i32, "n_name": memref<25xi8>, "n_regionkey": i32, "n_comment": memref<152xi8>> -> i32
    %uniq_39 = sdql.unique %fieldnode_40 : i32 -> i32
    %fieldnode_41 = sdql.access_record %n "n_name" : record<"n_nationkey": i32, "n_name": memref<25xi8>, "n_regionkey": i32, "n_comment": memref<152xi8>> -> memref<25xi8>
    %dict_38 = sdql.create_dictionary %uniq_39, %fieldnode_41 : i32, memref<25xi8> -> dictionary<i32, memref<25xi8>>
    "scf.yield"(%dict_38) : (dictionary<i32, memref<25xi8>>) -> ()
  }, {
    %dict_42 = sdql.empty_dictionary : dictionary<i32, memref<25xi8>>
    "scf.yield"(%dict_42) : (dictionary<i32, memref<25xi8>>) -> ()
  }) : (i1) -> dictionary<i32, memref<25xi8>>
  sdql.yield %if_37 : dictionary<i32, memref<25xi8>>
}
%fieldnode_43 = sdql.access_record %customer "size" : record<"c_custkey": dictionary<i32, i32>, "c_name": dictionary<i32, memref<25xi8>>, "c_address": dictionary<i32, memref<40xi8>>, "c_nationkey": dictionary<i32, i32>, "c_phone": dictionary<i32, memref<15xi8>>, "c_acctbal": dictionary<i32, f64>, "c_mktsegment": dictionary<i32, memref<10xi8>>, "c_comment": dictionary<i32, memref<117xi8>>, "size": i32> -> i32
%rangenode_44 = "func.call"(%fieldnode_43) <{callee = @range_builtin}> : (i32) -> dictionary<i32, i32>
%c_h = sdql.sum %rangenode_44 : dictionary<i32, i32> -> dictionary<i32, record<"_" : memref<25xi8>, "_" : i32>> {
^bb_45(%i: i32, %_: i32):
  %fieldnode_47 = sdql.access_record %customer "c_custkey" : record<"c_custkey": dictionary<i32, i32>, "c_name": dictionary<i32, memref<25xi8>>, "c_address": dictionary<i32, memref<40xi8>>, "c_nationkey": dictionary<i32, i32>, "c_phone": dictionary<i32, memref<15xi8>>, "c_acctbal": dictionary<i32, f64>, "c_mktsegment": dictionary<i32, memref<10xi8>>, "c_comment": dictionary<i32, memref<117xi8>>, "size": i32> -> dictionary<i32, i32>
  %get_46 = sdql.lookup_dictionary %fieldnode_47 [%i : i32] : dictionary<i32, i32> -> i32
  %fieldnode_49 = sdql.access_record %customer "c_name" : record<"c_custkey": dictionary<i32, i32>, "c_name": dictionary<i32, memref<25xi8>>, "c_address": dictionary<i32, memref<40xi8>>, "c_nationkey": dictionary<i32, i32>, "c_phone": dictionary<i32, memref<15xi8>>, "c_acctbal": dictionary<i32, f64>, "c_mktsegment": dictionary<i32, memref<10xi8>>, "c_comment": dictionary<i32, memref<117xi8>>, "size": i32> -> dictionary<i32, memref<25xi8>>
  %get_48 = sdql.lookup_dictionary %fieldnode_49 [%i : i32] : dictionary<i32, memref<25xi8>> -> memref<25xi8>
  %fieldnode_51 = sdql.access_record %customer "c_address" : record<"c_custkey": dictionary<i32, i32>, "c_name": dictionary<i32, memref<25xi8>>, "c_address": dictionary<i32, memref<40xi8>>, "c_nationkey": dictionary<i32, i32>, "c_phone": dictionary<i32, memref<15xi8>>, "c_acctbal": dictionary<i32, f64>, "c_mktsegment": dictionary<i32, memref<10xi8>>, "c_comment": dictionary<i32, memref<117xi8>>, "size": i32> -> dictionary<i32, memref<40xi8>>
  %get_50 = sdql.lookup_dictionary %fieldnode_51 [%i : i32] : dictionary<i32, memref<40xi8>> -> memref<40xi8>
  %fieldnode_53 = sdql.access_record %customer "c_nationkey" : record<"c_custkey": dictionary<i32, i32>, "c_name": dictionary<i32, memref<25xi8>>, "c_address": dictionary<i32, memref<40xi8>>, "c_nationkey": dictionary<i32, i32>, "c_phone": dictionary<i32, memref<15xi8>>, "c_acctbal": dictionary<i32, f64>, "c_mktsegment": dictionary<i32, memref<10xi8>>, "c_comment": dictionary<i32, memref<117xi8>>, "size": i32> -> dictionary<i32, i32>
  %get_52 = sdql.lookup_dictionary %fieldnode_53 [%i : i32] : dictionary<i32, i32> -> i32
  %fieldnode_55 = sdql.access_record %customer "c_phone" : record<"c_custkey": dictionary<i32, i32>, "c_name": dictionary<i32, memref<25xi8>>, "c_address": dictionary<i32, memref<40xi8>>, "c_nationkey": dictionary<i32, i32>, "c_phone": dictionary<i32, memref<15xi8>>, "c_acctbal": dictionary<i32, f64>, "c_mktsegment": dictionary<i32, memref<10xi8>>, "c_comment": dictionary<i32, memref<117xi8>>, "size": i32> -> dictionary<i32, memref<15xi8>>
  %get_54 = sdql.lookup_dictionary %fieldnode_55 [%i : i32] : dictionary<i32, memref<15xi8>> -> memref<15xi8>
  %fieldnode_57 = sdql.access_record %customer "c_acctbal" : record<"c_custkey": dictionary<i32, i32>, "c_name": dictionary<i32, memref<25xi8>>, "c_address": dictionary<i32, memref<40xi8>>, "c_nationkey": dictionary<i32, i32>, "c_phone": dictionary<i32, memref<15xi8>>, "c_acctbal": dictionary<i32, f64>, "c_mktsegment": dictionary<i32, memref<10xi8>>, "c_comment": dictionary<i32, memref<117xi8>>, "size": i32> -> dictionary<i32, f64>
  %get_56 = sdql.lookup_dictionary %fieldnode_57 [%i : i32] : dictionary<i32, f64> -> f64
  %fieldnode_59 = sdql.access_record %customer "c_mktsegment" : record<"c_custkey": dictionary<i32, i32>, "c_name": dictionary<i32, memref<25xi8>>, "c_address": dictionary<i32, memref<40xi8>>, "c_nationkey": dictionary<i32, i32>, "c_phone": dictionary<i32, memref<15xi8>>, "c_acctbal": dictionary<i32, f64>, "c_mktsegment": dictionary<i32, memref<10xi8>>, "c_comment": dictionary<i32, memref<117xi8>>, "size": i32> -> dictionary<i32, memref<10xi8>>
  %get_58 = sdql.lookup_dictionary %fieldnode_59 [%i : i32] : dictionary<i32, memref<10xi8>> -> memref<10xi8>
  %fieldnode_61 = sdql.access_record %customer "c_comment" : record<"c_custkey": dictionary<i32, i32>, "c_name": dictionary<i32, memref<25xi8>>, "c_address": dictionary<i32, memref<40xi8>>, "c_nationkey": dictionary<i32, i32>, "c_phone": dictionary<i32, memref<15xi8>>, "c_acctbal": dictionary<i32, f64>, "c_mktsegment": dictionary<i32, memref<10xi8>>, "c_comment": dictionary<i32, memref<117xi8>>, "size": i32> -> dictionary<i32, memref<117xi8>>
  %get_60 = sdql.lookup_dictionary %fieldnode_61 [%i : i32] : dictionary<i32, memref<117xi8>> -> memref<117xi8>
  %c = sdql.create_record {fields = ["c_custkey", "c_name", "c_address", "c_nationkey", "c_phone", "c_acctbal", "c_mktsegment", "c_comment"]} %get_46, %get_48, %get_50, %get_52, %get_54, %get_56, %get_58, %get_60 : i32, memref<25xi8>, memref<40xi8>, i32, memref<15xi8>, f64, memref<10xi8>, memref<117xi8> -> record<"c_custkey": i32, "c_name": memref<25xi8>, "c_address": memref<40xi8>, "c_nationkey": i32, "c_phone": memref<15xi8>, "c_acctbal": f64, "c_mktsegment": memref<10xi8>, "c_comment": memref<117xi8>>
  %fieldnode_65 = sdql.access_record %c "c_nationkey" : record<"c_custkey": i32, "c_name": memref<25xi8>, "c_address": memref<40xi8>, "c_nationkey": i32, "c_phone": memref<15xi8>, "c_acctbal": f64, "c_mktsegment": memref<10xi8>, "c_comment": memref<117xi8>> -> i32
  %get_64 = sdql.lookup_dictionary %n_h [%fieldnode_65 : i32] : dictionary<i32, memref<25xi8>> -> memref<25xi8>
  %dict_66 = sdql.empty_dictionary : memref<25xi8>
  %cmp_67 = sdql.cmp %get_64, %dict_66 : memref<25xi8>, memref<25xi8> -> i1
  %zero_62 = "arith.constant"() <{value = 0 : i1}> : () -> i1
  %neg_63 = "arith.cmpi"(%cmp_67, %zero_62) <{predicate = 0}> : (i1, i1) -> i1
  %if_68 = "scf.if"(%neg_63) ({
    %fieldnode_71 = sdql.access_record %c "c_custkey" : record<"c_custkey": i32, "c_name": memref<25xi8>, "c_address": memref<40xi8>, "c_nationkey": i32, "c_phone": memref<15xi8>, "c_acctbal": f64, "c_mktsegment": memref<10xi8>, "c_comment": memref<117xi8>> -> i32
    %uniq_70 = sdql.unique %fieldnode_71 : i32 -> i32
    %fieldnode_73 = sdql.access_record %c "c_nationkey" : record<"c_custkey": i32, "c_name": memref<25xi8>, "c_address": memref<40xi8>, "c_nationkey": i32, "c_phone": memref<15xi8>, "c_acctbal": f64, "c_mktsegment": memref<10xi8>, "c_comment": memref<117xi8>> -> i32
    %get_72 = sdql.lookup_dictionary %n_h [%fieldnode_73 : i32] : dictionary<i32, memref<25xi8>> -> memref<25xi8>
    %fieldnode_74 = sdql.access_record %c "c_nationkey" : record<"c_custkey": i32, "c_name": memref<25xi8>, "c_address": memref<40xi8>, "c_nationkey": i32, "c_phone": memref<15xi8>, "c_acctbal": f64, "c_mktsegment": memref<10xi8>, "c_comment": memref<117xi8>> -> i32
    %recnode_75 = sdql.create_record {fields = ["_", "_"]} %get_72, %fieldnode_74 : memref<25xi8>, i32 -> record<"_": memref<25xi8>, "_": i32>
    %dict_69 = sdql.create_dictionary %uniq_70, %recnode_75 : i32, record<"_" : memref<25xi8>, "_" : i32> -> dictionary<i32, record<"_" : memref<25xi8>, "_" : i32>>
    "scf.yield"(%dict_69) : (dictionary<i32, record<"_" : memref<25xi8>, "_" : i32>>) -> ()
  }, {
    %dict_76 = sdql.empty_dictionary : dictionary<i32, record<"_" : memref<25xi8>, "_" : i32>>
    "scf.yield"(%dict_76) : (dictionary<i32, record<"_" : memref<25xi8>, "_" : i32>>) -> ()
  }) : (i1) -> dictionary<i32, record<"_" : memref<25xi8>, "_" : i32>>
  sdql.yield %if_68 : dictionary<i32, record<"_" : memref<25xi8>, "_" : i32>>
}
%fieldnode_77 = sdql.access_record %orders "size" : record<"o_orderkey": dictionary<i32, i32>, "o_custkey": dictionary<i32, i32>, "o_orderstatus": dictionary<i32, memref<1xi8>>, "o_totalprice": dictionary<i32, f64>, "o_orderdate": dictionary<i32, i32>, "o_orderpriority": dictionary<i32, memref<15xi8>>, "o_clerk": dictionary<i32, memref<15xi8>>, "o_shippriority": dictionary<i32, i32>, "o_comment": dictionary<i32, memref<79xi8>>, "size": i32> -> i32
%rangenode_78 = "func.call"(%fieldnode_77) <{callee = @range_builtin}> : (i32) -> dictionary<i32, i32>
%o_h = sdql.sum %rangenode_78 : dictionary<i32, i32> -> dictionary<i32, record<"_" : memref<25xi8>, "_" : i32>> {
^bb_79(%i: i32, %_: i32):
  %fieldnode_81 = sdql.access_record %orders "o_orderkey" : record<"o_orderkey": dictionary<i32, i32>, "o_custkey": dictionary<i32, i32>, "o_orderstatus": dictionary<i32, memref<1xi8>>, "o_totalprice": dictionary<i32, f64>, "o_orderdate": dictionary<i32, i32>, "o_orderpriority": dictionary<i32, memref<15xi8>>, "o_clerk": dictionary<i32, memref<15xi8>>, "o_shippriority": dictionary<i32, i32>, "o_comment": dictionary<i32, memref<79xi8>>, "size": i32> -> dictionary<i32, i32>
  %get_80 = sdql.lookup_dictionary %fieldnode_81 [%i : i32] : dictionary<i32, i32> -> i32
  %fieldnode_83 = sdql.access_record %orders "o_custkey" : record<"o_orderkey": dictionary<i32, i32>, "o_custkey": dictionary<i32, i32>, "o_orderstatus": dictionary<i32, memref<1xi8>>, "o_totalprice": dictionary<i32, f64>, "o_orderdate": dictionary<i32, i32>, "o_orderpriority": dictionary<i32, memref<15xi8>>, "o_clerk": dictionary<i32, memref<15xi8>>, "o_shippriority": dictionary<i32, i32>, "o_comment": dictionary<i32, memref<79xi8>>, "size": i32> -> dictionary<i32, i32>
  %get_82 = sdql.lookup_dictionary %fieldnode_83 [%i : i32] : dictionary<i32, i32> -> i32
  %fieldnode_85 = sdql.access_record %orders "o_orderstatus" : record<"o_orderkey": dictionary<i32, i32>, "o_custkey": dictionary<i32, i32>, "o_orderstatus": dictionary<i32, memref<1xi8>>, "o_totalprice": dictionary<i32, f64>, "o_orderdate": dictionary<i32, i32>, "o_orderpriority": dictionary<i32, memref<15xi8>>, "o_clerk": dictionary<i32, memref<15xi8>>, "o_shippriority": dictionary<i32, i32>, "o_comment": dictionary<i32, memref<79xi8>>, "size": i32> -> dictionary<i32, memref<1xi8>>
  %get_84 = sdql.lookup_dictionary %fieldnode_85 [%i : i32] : dictionary<i32, memref<1xi8>> -> memref<1xi8>
  %fieldnode_87 = sdql.access_record %orders "o_totalprice" : record<"o_orderkey": dictionary<i32, i32>, "o_custkey": dictionary<i32, i32>, "o_orderstatus": dictionary<i32, memref<1xi8>>, "o_totalprice": dictionary<i32, f64>, "o_orderdate": dictionary<i32, i32>, "o_orderpriority": dictionary<i32, memref<15xi8>>, "o_clerk": dictionary<i32, memref<15xi8>>, "o_shippriority": dictionary<i32, i32>, "o_comment": dictionary<i32, memref<79xi8>>, "size": i32> -> dictionary<i32, f64>
  %get_86 = sdql.lookup_dictionary %fieldnode_87 [%i : i32] : dictionary<i32, f64> -> f64
  %fieldnode_89 = sdql.access_record %orders "o_orderdate" : record<"o_orderkey": dictionary<i32, i32>, "o_custkey": dictionary<i32, i32>, "o_orderstatus": dictionary<i32, memref<1xi8>>, "o_totalprice": dictionary<i32, f64>, "o_orderdate": dictionary<i32, i32>, "o_orderpriority": dictionary<i32, memref<15xi8>>, "o_clerk": dictionary<i32, memref<15xi8>>, "o_shippriority": dictionary<i32, i32>, "o_comment": dictionary<i32, memref<79xi8>>, "size": i32> -> dictionary<i32, i32>
  %get_88 = sdql.lookup_dictionary %fieldnode_89 [%i : i32] : dictionary<i32, i32> -> i32
  %fieldnode_91 = sdql.access_record %orders "o_orderpriority" : record<"o_orderkey": dictionary<i32, i32>, "o_custkey": dictionary<i32, i32>, "o_orderstatus": dictionary<i32, memref<1xi8>>, "o_totalprice": dictionary<i32, f64>, "o_orderdate": dictionary<i32, i32>, "o_orderpriority": dictionary<i32, memref<15xi8>>, "o_clerk": dictionary<i32, memref<15xi8>>, "o_shippriority": dictionary<i32, i32>, "o_comment": dictionary<i32, memref<79xi8>>, "size": i32> -> dictionary<i32, memref<15xi8>>
  %get_90 = sdql.lookup_dictionary %fieldnode_91 [%i : i32] : dictionary<i32, memref<15xi8>> -> memref<15xi8>
  %fieldnode_93 = sdql.access_record %orders "o_clerk" : record<"o_orderkey": dictionary<i32, i32>, "o_custkey": dictionary<i32, i32>, "o_orderstatus": dictionary<i32, memref<1xi8>>, "o_totalprice": dictionary<i32, f64>, "o_orderdate": dictionary<i32, i32>, "o_orderpriority": dictionary<i32, memref<15xi8>>, "o_clerk": dictionary<i32, memref<15xi8>>, "o_shippriority": dictionary<i32, i32>, "o_comment": dictionary<i32, memref<79xi8>>, "size": i32> -> dictionary<i32, memref<15xi8>>
  %get_92 = sdql.lookup_dictionary %fieldnode_93 [%i : i32] : dictionary<i32, memref<15xi8>> -> memref<15xi8>
  %fieldnode_95 = sdql.access_record %orders "o_shippriority" : record<"o_orderkey": dictionary<i32, i32>, "o_custkey": dictionary<i32, i32>, "o_orderstatus": dictionary<i32, memref<1xi8>>, "o_totalprice": dictionary<i32, f64>, "o_orderdate": dictionary<i32, i32>, "o_orderpriority": dictionary<i32, memref<15xi8>>, "o_clerk": dictionary<i32, memref<15xi8>>, "o_shippriority": dictionary<i32, i32>, "o_comment": dictionary<i32, memref<79xi8>>, "size": i32> -> dictionary<i32, i32>
  %get_94 = sdql.lookup_dictionary %fieldnode_95 [%i : i32] : dictionary<i32, i32> -> i32
  %fieldnode_97 = sdql.access_record %orders "o_comment" : record<"o_orderkey": dictionary<i32, i32>, "o_custkey": dictionary<i32, i32>, "o_orderstatus": dictionary<i32, memref<1xi8>>, "o_totalprice": dictionary<i32, f64>, "o_orderdate": dictionary<i32, i32>, "o_orderpriority": dictionary<i32, memref<15xi8>>, "o_clerk": dictionary<i32, memref<15xi8>>, "o_shippriority": dictionary<i32, i32>, "o_comment": dictionary<i32, memref<79xi8>>, "size": i32> -> dictionary<i32, memref<79xi8>>
  %get_96 = sdql.lookup_dictionary %fieldnode_97 [%i : i32] : dictionary<i32, memref<79xi8>> -> memref<79xi8>
  %o = sdql.create_record {fields = ["o_orderkey", "o_custkey", "o_orderstatus", "o_totalprice", "o_orderdate", "o_orderpriority", "o_clerk", "o_shippriority", "o_comment"]} %get_80, %get_82, %get_84, %get_86, %get_88, %get_90, %get_92, %get_94, %get_96 : i32, i32, memref<1xi8>, f64, i32, memref<15xi8>, memref<15xi8>, i32, memref<79xi8> -> record<"o_orderkey": i32, "o_custkey": i32, "o_orderstatus": memref<1xi8>, "o_totalprice": f64, "o_orderdate": i32, "o_orderpriority": memref<15xi8>, "o_clerk": memref<15xi8>, "o_shippriority": i32, "o_comment": memref<79xi8>>
  %fieldnode_98 = sdql.access_record %o "o_orderdate" : record<"o_orderkey": i32, "o_custkey": i32, "o_orderstatus": memref<1xi8>, "o_totalprice": f64, "o_orderdate": i32, "o_orderpriority": memref<15xi8>, "o_clerk": memref<15xi8>, "o_shippriority": i32, "o_comment": memref<79xi8>> -> i32
  %constda_99 = "arith.constant"() <{value = 19950101 : i32}> : () -> i32
  %cmpi_100 = "arith.cmpi"(%fieldnode_98, %constda_99) <{predicate = 2}> : (i32, i32) -> i1
  %if_101 = "scf.if"(%cmpi_100) ({
    %constda_102 = "arith.constant"() <{value = 19940101 : i32}> : () -> i32
    %fieldnode_103 = sdql.access_record %o "o_orderdate" : record<"o_orderkey": i32, "o_custkey": i32, "o_orderstatus": memref<1xi8>, "o_totalprice": f64, "o_orderdate": i32, "o_orderpriority": memref<15xi8>, "o_clerk": memref<15xi8>, "o_shippriority": i32, "o_comment": memref<79xi8>> -> i32
    %cmpi_104 = "arith.cmpi"(%constda_102, %fieldnode_103) <{predicate = 3}> : (i32, i32) -> i1
    "scf.yield"(%cmpi_104) : (i1) -> ()
  }, {
    %constv_105 = "arith.constant"() <{value = 0 : i1}> : () -> i1
    "scf.yield"(%constv_105) : (i1) -> ()
  }) : (i1) -> i1
  %if_106 = "scf.if"(%if_101) ({
    %fieldnode_110 = sdql.access_record %o "o_custkey" : record<"o_orderkey": i32, "o_custkey": i32, "o_orderstatus": memref<1xi8>, "o_totalprice": f64, "o_orderdate": i32, "o_orderpriority": memref<15xi8>, "o_clerk": memref<15xi8>, "o_shippriority": i32, "o_comment": memref<79xi8>> -> i32
    %get_109 = sdql.lookup_dictionary %c_h [%fieldnode_110 : i32] : dictionary<i32, record<"_" : memref<25xi8>, "_" : i32>> -> record<"_" : memref<25xi8>, "_" : i32>
    %dict_111 = sdql.empty_dictionary : record<"_" : memref<25xi8>, "_" : i32>
    %cmp_112 = sdql.cmp %get_109, %dict_111 : record<"_" : memref<25xi8>, "_" : i32>, record<"_" : memref<25xi8>, "_" : i32> -> i1
    %zero_107 = "arith.constant"() <{value = 0 : i1}> : () -> i1
    %neg_108 = "arith.cmpi"(%cmp_112, %zero_107) <{predicate = 0}> : (i1, i1) -> i1
    "scf.yield"(%neg_108) : (i1) -> ()
  }, {
    %constv_113 = "arith.constant"() <{value = 0 : i1}> : () -> i1
    "scf.yield"(%constv_113) : (i1) -> ()
  }) : (i1) -> i1
  %if_114 = "scf.if"(%if_106) ({
    %fieldnode_117 = sdql.access_record %o "o_orderkey" : record<"o_orderkey": i32, "o_custkey": i32, "o_orderstatus": memref<1xi8>, "o_totalprice": f64, "o_orderdate": i32, "o_orderpriority": memref<15xi8>, "o_clerk": memref<15xi8>, "o_shippriority": i32, "o_comment": memref<79xi8>> -> i32
    %uniq_116 = sdql.unique %fieldnode_117 : i32 -> i32
    %fieldnode_120 = sdql.access_record %o "o_custkey" : record<"o_orderkey": i32, "o_custkey": i32, "o_orderstatus": memref<1xi8>, "o_totalprice": f64, "o_orderdate": i32, "o_orderpriority": memref<15xi8>, "o_clerk": memref<15xi8>, "o_shippriority": i32, "o_comment": memref<79xi8>> -> i32
    %get_119 = sdql.lookup_dictionary %c_h [%fieldnode_120 : i32] : dictionary<i32, record<"_" : memref<25xi8>, "_" : i32>> -> record<"_" : memref<25xi8>, "_" : i32>
    %consti_121 = "arith.constant"() <{value = 0 : i32}> : () -> i32
    %get_118 = sdql.access_record %get_119 "%consti_121" : record<"_" : memref<25xi8>, "_" : i32> -> memref<25xi8>
    %fieldnode_124 = sdql.access_record %o "o_custkey" : record<"o_orderkey": i32, "o_custkey": i32, "o_orderstatus": memref<1xi8>, "o_totalprice": f64, "o_orderdate": i32, "o_orderpriority": memref<15xi8>, "o_clerk": memref<15xi8>, "o_shippriority": i32, "o_comment": memref<79xi8>> -> i32
    %get_123 = sdql.lookup_dictionary %c_h [%fieldnode_124 : i32] : dictionary<i32, record<"_" : memref<25xi8>, "_" : i32>> -> record<"_" : memref<25xi8>, "_" : i32>
    %consti_125 = "arith.constant"() <{value = 1 : i32}> : () -> i32
    %get_122 = sdql.access_record %get_123 "%consti_125" : record<"_" : memref<25xi8>, "_" : i32> -> i32
    %recnode_126 = sdql.create_record {fields = ["_", "_"]} %get_118, %get_122 : memref<25xi8>, i32 -> record<"_": memref<25xi8>, "_": i32>
    %dict_115 = sdql.create_dictionary %uniq_116, %recnode_126 : i32, record<"_" : memref<25xi8>, "_" : i32> -> dictionary<i32, record<"_" : memref<25xi8>, "_" : i32>>
    "scf.yield"(%dict_115) : (dictionary<i32, record<"_" : memref<25xi8>, "_" : i32>>) -> ()
  }, {
    %dict_127 = sdql.empty_dictionary : dictionary<i32, record<"_" : memref<25xi8>, "_" : i32>>
    "scf.yield"(%dict_127) : (dictionary<i32, record<"_" : memref<25xi8>, "_" : i32>>) -> ()
  }) : (i1) -> dictionary<i32, record<"_" : memref<25xi8>, "_" : i32>>
  sdql.yield %if_114 : dictionary<i32, record<"_" : memref<25xi8>, "_" : i32>>
}
%fieldnode_128 = sdql.access_record %supplier "size" : record<"s_suppkey": dictionary<i32, i32>, "s_name": dictionary<i32, memref<25xi8>>, "s_address": dictionary<i32, memref<40xi8>>, "s_nationkey": dictionary<i32, i32>, "s_phone": dictionary<i32, memref<15xi8>>, "s_acctbal": dictionary<i32, f64>, "s_comment": dictionary<i32, memref<101xi8>>, "size": i32> -> i32
%rangenode_129 = "func.call"(%fieldnode_128) <{callee = @range_builtin}> : (i32) -> dictionary<i32, i32>
%s_h = sdql.sum %rangenode_129 : dictionary<i32, i32> -> dictionary<record<"_" : i32, "_" : i32>, i32> {
^bb_130(%i: i32, %_: i32):
  %fieldnode_132 = sdql.access_record %supplier "s_suppkey" : record<"s_suppkey": dictionary<i32, i32>, "s_name": dictionary<i32, memref<25xi8>>, "s_address": dictionary<i32, memref<40xi8>>, "s_nationkey": dictionary<i32, i32>, "s_phone": dictionary<i32, memref<15xi8>>, "s_acctbal": dictionary<i32, f64>, "s_comment": dictionary<i32, memref<101xi8>>, "size": i32> -> dictionary<i32, i32>
  %get_131 = sdql.lookup_dictionary %fieldnode_132 [%i : i32] : dictionary<i32, i32> -> i32
  %fieldnode_134 = sdql.access_record %supplier "s_name" : record<"s_suppkey": dictionary<i32, i32>, "s_name": dictionary<i32, memref<25xi8>>, "s_address": dictionary<i32, memref<40xi8>>, "s_nationkey": dictionary<i32, i32>, "s_phone": dictionary<i32, memref<15xi8>>, "s_acctbal": dictionary<i32, f64>, "s_comment": dictionary<i32, memref<101xi8>>, "size": i32> -> dictionary<i32, memref<25xi8>>
  %get_133 = sdql.lookup_dictionary %fieldnode_134 [%i : i32] : dictionary<i32, memref<25xi8>> -> memref<25xi8>
  %fieldnode_136 = sdql.access_record %supplier "s_address" : record<"s_suppkey": dictionary<i32, i32>, "s_name": dictionary<i32, memref<25xi8>>, "s_address": dictionary<i32, memref<40xi8>>, "s_nationkey": dictionary<i32, i32>, "s_phone": dictionary<i32, memref<15xi8>>, "s_acctbal": dictionary<i32, f64>, "s_comment": dictionary<i32, memref<101xi8>>, "size": i32> -> dictionary<i32, memref<40xi8>>
  %get_135 = sdql.lookup_dictionary %fieldnode_136 [%i : i32] : dictionary<i32, memref<40xi8>> -> memref<40xi8>
  %fieldnode_138 = sdql.access_record %supplier "s_nationkey" : record<"s_suppkey": dictionary<i32, i32>, "s_name": dictionary<i32, memref<25xi8>>, "s_address": dictionary<i32, memref<40xi8>>, "s_nationkey": dictionary<i32, i32>, "s_phone": dictionary<i32, memref<15xi8>>, "s_acctbal": dictionary<i32, f64>, "s_comment": dictionary<i32, memref<101xi8>>, "size": i32> -> dictionary<i32, i32>
  %get_137 = sdql.lookup_dictionary %fieldnode_138 [%i : i32] : dictionary<i32, i32> -> i32
  %fieldnode_140 = sdql.access_record %supplier "s_phone" : record<"s_suppkey": dictionary<i32, i32>, "s_name": dictionary<i32, memref<25xi8>>, "s_address": dictionary<i32, memref<40xi8>>, "s_nationkey": dictionary<i32, i32>, "s_phone": dictionary<i32, memref<15xi8>>, "s_acctbal": dictionary<i32, f64>, "s_comment": dictionary<i32, memref<101xi8>>, "size": i32> -> dictionary<i32, memref<15xi8>>
  %get_139 = sdql.lookup_dictionary %fieldnode_140 [%i : i32] : dictionary<i32, memref<15xi8>> -> memref<15xi8>
  %fieldnode_142 = sdql.access_record %supplier "s_acctbal" : record<"s_suppkey": dictionary<i32, i32>, "s_name": dictionary<i32, memref<25xi8>>, "s_address": dictionary<i32, memref<40xi8>>, "s_nationkey": dictionary<i32, i32>, "s_phone": dictionary<i32, memref<15xi8>>, "s_acctbal": dictionary<i32, f64>, "s_comment": dictionary<i32, memref<101xi8>>, "size": i32> -> dictionary<i32, f64>
  %get_141 = sdql.lookup_dictionary %fieldnode_142 [%i : i32] : dictionary<i32, f64> -> f64
  %fieldnode_144 = sdql.access_record %supplier "s_comment" : record<"s_suppkey": dictionary<i32, i32>, "s_name": dictionary<i32, memref<25xi8>>, "s_address": dictionary<i32, memref<40xi8>>, "s_nationkey": dictionary<i32, i32>, "s_phone": dictionary<i32, memref<15xi8>>, "s_acctbal": dictionary<i32, f64>, "s_comment": dictionary<i32, memref<101xi8>>, "size": i32> -> dictionary<i32, memref<101xi8>>
  %get_143 = sdql.lookup_dictionary %fieldnode_144 [%i : i32] : dictionary<i32, memref<101xi8>> -> memref<101xi8>
  %s = sdql.create_record {fields = ["s_suppkey", "s_name", "s_address", "s_nationkey", "s_phone", "s_acctbal", "s_comment"]} %get_131, %get_133, %get_135, %get_137, %get_139, %get_141, %get_143 : i32, memref<25xi8>, memref<40xi8>, i32, memref<15xi8>, f64, memref<101xi8> -> record<"s_suppkey": i32, "s_name": memref<25xi8>, "s_address": memref<40xi8>, "s_nationkey": i32, "s_phone": memref<15xi8>, "s_acctbal": f64, "s_comment": memref<101xi8>>
  %fieldnode_147 = sdql.access_record %s "s_suppkey" : record<"s_suppkey": i32, "s_name": memref<25xi8>, "s_address": memref<40xi8>, "s_nationkey": i32, "s_phone": memref<15xi8>, "s_acctbal": f64, "s_comment": memref<101xi8>> -> i32
  %fieldnode_148 = sdql.access_record %s "s_nationkey" : record<"s_suppkey": i32, "s_name": memref<25xi8>, "s_address": memref<40xi8>, "s_nationkey": i32, "s_phone": memref<15xi8>, "s_acctbal": f64, "s_comment": memref<101xi8>> -> i32
  %recnode_149 = sdql.create_record {fields = ["_", "_"]} %fieldnode_147, %fieldnode_148 : i32, i32 -> record<"_": i32, "_": i32>
  %uniq_146 = sdql.unique %recnode_149 : record<"_" : i32, "_" : i32> -> record<"_" : i32, "_" : i32>
  %consti_150 = "arith.constant"() <{value = 1 : i32}> : () -> i32
  %dict_145 = sdql.create_dictionary %uniq_146, %consti_150 : record<"_" : i32, "_" : i32>, i32 -> dictionary<record<"_" : i32, "_" : i32>, i32>
  sdql.yield %dict_145 : dictionary<record<"_" : i32, "_" : i32>, i32>
}
%fieldnode_151 = sdql.access_record %lineitem "size" : record<"l_orderkey": dictionary<i32, i32>, "l_partkey": dictionary<i32, i32>, "l_suppkey": dictionary<i32, i32>, "l_linenumber": dictionary<i32, i32>, "l_quantity": dictionary<i32, f64>, "l_extendedprice": dictionary<i32, f64>, "l_discount": dictionary<i32, f64>, "l_tax": dictionary<i32, f64>, "l_returnflag": dictionary<i32, memref<1xi8>>, "l_linestatus": dictionary<i32, memref<1xi8>>, "l_shipdate": dictionary<i32, i32>, "l_commitdate": dictionary<i32, i32>, "l_receiptdate": dictionary<i32, i32>, "l_shipinstruct": dictionary<i32, memref<25xi8>>, "l_shipmode": dictionary<i32, memref<10xi8>>, "l_comment": dictionary<i32, memref<44xi8>>, "size": i32> -> i32
%rangenode_152 = "func.call"(%fieldnode_151) <{callee = @range_builtin}> : (i32) -> dictionary<i32, i32>
%l_h = sdql.sum %rangenode_152 : dictionary<i32, i32> -> dictionary<memref<25xi8>, f64> {
^bb_153(%i: i32, %_: i32):
  %fieldnode_155 = sdql.access_record %lineitem "l_orderkey" : record<"l_orderkey": dictionary<i32, i32>, "l_partkey": dictionary<i32, i32>, "l_suppkey": dictionary<i32, i32>, "l_linenumber": dictionary<i32, i32>, "l_quantity": dictionary<i32, f64>, "l_extendedprice": dictionary<i32, f64>, "l_discount": dictionary<i32, f64>, "l_tax": dictionary<i32, f64>, "l_returnflag": dictionary<i32, memref<1xi8>>, "l_linestatus": dictionary<i32, memref<1xi8>>, "l_shipdate": dictionary<i32, i32>, "l_commitdate": dictionary<i32, i32>, "l_receiptdate": dictionary<i32, i32>, "l_shipinstruct": dictionary<i32, memref<25xi8>>, "l_shipmode": dictionary<i32, memref<10xi8>>, "l_comment": dictionary<i32, memref<44xi8>>, "size": i32> -> dictionary<i32, i32>
  %get_154 = sdql.lookup_dictionary %fieldnode_155 [%i : i32] : dictionary<i32, i32> -> i32
  %fieldnode_157 = sdql.access_record %lineitem "l_partkey" : record<"l_orderkey": dictionary<i32, i32>, "l_partkey": dictionary<i32, i32>, "l_suppkey": dictionary<i32, i32>, "l_linenumber": dictionary<i32, i32>, "l_quantity": dictionary<i32, f64>, "l_extendedprice": dictionary<i32, f64>, "l_discount": dictionary<i32, f64>, "l_tax": dictionary<i32, f64>, "l_returnflag": dictionary<i32, memref<1xi8>>, "l_linestatus": dictionary<i32, memref<1xi8>>, "l_shipdate": dictionary<i32, i32>, "l_commitdate": dictionary<i32, i32>, "l_receiptdate": dictionary<i32, i32>, "l_shipinstruct": dictionary<i32, memref<25xi8>>, "l_shipmode": dictionary<i32, memref<10xi8>>, "l_comment": dictionary<i32, memref<44xi8>>, "size": i32> -> dictionary<i32, i32>
  %get_156 = sdql.lookup_dictionary %fieldnode_157 [%i : i32] : dictionary<i32, i32> -> i32
  %fieldnode_159 = sdql.access_record %lineitem "l_suppkey" : record<"l_orderkey": dictionary<i32, i32>, "l_partkey": dictionary<i32, i32>, "l_suppkey": dictionary<i32, i32>, "l_linenumber": dictionary<i32, i32>, "l_quantity": dictionary<i32, f64>, "l_extendedprice": dictionary<i32, f64>, "l_discount": dictionary<i32, f64>, "l_tax": dictionary<i32, f64>, "l_returnflag": dictionary<i32, memref<1xi8>>, "l_linestatus": dictionary<i32, memref<1xi8>>, "l_shipdate": dictionary<i32, i32>, "l_commitdate": dictionary<i32, i32>, "l_receiptdate": dictionary<i32, i32>, "l_shipinstruct": dictionary<i32, memref<25xi8>>, "l_shipmode": dictionary<i32, memref<10xi8>>, "l_comment": dictionary<i32, memref<44xi8>>, "size": i32> -> dictionary<i32, i32>
  %get_158 = sdql.lookup_dictionary %fieldnode_159 [%i : i32] : dictionary<i32, i32> -> i32
  %fieldnode_161 = sdql.access_record %lineitem "l_linenumber" : record<"l_orderkey": dictionary<i32, i32>, "l_partkey": dictionary<i32, i32>, "l_suppkey": dictionary<i32, i32>, "l_linenumber": dictionary<i32, i32>, "l_quantity": dictionary<i32, f64>, "l_extendedprice": dictionary<i32, f64>, "l_discount": dictionary<i32, f64>, "l_tax": dictionary<i32, f64>, "l_returnflag": dictionary<i32, memref<1xi8>>, "l_linestatus": dictionary<i32, memref<1xi8>>, "l_shipdate": dictionary<i32, i32>, "l_commitdate": dictionary<i32, i32>, "l_receiptdate": dictionary<i32, i32>, "l_shipinstruct": dictionary<i32, memref<25xi8>>, "l_shipmode": dictionary<i32, memref<10xi8>>, "l_comment": dictionary<i32, memref<44xi8>>, "size": i32> -> dictionary<i32, i32>
  %get_160 = sdql.lookup_dictionary %fieldnode_161 [%i : i32] : dictionary<i32, i32> -> i32
  %fieldnode_163 = sdql.access_record %lineitem "l_quantity" : record<"l_orderkey": dictionary<i32, i32>, "l_partkey": dictionary<i32, i32>, "l_suppkey": dictionary<i32, i32>, "l_linenumber": dictionary<i32, i32>, "l_quantity": dictionary<i32, f64>, "l_extendedprice": dictionary<i32, f64>, "l_discount": dictionary<i32, f64>, "l_tax": dictionary<i32, f64>, "l_returnflag": dictionary<i32, memref<1xi8>>, "l_linestatus": dictionary<i32, memref<1xi8>>, "l_shipdate": dictionary<i32, i32>, "l_commitdate": dictionary<i32, i32>, "l_receiptdate": dictionary<i32, i32>, "l_shipinstruct": dictionary<i32, memref<25xi8>>, "l_shipmode": dictionary<i32, memref<10xi8>>, "l_comment": dictionary<i32, memref<44xi8>>, "size": i32> -> dictionary<i32, f64>
  %get_162 = sdql.lookup_dictionary %fieldnode_163 [%i : i32] : dictionary<i32, f64> -> f64
  %fieldnode_165 = sdql.access_record %lineitem "l_extendedprice" : record<"l_orderkey": dictionary<i32, i32>, "l_partkey": dictionary<i32, i32>, "l_suppkey": dictionary<i32, i32>, "l_linenumber": dictionary<i32, i32>, "l_quantity": dictionary<i32, f64>, "l_extendedprice": dictionary<i32, f64>, "l_discount": dictionary<i32, f64>, "l_tax": dictionary<i32, f64>, "l_returnflag": dictionary<i32, memref<1xi8>>, "l_linestatus": dictionary<i32, memref<1xi8>>, "l_shipdate": dictionary<i32, i32>, "l_commitdate": dictionary<i32, i32>, "l_receiptdate": dictionary<i32, i32>, "l_shipinstruct": dictionary<i32, memref<25xi8>>, "l_shipmode": dictionary<i32, memref<10xi8>>, "l_comment": dictionary<i32, memref<44xi8>>, "size": i32> -> dictionary<i32, f64>
  %get_164 = sdql.lookup_dictionary %fieldnode_165 [%i : i32] : dictionary<i32, f64> -> f64
  %fieldnode_167 = sdql.access_record %lineitem "l_discount" : record<"l_orderkey": dictionary<i32, i32>, "l_partkey": dictionary<i32, i32>, "l_suppkey": dictionary<i32, i32>, "l_linenumber": dictionary<i32, i32>, "l_quantity": dictionary<i32, f64>, "l_extendedprice": dictionary<i32, f64>, "l_discount": dictionary<i32, f64>, "l_tax": dictionary<i32, f64>, "l_returnflag": dictionary<i32, memref<1xi8>>, "l_linestatus": dictionary<i32, memref<1xi8>>, "l_shipdate": dictionary<i32, i32>, "l_commitdate": dictionary<i32, i32>, "l_receiptdate": dictionary<i32, i32>, "l_shipinstruct": dictionary<i32, memref<25xi8>>, "l_shipmode": dictionary<i32, memref<10xi8>>, "l_comment": dictionary<i32, memref<44xi8>>, "size": i32> -> dictionary<i32, f64>
  %get_166 = sdql.lookup_dictionary %fieldnode_167 [%i : i32] : dictionary<i32, f64> -> f64
  %fieldnode_169 = sdql.access_record %lineitem "l_tax" : record<"l_orderkey": dictionary<i32, i32>, "l_partkey": dictionary<i32, i32>, "l_suppkey": dictionary<i32, i32>, "l_linenumber": dictionary<i32, i32>, "l_quantity": dictionary<i32, f64>, "l_extendedprice": dictionary<i32, f64>, "l_discount": dictionary<i32, f64>, "l_tax": dictionary<i32, f64>, "l_returnflag": dictionary<i32, memref<1xi8>>, "l_linestatus": dictionary<i32, memref<1xi8>>, "l_shipdate": dictionary<i32, i32>, "l_commitdate": dictionary<i32, i32>, "l_receiptdate": dictionary<i32, i32>, "l_shipinstruct": dictionary<i32, memref<25xi8>>, "l_shipmode": dictionary<i32, memref<10xi8>>, "l_comment": dictionary<i32, memref<44xi8>>, "size": i32> -> dictionary<i32, f64>
  %get_168 = sdql.lookup_dictionary %fieldnode_169 [%i : i32] : dictionary<i32, f64> -> f64
  %fieldnode_171 = sdql.access_record %lineitem "l_returnflag" : record<"l_orderkey": dictionary<i32, i32>, "l_partkey": dictionary<i32, i32>, "l_suppkey": dictionary<i32, i32>, "l_linenumber": dictionary<i32, i32>, "l_quantity": dictionary<i32, f64>, "l_extendedprice": dictionary<i32, f64>, "l_discount": dictionary<i32, f64>, "l_tax": dictionary<i32, f64>, "l_returnflag": dictionary<i32, memref<1xi8>>, "l_linestatus": dictionary<i32, memref<1xi8>>, "l_shipdate": dictionary<i32, i32>, "l_commitdate": dictionary<i32, i32>, "l_receiptdate": dictionary<i32, i32>, "l_shipinstruct": dictionary<i32, memref<25xi8>>, "l_shipmode": dictionary<i32, memref<10xi8>>, "l_comment": dictionary<i32, memref<44xi8>>, "size": i32> -> dictionary<i32, memref<1xi8>>
  %get_170 = sdql.lookup_dictionary %fieldnode_171 [%i : i32] : dictionary<i32, memref<1xi8>> -> memref<1xi8>
  %fieldnode_173 = sdql.access_record %lineitem "l_linestatus" : record<"l_orderkey": dictionary<i32, i32>, "l_partkey": dictionary<i32, i32>, "l_suppkey": dictionary<i32, i32>, "l_linenumber": dictionary<i32, i32>, "l_quantity": dictionary<i32, f64>, "l_extendedprice": dictionary<i32, f64>, "l_discount": dictionary<i32, f64>, "l_tax": dictionary<i32, f64>, "l_returnflag": dictionary<i32, memref<1xi8>>, "l_linestatus": dictionary<i32, memref<1xi8>>, "l_shipdate": dictionary<i32, i32>, "l_commitdate": dictionary<i32, i32>, "l_receiptdate": dictionary<i32, i32>, "l_shipinstruct": dictionary<i32, memref<25xi8>>, "l_shipmode": dictionary<i32, memref<10xi8>>, "l_comment": dictionary<i32, memref<44xi8>>, "size": i32> -> dictionary<i32, memref<1xi8>>
  %get_172 = sdql.lookup_dictionary %fieldnode_173 [%i : i32] : dictionary<i32, memref<1xi8>> -> memref<1xi8>
  %fieldnode_175 = sdql.access_record %lineitem "l_shipdate" : record<"l_orderkey": dictionary<i32, i32>, "l_partkey": dictionary<i32, i32>, "l_suppkey": dictionary<i32, i32>, "l_linenumber": dictionary<i32, i32>, "l_quantity": dictionary<i32, f64>, "l_extendedprice": dictionary<i32, f64>, "l_discount": dictionary<i32, f64>, "l_tax": dictionary<i32, f64>, "l_returnflag": dictionary<i32, memref<1xi8>>, "l_linestatus": dictionary<i32, memref<1xi8>>, "l_shipdate": dictionary<i32, i32>, "l_commitdate": dictionary<i32, i32>, "l_receiptdate": dictionary<i32, i32>, "l_shipinstruct": dictionary<i32, memref<25xi8>>, "l_shipmode": dictionary<i32, memref<10xi8>>, "l_comment": dictionary<i32, memref<44xi8>>, "size": i32> -> dictionary<i32, i32>
  %get_174 = sdql.lookup_dictionary %fieldnode_175 [%i : i32] : dictionary<i32, i32> -> i32
  %fieldnode_177 = sdql.access_record %lineitem "l_commitdate" : record<"l_orderkey": dictionary<i32, i32>, "l_partkey": dictionary<i32, i32>, "l_suppkey": dictionary<i32, i32>, "l_linenumber": dictionary<i32, i32>, "l_quantity": dictionary<i32, f64>, "l_extendedprice": dictionary<i32, f64>, "l_discount": dictionary<i32, f64>, "l_tax": dictionary<i32, f64>, "l_returnflag": dictionary<i32, memref<1xi8>>, "l_linestatus": dictionary<i32, memref<1xi8>>, "l_shipdate": dictionary<i32, i32>, "l_commitdate": dictionary<i32, i32>, "l_receiptdate": dictionary<i32, i32>, "l_shipinstruct": dictionary<i32, memref<25xi8>>, "l_shipmode": dictionary<i32, memref<10xi8>>, "l_comment": dictionary<i32, memref<44xi8>>, "size": i32> -> dictionary<i32, i32>
  %get_176 = sdql.lookup_dictionary %fieldnode_177 [%i : i32] : dictionary<i32, i32> -> i32
  %fieldnode_179 = sdql.access_record %lineitem "l_receiptdate" : record<"l_orderkey": dictionary<i32, i32>, "l_partkey": dictionary<i32, i32>, "l_suppkey": dictionary<i32, i32>, "l_linenumber": dictionary<i32, i32>, "l_quantity": dictionary<i32, f64>, "l_extendedprice": dictionary<i32, f64>, "l_discount": dictionary<i32, f64>, "l_tax": dictionary<i32, f64>, "l_returnflag": dictionary<i32, memref<1xi8>>, "l_linestatus": dictionary<i32, memref<1xi8>>, "l_shipdate": dictionary<i32, i32>, "l_commitdate": dictionary<i32, i32>, "l_receiptdate": dictionary<i32, i32>, "l_shipinstruct": dictionary<i32, memref<25xi8>>, "l_shipmode": dictionary<i32, memref<10xi8>>, "l_comment": dictionary<i32, memref<44xi8>>, "size": i32> -> dictionary<i32, i32>
  %get_178 = sdql.lookup_dictionary %fieldnode_179 [%i : i32] : dictionary<i32, i32> -> i32
  %fieldnode_181 = sdql.access_record %lineitem "l_shipinstruct" : record<"l_orderkey": dictionary<i32, i32>, "l_partkey": dictionary<i32, i32>, "l_suppkey": dictionary<i32, i32>, "l_linenumber": dictionary<i32, i32>, "l_quantity": dictionary<i32, f64>, "l_extendedprice": dictionary<i32, f64>, "l_discount": dictionary<i32, f64>, "l_tax": dictionary<i32, f64>, "l_returnflag": dictionary<i32, memref<1xi8>>, "l_linestatus": dictionary<i32, memref<1xi8>>, "l_shipdate": dictionary<i32, i32>, "l_commitdate": dictionary<i32, i32>, "l_receiptdate": dictionary<i32, i32>, "l_shipinstruct": dictionary<i32, memref<25xi8>>, "l_shipmode": dictionary<i32, memref<10xi8>>, "l_comment": dictionary<i32, memref<44xi8>>, "size": i32> -> dictionary<i32, memref<25xi8>>
  %get_180 = sdql.lookup_dictionary %fieldnode_181 [%i : i32] : dictionary<i32, memref<25xi8>> -> memref<25xi8>
  %fieldnode_183 = sdql.access_record %lineitem "l_shipmode" : record<"l_orderkey": dictionary<i32, i32>, "l_partkey": dictionary<i32, i32>, "l_suppkey": dictionary<i32, i32>, "l_linenumber": dictionary<i32, i32>, "l_quantity": dictionary<i32, f64>, "l_extendedprice": dictionary<i32, f64>, "l_discount": dictionary<i32, f64>, "l_tax": dictionary<i32, f64>, "l_returnflag": dictionary<i32, memref<1xi8>>, "l_linestatus": dictionary<i32, memref<1xi8>>, "l_shipdate": dictionary<i32, i32>, "l_commitdate": dictionary<i32, i32>, "l_receiptdate": dictionary<i32, i32>, "l_shipinstruct": dictionary<i32, memref<25xi8>>, "l_shipmode": dictionary<i32, memref<10xi8>>, "l_comment": dictionary<i32, memref<44xi8>>, "size": i32> -> dictionary<i32, memref<10xi8>>
  %get_182 = sdql.lookup_dictionary %fieldnode_183 [%i : i32] : dictionary<i32, memref<10xi8>> -> memref<10xi8>
  %fieldnode_185 = sdql.access_record %lineitem "l_comment" : record<"l_orderkey": dictionary<i32, i32>, "l_partkey": dictionary<i32, i32>, "l_suppkey": dictionary<i32, i32>, "l_linenumber": dictionary<i32, i32>, "l_quantity": dictionary<i32, f64>, "l_extendedprice": dictionary<i32, f64>, "l_discount": dictionary<i32, f64>, "l_tax": dictionary<i32, f64>, "l_returnflag": dictionary<i32, memref<1xi8>>, "l_linestatus": dictionary<i32, memref<1xi8>>, "l_shipdate": dictionary<i32, i32>, "l_commitdate": dictionary<i32, i32>, "l_receiptdate": dictionary<i32, i32>, "l_shipinstruct": dictionary<i32, memref<25xi8>>, "l_shipmode": dictionary<i32, memref<10xi8>>, "l_comment": dictionary<i32, memref<44xi8>>, "size": i32> -> dictionary<i32, memref<44xi8>>
  %get_184 = sdql.lookup_dictionary %fieldnode_185 [%i : i32] : dictionary<i32, memref<44xi8>> -> memref<44xi8>
  %l = sdql.create_record {fields = ["l_orderkey", "l_partkey", "l_suppkey", "l_linenumber", "l_quantity", "l_extendedprice", "l_discount", "l_tax", "l_returnflag", "l_linestatus", "l_shipdate", "l_commitdate", "l_receiptdate", "l_shipinstruct", "l_shipmode", "l_comment"]} %get_154, %get_156, %get_158, %get_160, %get_162, %get_164, %get_166, %get_168, %get_170, %get_172, %get_174, %get_176, %get_178, %get_180, %get_182, %get_184 : i32, i32, i32, i32, f64, f64, f64, f64, memref<1xi8>, memref<1xi8>, i32, i32, i32, memref<25xi8>, memref<10xi8>, memref<44xi8> -> record<"l_orderkey": i32, "l_partkey": i32, "l_suppkey": i32, "l_linenumber": i32, "l_quantity": f64, "l_extendedprice": f64, "l_discount": f64, "l_tax": f64, "l_returnflag": memref<1xi8>, "l_linestatus": memref<1xi8>, "l_shipdate": i32, "l_commitdate": i32, "l_receiptdate": i32, "l_shipinstruct": memref<25xi8>, "l_shipmode": memref<10xi8>, "l_comment": memref<44xi8>>
  %fieldnode_189 = sdql.access_record %l "l_orderkey" : record<"l_orderkey": i32, "l_partkey": i32, "l_suppkey": i32, "l_linenumber": i32, "l_quantity": f64, "l_extendedprice": f64, "l_discount": f64, "l_tax": f64, "l_returnflag": memref<1xi8>, "l_linestatus": memref<1xi8>, "l_shipdate": i32, "l_commitdate": i32, "l_receiptdate": i32, "l_shipinstruct": memref<25xi8>, "l_shipmode": memref<10xi8>, "l_comment": memref<44xi8>> -> i32
  %get_188 = sdql.lookup_dictionary %o_h [%fieldnode_189 : i32] : dictionary<i32, record<"_" : memref<25xi8>, "_" : i32>> -> record<"_" : memref<25xi8>, "_" : i32>
  %dict_190 = sdql.empty_dictionary : record<"_" : memref<25xi8>, "_" : i32>
  %cmp_191 = sdql.cmp %get_188, %dict_190 : record<"_" : memref<25xi8>, "_" : i32>, record<"_" : memref<25xi8>, "_" : i32> -> i1
  %zero_186 = "arith.constant"() <{value = 0 : i1}> : () -> i1
  %neg_187 = "arith.cmpi"(%cmp_191, %zero_186) <{predicate = 0}> : (i1, i1) -> i1
  %if_192 = "scf.if"(%neg_187) ({
    %fieldnode_194 = sdql.access_record %l "l_suppkey" : record<"l_orderkey": i32, "l_partkey": i32, "l_suppkey": i32, "l_linenumber": i32, "l_quantity": f64, "l_extendedprice": f64, "l_discount": f64, "l_tax": f64, "l_returnflag": memref<1xi8>, "l_linestatus": memref<1xi8>, "l_shipdate": i32, "l_commitdate": i32, "l_receiptdate": i32, "l_shipinstruct": memref<25xi8>, "l_shipmode": memref<10xi8>, "l_comment": memref<44xi8>> -> i32
    %fieldnode_197 = sdql.access_record %l "l_orderkey" : record<"l_orderkey": i32, "l_partkey": i32, "l_suppkey": i32, "l_linenumber": i32, "l_quantity": f64, "l_extendedprice": f64, "l_discount": f64, "l_tax": f64, "l_returnflag": memref<1xi8>, "l_linestatus": memref<1xi8>, "l_shipdate": i32, "l_commitdate": i32, "l_receiptdate": i32, "l_shipinstruct": memref<25xi8>, "l_shipmode": memref<10xi8>, "l_comment": memref<44xi8>> -> i32
    %get_196 = sdql.lookup_dictionary %o_h [%fieldnode_197 : i32] : dictionary<i32, record<"_" : memref<25xi8>, "_" : i32>> -> record<"_" : memref<25xi8>, "_" : i32>
    %consti_198 = "arith.constant"() <{value = 1 : i32}> : () -> i32
    %get_195 = sdql.access_record %get_196 "%consti_198" : record<"_" : memref<25xi8>, "_" : i32> -> i32
    %recnode_199 = sdql.create_record {fields = ["_", "_"]} %fieldnode_194, %get_195 : i32, i32 -> record<"_": i32, "_": i32>
    %get_193 = sdql.lookup_dictionary %s_h [%recnode_199 : record<"_" : i32, "_" : i32>] : dictionary<record<"_" : i32, "_" : i32>, i32> -> i32
    %dict_200 = sdql.empty_dictionary : i32
    %cmpi_201 = "arith.cmpi"(%get_193, %dict_200) <{predicate = 1}> : (i32, i32) -> i1
    "scf.yield"(%cmpi_201) : (i1) -> ()
  }, {
    %constv_202 = "arith.constant"() <{value = 0 : i1}> : () -> i1
    "scf.yield"(%constv_202) : (i1) -> ()
  }) : (i1) -> i1
  %if_203 = "scf.if"(%if_192) ({
    %fieldnode_207 = sdql.access_record %l "l_orderkey" : record<"l_orderkey": i32, "l_partkey": i32, "l_suppkey": i32, "l_linenumber": i32, "l_quantity": f64, "l_extendedprice": f64, "l_discount": f64, "l_tax": f64, "l_returnflag": memref<1xi8>, "l_linestatus": memref<1xi8>, "l_shipdate": i32, "l_commitdate": i32, "l_receiptdate": i32, "l_shipinstruct": memref<25xi8>, "l_shipmode": memref<10xi8>, "l_comment": memref<44xi8>> -> i32
    %get_206 = sdql.lookup_dictionary %o_h [%fieldnode_207 : i32] : dictionary<i32, record<"_" : memref<25xi8>, "_" : i32>> -> record<"_" : memref<25xi8>, "_" : i32>
    %consti_208 = "arith.constant"() <{value = 0 : i32}> : () -> i32
    %get_205 = sdql.access_record %get_206 "%consti_208" : record<"_" : memref<25xi8>, "_" : i32> -> memref<25xi8>
    %fieldnode_210 = sdql.access_record %l "l_extendedprice" : record<"l_orderkey": i32, "l_partkey": i32, "l_suppkey": i32, "l_linenumber": i32, "l_quantity": f64, "l_extendedprice": f64, "l_discount": f64, "l_tax": f64, "l_returnflag": memref<1xi8>, "l_linestatus": memref<1xi8>, "l_shipdate": i32, "l_commitdate": i32, "l_receiptdate": i32, "l_shipinstruct": memref<25xi8>, "l_shipmode": memref<10xi8>, "l_comment": memref<44xi8>> -> f64
    %constd_212 = "arith.constant"() <{value = 1.0  : f64}> : () -> f64
    %fieldnode_214 = sdql.access_record %l "l_discount" : record<"l_orderkey": i32, "l_partkey": i32, "l_suppkey": i32, "l_linenumber": i32, "l_quantity": f64, "l_extendedprice": f64, "l_discount": f64, "l_tax": f64, "l_returnflag": memref<1xi8>, "l_linestatus": memref<1xi8>, "l_shipdate": i32, "l_commitdate": i32, "l_receiptdate": i32, "l_shipinstruct": memref<25xi8>, "l_shipmode": memref<10xi8>, "l_comment": memref<44xi8>> -> f64
    %zero_215 = "arith.constant"() <{value = 0.0 : f64}> : () -> f64
    %negf_213 = "arith.subf"(%zero_215, %fieldnode_214) <{fastmath = #arith.fastmath<none>}> : (f64, f64) -> f64
    %addf_211 = "arith.addf"(%constd_212, %negf_213) <{fastmath = #arith.fastmath<none>}> : (f64, f64) -> f64
    %multf_209 = "arith.mulf"(%fieldnode_210, %addf_211) <{fastmath = #arith.fastmath<none>}> : (f64, f64) -> f64
    %dict_204 = sdql.create_dictionary %get_205, %multf_209 : memref<25xi8>, f64 -> dictionary<memref<25xi8>, f64>
    "scf.yield"(%dict_204) : (dictionary<memref<25xi8>, f64>) -> ()
  }, {
    %dict_216 = sdql.empty_dictionary : dictionary<memref<25xi8>, f64>
    "scf.yield"(%dict_216) : (dictionary<memref<25xi8>, f64>) -> ()
  }) : (i1) -> dictionary<memref<25xi8>, f64>
  sdql.yield %if_203 : dictionary<memref<25xi8>, f64>
}
%sum_217 = sdql.sum %l_h : dictionary<memref<25xi8>, f64> -> dictionary<record<"_" : memref<25xi8>, "_" : f64>, i32> {
^bb_218(%k: memref<25xi8>, %v: f64):
  %recnode_221 = sdql.create_record {fields = ["_", "_"]} %k, %v : memref<25xi8>, f64 -> record<"_": memref<25xi8>, "_": f64>
  %uniq_220 = sdql.unique %recnode_221 : record<"_" : memref<25xi8>, "_" : f64> -> record<"_" : memref<25xi8>, "_" : f64>
  %consti_222 = "arith.constant"() <{value = 1 : i32}> : () -> i32
  %dict_219 = sdql.create_dictionary %uniq_220, %consti_222 : record<"_" : memref<25xi8>, "_" : f64>, i32 -> dictionary<record<"_" : memref<25xi8>, "_" : f64>, i32>
  sdql.yield %dict_219 : dictionary<record<"_" : memref<25xi8>, "_" : f64>, i32>
}
