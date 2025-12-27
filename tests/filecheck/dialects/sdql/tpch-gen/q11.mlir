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
%partsupp = sdql.load "datasets/tpch/partsupp.tbl" : record<"ps_partkey" : dictionary<i32, i32>, "ps_suppkey" : dictionary<i32, i32>, "ps_availqty" : dictionary<i32, f64>, "ps_supplycost" : dictionary<i32, f64>, "ps_comment" : dictionary<i32, memref<199xi8>>, "size" : i32>
%nation = sdql.load "datasets/tpch/nation.tbl" : record<"n_nationkey" : dictionary<i32, i32>, "n_name" : dictionary<i32, memref<25xi8>>, "n_regionkey" : dictionary<i32, i32>, "n_comment" : dictionary<i32, memref<152xi8>>, "size" : i32>
// GERMANY
%constds_0 = "arith.constant"() <{value = dense<[71, 69, 82, 77, 65, 78, 89]> : memref<?xi8>}> : () -> memref<?xi8>
%consti_1 = "arith.constant"() <{value = 8 : i32}> : () -> i32
%germany = sdql.external "ConstantString", %constds_0, %consti_1 : memref<?xi8>, i32 -> memref<7xi8>
%fieldnode_2 = sdql.access_record %nation "size" : record<"n_nationkey": dictionary<i32, i32>, "n_name": dictionary<i32, memref<25xi8>>, "n_regionkey": dictionary<i32, i32>, "n_comment": dictionary<i32, memref<152xi8>>, "size": i32> -> i32
%rangenode_3 = "func.call"(%fieldnode_2) <{callee = @range_builtin}> : (i32) -> dictionary<i32, i32>
%n_h = sdql.sum %rangenode_3 : dictionary<i32, i32> -> dictionary<i32, record<"_" : i32>> {
^bb_4(%i: i32, %_: i32):
  %fieldnode_6 = sdql.access_record %nation "n_nationkey" : record<"n_nationkey": dictionary<i32, i32>, "n_name": dictionary<i32, memref<25xi8>>, "n_regionkey": dictionary<i32, i32>, "n_comment": dictionary<i32, memref<152xi8>>, "size": i32> -> dictionary<i32, i32>
  %get_5 = sdql.lookup_dictionary %fieldnode_6 [%i : i32] : dictionary<i32, i32> -> i32
  %fieldnode_8 = sdql.access_record %nation "n_name" : record<"n_nationkey": dictionary<i32, i32>, "n_name": dictionary<i32, memref<25xi8>>, "n_regionkey": dictionary<i32, i32>, "n_comment": dictionary<i32, memref<152xi8>>, "size": i32> -> dictionary<i32, memref<25xi8>>
  %get_7 = sdql.lookup_dictionary %fieldnode_8 [%i : i32] : dictionary<i32, memref<25xi8>> -> memref<25xi8>
  %fieldnode_10 = sdql.access_record %nation "n_regionkey" : record<"n_nationkey": dictionary<i32, i32>, "n_name": dictionary<i32, memref<25xi8>>, "n_regionkey": dictionary<i32, i32>, "n_comment": dictionary<i32, memref<152xi8>>, "size": i32> -> dictionary<i32, i32>
  %get_9 = sdql.lookup_dictionary %fieldnode_10 [%i : i32] : dictionary<i32, i32> -> i32
  %fieldnode_12 = sdql.access_record %nation "n_comment" : record<"n_nationkey": dictionary<i32, i32>, "n_name": dictionary<i32, memref<25xi8>>, "n_regionkey": dictionary<i32, i32>, "n_comment": dictionary<i32, memref<152xi8>>, "size": i32> -> dictionary<i32, memref<152xi8>>
  %get_11 = sdql.lookup_dictionary %fieldnode_12 [%i : i32] : dictionary<i32, memref<152xi8>> -> memref<152xi8>
  %n = sdql.create_record {fields = ["n_nationkey", "n_name", "n_regionkey", "n_comment"]} %get_5, %get_7, %get_9, %get_11 : i32, memref<25xi8>, i32, memref<152xi8> -> record<"n_nationkey": i32, "n_name": memref<25xi8>, "n_regionkey": i32, "n_comment": memref<152xi8>>
  %fieldnode_13 = sdql.access_record %n "n_name" : record<"n_nationkey": i32, "n_name": memref<25xi8>, "n_regionkey": i32, "n_comment": memref<152xi8>> -> memref<25xi8>
  %cmp_14 = sdql.cmp %fieldnode_13, %germany : memref<25xi8>, memref<7xi8> -> i1
  %if_15 = "scf.if"(%cmp_14) ({
    %fieldnode_18 = sdql.access_record %n "n_nationkey" : record<"n_nationkey": i32, "n_name": memref<25xi8>, "n_regionkey": i32, "n_comment": memref<152xi8>> -> i32
    %uniq_17 = sdql.unique %fieldnode_18 : i32 -> i32
    %fieldnode_19 = sdql.access_record %n "n_nationkey" : record<"n_nationkey": i32, "n_name": memref<25xi8>, "n_regionkey": i32, "n_comment": memref<152xi8>> -> i32
    %recnode_20 = sdql.create_record {fields = ["_"]} %fieldnode_19 : i32 -> record<"_": i32>
    %dict_16 = sdql.create_dictionary %uniq_17, %recnode_20 : i32, record<"_" : i32> -> dictionary<i32, record<"_" : i32>>
    "scf.yield"(%dict_16) : (dictionary<i32, record<"_" : i32>>) -> ()
  }, {
    %dict_21 = sdql.empty_dictionary : dictionary<i32, record<"_" : i32>>
    "scf.yield"(%dict_21) : (dictionary<i32, record<"_" : i32>>) -> ()
  }) : (i1) -> dictionary<i32, record<"_" : i32>>
  sdql.yield %if_15 : dictionary<i32, record<"_" : i32>>
}
%fieldnode_22 = sdql.access_record %supplier "size" : record<"s_suppkey": dictionary<i32, i32>, "s_name": dictionary<i32, memref<25xi8>>, "s_address": dictionary<i32, memref<40xi8>>, "s_nationkey": dictionary<i32, i32>, "s_phone": dictionary<i32, memref<15xi8>>, "s_acctbal": dictionary<i32, f64>, "s_comment": dictionary<i32, memref<101xi8>>, "size": i32> -> i32
%rangenode_23 = "func.call"(%fieldnode_22) <{callee = @range_builtin}> : (i32) -> dictionary<i32, i32>
%s_h = sdql.sum %rangenode_23 : dictionary<i32, i32> -> dictionary<i32, i32> {
^bb_24(%i: i32, %_: i32):
  %fieldnode_26 = sdql.access_record %supplier "s_suppkey" : record<"s_suppkey": dictionary<i32, i32>, "s_name": dictionary<i32, memref<25xi8>>, "s_address": dictionary<i32, memref<40xi8>>, "s_nationkey": dictionary<i32, i32>, "s_phone": dictionary<i32, memref<15xi8>>, "s_acctbal": dictionary<i32, f64>, "s_comment": dictionary<i32, memref<101xi8>>, "size": i32> -> dictionary<i32, i32>
  %get_25 = sdql.lookup_dictionary %fieldnode_26 [%i : i32] : dictionary<i32, i32> -> i32
  %fieldnode_28 = sdql.access_record %supplier "s_name" : record<"s_suppkey": dictionary<i32, i32>, "s_name": dictionary<i32, memref<25xi8>>, "s_address": dictionary<i32, memref<40xi8>>, "s_nationkey": dictionary<i32, i32>, "s_phone": dictionary<i32, memref<15xi8>>, "s_acctbal": dictionary<i32, f64>, "s_comment": dictionary<i32, memref<101xi8>>, "size": i32> -> dictionary<i32, memref<25xi8>>
  %get_27 = sdql.lookup_dictionary %fieldnode_28 [%i : i32] : dictionary<i32, memref<25xi8>> -> memref<25xi8>
  %fieldnode_30 = sdql.access_record %supplier "s_address" : record<"s_suppkey": dictionary<i32, i32>, "s_name": dictionary<i32, memref<25xi8>>, "s_address": dictionary<i32, memref<40xi8>>, "s_nationkey": dictionary<i32, i32>, "s_phone": dictionary<i32, memref<15xi8>>, "s_acctbal": dictionary<i32, f64>, "s_comment": dictionary<i32, memref<101xi8>>, "size": i32> -> dictionary<i32, memref<40xi8>>
  %get_29 = sdql.lookup_dictionary %fieldnode_30 [%i : i32] : dictionary<i32, memref<40xi8>> -> memref<40xi8>
  %fieldnode_32 = sdql.access_record %supplier "s_nationkey" : record<"s_suppkey": dictionary<i32, i32>, "s_name": dictionary<i32, memref<25xi8>>, "s_address": dictionary<i32, memref<40xi8>>, "s_nationkey": dictionary<i32, i32>, "s_phone": dictionary<i32, memref<15xi8>>, "s_acctbal": dictionary<i32, f64>, "s_comment": dictionary<i32, memref<101xi8>>, "size": i32> -> dictionary<i32, i32>
  %get_31 = sdql.lookup_dictionary %fieldnode_32 [%i : i32] : dictionary<i32, i32> -> i32
  %fieldnode_34 = sdql.access_record %supplier "s_phone" : record<"s_suppkey": dictionary<i32, i32>, "s_name": dictionary<i32, memref<25xi8>>, "s_address": dictionary<i32, memref<40xi8>>, "s_nationkey": dictionary<i32, i32>, "s_phone": dictionary<i32, memref<15xi8>>, "s_acctbal": dictionary<i32, f64>, "s_comment": dictionary<i32, memref<101xi8>>, "size": i32> -> dictionary<i32, memref<15xi8>>
  %get_33 = sdql.lookup_dictionary %fieldnode_34 [%i : i32] : dictionary<i32, memref<15xi8>> -> memref<15xi8>
  %fieldnode_36 = sdql.access_record %supplier "s_acctbal" : record<"s_suppkey": dictionary<i32, i32>, "s_name": dictionary<i32, memref<25xi8>>, "s_address": dictionary<i32, memref<40xi8>>, "s_nationkey": dictionary<i32, i32>, "s_phone": dictionary<i32, memref<15xi8>>, "s_acctbal": dictionary<i32, f64>, "s_comment": dictionary<i32, memref<101xi8>>, "size": i32> -> dictionary<i32, f64>
  %get_35 = sdql.lookup_dictionary %fieldnode_36 [%i : i32] : dictionary<i32, f64> -> f64
  %fieldnode_38 = sdql.access_record %supplier "s_comment" : record<"s_suppkey": dictionary<i32, i32>, "s_name": dictionary<i32, memref<25xi8>>, "s_address": dictionary<i32, memref<40xi8>>, "s_nationkey": dictionary<i32, i32>, "s_phone": dictionary<i32, memref<15xi8>>, "s_acctbal": dictionary<i32, f64>, "s_comment": dictionary<i32, memref<101xi8>>, "size": i32> -> dictionary<i32, memref<101xi8>>
  %get_37 = sdql.lookup_dictionary %fieldnode_38 [%i : i32] : dictionary<i32, memref<101xi8>> -> memref<101xi8>
  %s = sdql.create_record {fields = ["s_suppkey", "s_name", "s_address", "s_nationkey", "s_phone", "s_acctbal", "s_comment"]} %get_25, %get_27, %get_29, %get_31, %get_33, %get_35, %get_37 : i32, memref<25xi8>, memref<40xi8>, i32, memref<15xi8>, f64, memref<101xi8> -> record<"s_suppkey": i32, "s_name": memref<25xi8>, "s_address": memref<40xi8>, "s_nationkey": i32, "s_phone": memref<15xi8>, "s_acctbal": f64, "s_comment": memref<101xi8>>
  %fieldnode_42 = sdql.access_record %s "s_nationkey" : record<"s_suppkey": i32, "s_name": memref<25xi8>, "s_address": memref<40xi8>, "s_nationkey": i32, "s_phone": memref<15xi8>, "s_acctbal": f64, "s_comment": memref<101xi8>> -> i32
  %get_41 = sdql.lookup_dictionary %n_h [%fieldnode_42 : i32] : dictionary<i32, record<"_" : i32>> -> record<"_" : i32>
  %dict_43 = sdql.empty_dictionary : record<"_" : i32>
  %cmp_44 = sdql.cmp %get_41, %dict_43 : record<"_" : i32>, record<"_" : i32> -> i1
  %zero_39 = "arith.constant"() <{value = 0 : i1}> : () -> i1
  %neg_40 = "arith.cmpi"(%cmp_44, %zero_39) <{predicate = 0}> : (i1, i1) -> i1
  %if_45 = "scf.if"(%neg_40) ({
    %fieldnode_48 = sdql.access_record %s "s_suppkey" : record<"s_suppkey": i32, "s_name": memref<25xi8>, "s_address": memref<40xi8>, "s_nationkey": i32, "s_phone": memref<15xi8>, "s_acctbal": f64, "s_comment": memref<101xi8>> -> i32
    %uniq_47 = sdql.unique %fieldnode_48 : i32 -> i32
    %consti_49 = "arith.constant"() <{value = 1 : i32}> : () -> i32
    %dict_46 = sdql.create_dictionary %uniq_47, %consti_49 : i32, i32 -> dictionary<i32, i32>
    "scf.yield"(%dict_46) : (dictionary<i32, i32>) -> ()
  }, {
    %dict_50 = sdql.empty_dictionary : dictionary<i32, i32>
    "scf.yield"(%dict_50) : (dictionary<i32, i32>) -> ()
  }) : (i1) -> dictionary<i32, i32>
  sdql.yield %if_45 : dictionary<i32, i32>
}
%fieldnode_51 = sdql.access_record %partsupp "size" : record<"ps_partkey": dictionary<i32, i32>, "ps_suppkey": dictionary<i32, i32>, "ps_availqty": dictionary<i32, f64>, "ps_supplycost": dictionary<i32, f64>, "ps_comment": dictionary<i32, memref<199xi8>>, "size": i32> -> i32
%rangenode_52 = "func.call"(%fieldnode_51) <{callee = @range_builtin}> : (i32) -> dictionary<i32, i32>
%ps_t = sdql.sum %rangenode_52 : dictionary<i32, i32> -> record<"_" : f64, "_" : dictionary<i32, f64>> {
^bb_53(%i: i32, %_: i32):
  %fieldnode_55 = sdql.access_record %partsupp "ps_partkey" : record<"ps_partkey": dictionary<i32, i32>, "ps_suppkey": dictionary<i32, i32>, "ps_availqty": dictionary<i32, f64>, "ps_supplycost": dictionary<i32, f64>, "ps_comment": dictionary<i32, memref<199xi8>>, "size": i32> -> dictionary<i32, i32>
  %get_54 = sdql.lookup_dictionary %fieldnode_55 [%i : i32] : dictionary<i32, i32> -> i32
  %fieldnode_57 = sdql.access_record %partsupp "ps_suppkey" : record<"ps_partkey": dictionary<i32, i32>, "ps_suppkey": dictionary<i32, i32>, "ps_availqty": dictionary<i32, f64>, "ps_supplycost": dictionary<i32, f64>, "ps_comment": dictionary<i32, memref<199xi8>>, "size": i32> -> dictionary<i32, i32>
  %get_56 = sdql.lookup_dictionary %fieldnode_57 [%i : i32] : dictionary<i32, i32> -> i32
  %fieldnode_59 = sdql.access_record %partsupp "ps_availqty" : record<"ps_partkey": dictionary<i32, i32>, "ps_suppkey": dictionary<i32, i32>, "ps_availqty": dictionary<i32, f64>, "ps_supplycost": dictionary<i32, f64>, "ps_comment": dictionary<i32, memref<199xi8>>, "size": i32> -> dictionary<i32, f64>
  %get_58 = sdql.lookup_dictionary %fieldnode_59 [%i : i32] : dictionary<i32, f64> -> f64
  %fieldnode_61 = sdql.access_record %partsupp "ps_supplycost" : record<"ps_partkey": dictionary<i32, i32>, "ps_suppkey": dictionary<i32, i32>, "ps_availqty": dictionary<i32, f64>, "ps_supplycost": dictionary<i32, f64>, "ps_comment": dictionary<i32, memref<199xi8>>, "size": i32> -> dictionary<i32, f64>
  %get_60 = sdql.lookup_dictionary %fieldnode_61 [%i : i32] : dictionary<i32, f64> -> f64
  %fieldnode_63 = sdql.access_record %partsupp "ps_comment" : record<"ps_partkey": dictionary<i32, i32>, "ps_suppkey": dictionary<i32, i32>, "ps_availqty": dictionary<i32, f64>, "ps_supplycost": dictionary<i32, f64>, "ps_comment": dictionary<i32, memref<199xi8>>, "size": i32> -> dictionary<i32, memref<199xi8>>
  %get_62 = sdql.lookup_dictionary %fieldnode_63 [%i : i32] : dictionary<i32, memref<199xi8>> -> memref<199xi8>
  %ps = sdql.create_record {fields = ["ps_partkey", "ps_suppkey", "ps_availqty", "ps_supplycost", "ps_comment"]} %get_54, %get_56, %get_58, %get_60, %get_62 : i32, i32, f64, f64, memref<199xi8> -> record<"ps_partkey": i32, "ps_suppkey": i32, "ps_availqty": f64, "ps_supplycost": f64, "ps_comment": memref<199xi8>>
  %fieldnode_65 = sdql.access_record %ps "ps_suppkey" : record<"ps_partkey": i32, "ps_suppkey": i32, "ps_availqty": f64, "ps_supplycost": f64, "ps_comment": memref<199xi8>> -> i32
  %get_64 = sdql.lookup_dictionary %s_h [%fieldnode_65 : i32] : dictionary<i32, i32> -> i32
  %dict_66 = sdql.empty_dictionary : i32
  %cmpi_67 = "arith.cmpi"(%get_64, %dict_66) <{predicate = 1}> : (i32, i32) -> i1
  %if_68 = "scf.if"(%cmpi_67) ({
    %fieldnode_71 = sdql.access_record %ps "ps_supplycost" : record<"ps_partkey": i32, "ps_suppkey": i32, "ps_availqty": f64, "ps_supplycost": f64, "ps_comment": memref<199xi8>> -> f64
    %fieldnode_72 = sdql.access_record %ps "ps_availqty" : record<"ps_partkey": i32, "ps_suppkey": i32, "ps_availqty": f64, "ps_supplycost": f64, "ps_comment": memref<199xi8>> -> f64
    %multf_70 = "arith.mulf"(%fieldnode_71, %fieldnode_72) <{fastmath = #arith.fastmath<none>}> : (f64, f64) -> f64
    %constd_73 = "arith.constant"() <{value = 1.0E-4  : f64}> : () -> f64
    %multf_69 = "arith.mulf"(%multf_70, %constd_73) <{fastmath = #arith.fastmath<none>}> : (f64, f64) -> f64
    %fieldnode_75 = sdql.access_record %ps "ps_partkey" : record<"ps_partkey": i32, "ps_suppkey": i32, "ps_availqty": f64, "ps_supplycost": f64, "ps_comment": memref<199xi8>> -> i32
    %fieldnode_77 = sdql.access_record %ps "ps_supplycost" : record<"ps_partkey": i32, "ps_suppkey": i32, "ps_availqty": f64, "ps_supplycost": f64, "ps_comment": memref<199xi8>> -> f64
    %fieldnode_78 = sdql.access_record %ps "ps_availqty" : record<"ps_partkey": i32, "ps_suppkey": i32, "ps_availqty": f64, "ps_supplycost": f64, "ps_comment": memref<199xi8>> -> f64
    %multf_76 = "arith.mulf"(%fieldnode_77, %fieldnode_78) <{fastmath = #arith.fastmath<none>}> : (f64, f64) -> f64
    %dict_74 = sdql.create_dictionary %fieldnode_75, %multf_76 : i32, f64 -> dictionary<i32, f64>
    %recnode_79 = sdql.create_record {fields = ["_", "_"]} %multf_69, %dict_74 : f64, dictionary<i32, f64> -> record<"_": f64, "_": dictionary<i32, f64>>
    "scf.yield"(%recnode_79) : (record<"_" : f64, "_" : dictionary<i32, f64>>) -> ()
  }, {
    %dict_80 = sdql.empty_dictionary : record<"_" : f64, "_" : dictionary<i32, f64>>
    "scf.yield"(%dict_80) : (record<"_" : f64, "_" : dictionary<i32, f64>>) -> ()
  }) : (i1) -> record<"_" : f64, "_" : dictionary<i32, f64>>
  sdql.yield %if_68 : record<"_" : f64, "_" : dictionary<i32, f64>>
}
%consti_82 = "arith.constant"() <{value = 1 : i32}> : () -> i32
%get_81 = sdql.access_record %ps_t "%consti_82" : record<"_" : f64, "_" : dictionary<i32, f64>> -> dictionary<i32, f64>
%sum_83 = sdql.sum %get_81 : dictionary<i32, f64> -> dictionary<record<"_" : i32, "_" : f64>, i32> {
^bb_84(%ps_partkey: i32, %ps_supplycost: f64):
  %consti_86 = "arith.constant"() <{value = 0 : i32}> : () -> i32
  %get_85 = sdql.access_record %ps_t "%consti_86" : record<"_" : f64, "_" : dictionary<i32, f64>> -> f64
  %cmpf_87 = "arith.cmpf"(%get_85, %ps_supplycost) <{fastmath = #arith.fastmath<none>, predicate = 4}> : (f64, f64) -> i1
  %if_88 = "scf.if"(%cmpf_87) ({
    %recnode_90 = sdql.create_record {fields = ["_", "_"]} %ps_partkey, %ps_supplycost : i32, f64 -> record<"_": i32, "_": f64>
    %consti_91 = "arith.constant"() <{value = 1 : i32}> : () -> i32
    %dict_89 = sdql.create_dictionary %recnode_90, %consti_91 : record<"_" : i32, "_" : f64>, i32 -> dictionary<record<"_" : i32, "_" : f64>, i32>
    "scf.yield"(%dict_89) : (dictionary<record<"_" : i32, "_" : f64>, i32>) -> ()
  }, {
    %dict_92 = sdql.empty_dictionary : dictionary<record<"_" : i32, "_" : f64>, i32>
    "scf.yield"(%dict_92) : (dictionary<record<"_" : i32, "_" : f64>, i32>) -> ()
  }) : (i1) -> dictionary<record<"_" : i32, "_" : f64>, i32>
  sdql.yield %if_88 : dictionary<record<"_" : i32, "_" : f64>, i32>
}
