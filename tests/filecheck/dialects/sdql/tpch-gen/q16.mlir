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
%partsupp = sdql.load "datasets/tpch/partsupp.tbl" : record<"ps_partkey" : dictionary<i32, i32>, "ps_suppkey" : dictionary<i32, i32>, "ps_availqty" : dictionary<i32, f64>, "ps_supplycost" : dictionary<i32, f64>, "ps_comment" : dictionary<i32, memref<199xi8>>, "size" : i32>
%part = sdql.load "datasets/tpch/part.tbl" : record<"p_partkey" : dictionary<i32, i32>, "p_name" : dictionary<i32, memref<55xi8>>, "p_mfgr" : dictionary<i32, memref<25xi8>>, "p_brand" : dictionary<i32, memref<10xi8>>, "p_type" : dictionary<i32, memref<25xi8>>, "p_size" : dictionary<i32, i32>, "p_container" : dictionary<i32, memref<10xi8>>, "p_retailprice" : dictionary<i32, f64>, "p_comment" : dictionary<i32, memref<23xi8>>, "size" : i32>
%supplier = sdql.load "datasets/tpch/supplier.tbl" : record<"s_suppkey" : dictionary<i32, i32>, "s_name" : dictionary<i32, memref<25xi8>>, "s_address" : dictionary<i32, memref<40xi8>>, "s_nationkey" : dictionary<i32, i32>, "s_phone" : dictionary<i32, memref<15xi8>>, "s_acctbal" : dictionary<i32, f64>, "s_comment" : dictionary<i32, memref<101xi8>>, "size" : i32>
// Brand#45
%constds_0 = "arith.constant"() <{value = dense<[66, 114, 97, 110, 100, 35, 52, 53]> : memref<?xi8>}> : () -> memref<?xi8>
%consti_1 = "arith.constant"() <{value = 9 : i32}> : () -> i32
%brand = sdql.external "ConstantString", %constds_0, %consti_1 : memref<?xi8>, i32 -> memref<8xi8>
// MEDIUM POLISHED
%constds_2 = "arith.constant"() <{value = dense<[77, 69, 68, 73, 85, 77, 32, 80, 79, 76, 73, 83, 72, 69, 68]> : memref<?xi8>}> : () -> memref<?xi8>
%consti_3 = "arith.constant"() <{value = 16 : i32}> : () -> i32
%medpol = sdql.external "ConstantString", %constds_2, %consti_3 : memref<?xi8>, i32 -> memref<15xi8>
// Customer
%constds_4 = "arith.constant"() <{value = dense<[67, 117, 115, 116, 111, 109, 101, 114]> : memref<?xi8>}> : () -> memref<?xi8>
%consti_5 = "arith.constant"() <{value = 9 : i32}> : () -> i32
%_customer = sdql.external "ConstantString", %constds_4, %consti_5 : memref<?xi8>, i32 -> memref<8xi8>
// Complaints
%constds_6 = "arith.constant"() <{value = dense<[67, 111, 109, 112, 108, 97, 105, 110, 116, 115]> : memref<?xi8>}> : () -> memref<?xi8>
%consti_7 = "arith.constant"() <{value = 11 : i32}> : () -> i32
%_complaints = sdql.external "ConstantString", %constds_6, %consti_7 : memref<?xi8>, i32 -> memref<10xi8>
%nchars_customer = "arith.constant"() <{value = 8 : i32}> : () -> i32
%fieldnode_8 = sdql.access_record %part "size" : record<"p_partkey": dictionary<i32, i32>, "p_name": dictionary<i32, memref<55xi8>>, "p_mfgr": dictionary<i32, memref<25xi8>>, "p_brand": dictionary<i32, memref<10xi8>>, "p_type": dictionary<i32, memref<25xi8>>, "p_size": dictionary<i32, i32>, "p_container": dictionary<i32, memref<10xi8>>, "p_retailprice": dictionary<i32, f64>, "p_comment": dictionary<i32, memref<23xi8>>, "size": i32> -> i32
%rangenode_9 = "func.call"(%fieldnode_8) <{callee = @range_builtin}> : (i32) -> dictionary<i32, i32>
%p_h = sdql.sum %rangenode_9 : dictionary<i32, i32> -> dictionary<i32, record<"_" : memref<10xi8>, "_" : memref<25xi8>, "_" : i32>> {
^bb_10(%i: i32, %_: i32):
  %fieldnode_12 = sdql.access_record %part "p_partkey" : record<"p_partkey": dictionary<i32, i32>, "p_name": dictionary<i32, memref<55xi8>>, "p_mfgr": dictionary<i32, memref<25xi8>>, "p_brand": dictionary<i32, memref<10xi8>>, "p_type": dictionary<i32, memref<25xi8>>, "p_size": dictionary<i32, i32>, "p_container": dictionary<i32, memref<10xi8>>, "p_retailprice": dictionary<i32, f64>, "p_comment": dictionary<i32, memref<23xi8>>, "size": i32> -> dictionary<i32, i32>
  %get_11 = sdql.lookup_dictionary %fieldnode_12 [%i : i32] : dictionary<i32, i32> -> i32
  %fieldnode_14 = sdql.access_record %part "p_name" : record<"p_partkey": dictionary<i32, i32>, "p_name": dictionary<i32, memref<55xi8>>, "p_mfgr": dictionary<i32, memref<25xi8>>, "p_brand": dictionary<i32, memref<10xi8>>, "p_type": dictionary<i32, memref<25xi8>>, "p_size": dictionary<i32, i32>, "p_container": dictionary<i32, memref<10xi8>>, "p_retailprice": dictionary<i32, f64>, "p_comment": dictionary<i32, memref<23xi8>>, "size": i32> -> dictionary<i32, memref<55xi8>>
  %get_13 = sdql.lookup_dictionary %fieldnode_14 [%i : i32] : dictionary<i32, memref<55xi8>> -> memref<55xi8>
  %fieldnode_16 = sdql.access_record %part "p_mfgr" : record<"p_partkey": dictionary<i32, i32>, "p_name": dictionary<i32, memref<55xi8>>, "p_mfgr": dictionary<i32, memref<25xi8>>, "p_brand": dictionary<i32, memref<10xi8>>, "p_type": dictionary<i32, memref<25xi8>>, "p_size": dictionary<i32, i32>, "p_container": dictionary<i32, memref<10xi8>>, "p_retailprice": dictionary<i32, f64>, "p_comment": dictionary<i32, memref<23xi8>>, "size": i32> -> dictionary<i32, memref<25xi8>>
  %get_15 = sdql.lookup_dictionary %fieldnode_16 [%i : i32] : dictionary<i32, memref<25xi8>> -> memref<25xi8>
  %fieldnode_18 = sdql.access_record %part "p_brand" : record<"p_partkey": dictionary<i32, i32>, "p_name": dictionary<i32, memref<55xi8>>, "p_mfgr": dictionary<i32, memref<25xi8>>, "p_brand": dictionary<i32, memref<10xi8>>, "p_type": dictionary<i32, memref<25xi8>>, "p_size": dictionary<i32, i32>, "p_container": dictionary<i32, memref<10xi8>>, "p_retailprice": dictionary<i32, f64>, "p_comment": dictionary<i32, memref<23xi8>>, "size": i32> -> dictionary<i32, memref<10xi8>>
  %get_17 = sdql.lookup_dictionary %fieldnode_18 [%i : i32] : dictionary<i32, memref<10xi8>> -> memref<10xi8>
  %fieldnode_20 = sdql.access_record %part "p_type" : record<"p_partkey": dictionary<i32, i32>, "p_name": dictionary<i32, memref<55xi8>>, "p_mfgr": dictionary<i32, memref<25xi8>>, "p_brand": dictionary<i32, memref<10xi8>>, "p_type": dictionary<i32, memref<25xi8>>, "p_size": dictionary<i32, i32>, "p_container": dictionary<i32, memref<10xi8>>, "p_retailprice": dictionary<i32, f64>, "p_comment": dictionary<i32, memref<23xi8>>, "size": i32> -> dictionary<i32, memref<25xi8>>
  %get_19 = sdql.lookup_dictionary %fieldnode_20 [%i : i32] : dictionary<i32, memref<25xi8>> -> memref<25xi8>
  %fieldnode_22 = sdql.access_record %part "p_size" : record<"p_partkey": dictionary<i32, i32>, "p_name": dictionary<i32, memref<55xi8>>, "p_mfgr": dictionary<i32, memref<25xi8>>, "p_brand": dictionary<i32, memref<10xi8>>, "p_type": dictionary<i32, memref<25xi8>>, "p_size": dictionary<i32, i32>, "p_container": dictionary<i32, memref<10xi8>>, "p_retailprice": dictionary<i32, f64>, "p_comment": dictionary<i32, memref<23xi8>>, "size": i32> -> dictionary<i32, i32>
  %get_21 = sdql.lookup_dictionary %fieldnode_22 [%i : i32] : dictionary<i32, i32> -> i32
  %fieldnode_24 = sdql.access_record %part "p_container" : record<"p_partkey": dictionary<i32, i32>, "p_name": dictionary<i32, memref<55xi8>>, "p_mfgr": dictionary<i32, memref<25xi8>>, "p_brand": dictionary<i32, memref<10xi8>>, "p_type": dictionary<i32, memref<25xi8>>, "p_size": dictionary<i32, i32>, "p_container": dictionary<i32, memref<10xi8>>, "p_retailprice": dictionary<i32, f64>, "p_comment": dictionary<i32, memref<23xi8>>, "size": i32> -> dictionary<i32, memref<10xi8>>
  %get_23 = sdql.lookup_dictionary %fieldnode_24 [%i : i32] : dictionary<i32, memref<10xi8>> -> memref<10xi8>
  %fieldnode_26 = sdql.access_record %part "p_retailprice" : record<"p_partkey": dictionary<i32, i32>, "p_name": dictionary<i32, memref<55xi8>>, "p_mfgr": dictionary<i32, memref<25xi8>>, "p_brand": dictionary<i32, memref<10xi8>>, "p_type": dictionary<i32, memref<25xi8>>, "p_size": dictionary<i32, i32>, "p_container": dictionary<i32, memref<10xi8>>, "p_retailprice": dictionary<i32, f64>, "p_comment": dictionary<i32, memref<23xi8>>, "size": i32> -> dictionary<i32, f64>
  %get_25 = sdql.lookup_dictionary %fieldnode_26 [%i : i32] : dictionary<i32, f64> -> f64
  %fieldnode_28 = sdql.access_record %part "p_comment" : record<"p_partkey": dictionary<i32, i32>, "p_name": dictionary<i32, memref<55xi8>>, "p_mfgr": dictionary<i32, memref<25xi8>>, "p_brand": dictionary<i32, memref<10xi8>>, "p_type": dictionary<i32, memref<25xi8>>, "p_size": dictionary<i32, i32>, "p_container": dictionary<i32, memref<10xi8>>, "p_retailprice": dictionary<i32, f64>, "p_comment": dictionary<i32, memref<23xi8>>, "size": i32> -> dictionary<i32, memref<23xi8>>
  %get_27 = sdql.lookup_dictionary %fieldnode_28 [%i : i32] : dictionary<i32, memref<23xi8>> -> memref<23xi8>
  %p = sdql.create_record {fields = ["p_partkey", "p_name", "p_mfgr", "p_brand", "p_type", "p_size", "p_container", "p_retailprice", "p_comment"]} %get_11, %get_13, %get_15, %get_17, %get_19, %get_21, %get_23, %get_25, %get_27 : i32, memref<55xi8>, memref<25xi8>, memref<10xi8>, memref<25xi8>, i32, memref<10xi8>, f64, memref<23xi8> -> record<"p_partkey": i32, "p_name": memref<55xi8>, "p_mfgr": memref<25xi8>, "p_brand": memref<10xi8>, "p_type": memref<25xi8>, "p_size": i32, "p_container": memref<10xi8>, "p_retailprice": f64, "p_comment": memref<23xi8>>
  %fieldnode_31 = sdql.access_record %p "p_brand" : record<"p_partkey": i32, "p_name": memref<55xi8>, "p_mfgr": memref<25xi8>, "p_brand": memref<10xi8>, "p_type": memref<25xi8>, "p_size": i32, "p_container": memref<10xi8>, "p_retailprice": f64, "p_comment": memref<23xi8>> -> memref<10xi8>
  %cmp_32 = sdql.cmp %fieldnode_31, %brand : memref<10xi8>, memref<8xi8> -> i1
  %zero_29 = "arith.constant"() <{value = 0 : i1}> : () -> i1
  %neg_30 = "arith.cmpi"(%cmp_32, %zero_29) <{predicate = 0}> : (i1, i1) -> i1
  %if_33 = "scf.if"(%neg_30) ({
    %fieldnode_34 = sdql.access_record %p "p_type" : record<"p_partkey": i32, "p_name": memref<55xi8>, "p_mfgr": memref<25xi8>, "p_brand": memref<10xi8>, "p_type": memref<25xi8>, "p_size": i32, "p_container": memref<10xi8>, "p_retailprice": f64, "p_comment": memref<23xi8>> -> memref<25xi8>
    %external_35 = sdql.external "StrStartsWith", %fieldnode_34, %medpol : memref<25xi8>, memref<15xi8> -> i1
    %if_36 = "scf.if"(%external_35) ({
      %constv_37 = "arith.constant"() <{value = 0 : i1}> : () -> i1
      "scf.yield"(%constv_37) : (i1) -> ()
    }, {
      %constv_38 = "arith.constant"() <{value = 1 : i1}> : () -> i1
      "scf.yield"(%constv_38) : (i1) -> ()
    }) : (i1) -> i1
    "scf.yield"(%if_36) : (i1) -> ()
  }, {
    %constv_39 = "arith.constant"() <{value = 0 : i1}> : () -> i1
    "scf.yield"(%constv_39) : (i1) -> ()
  }) : (i1) -> i1
  %if_40 = "scf.if"(%if_33) ({
    %fieldnode_41 = sdql.access_record %p "p_size" : record<"p_partkey": i32, "p_name": memref<55xi8>, "p_mfgr": memref<25xi8>, "p_brand": memref<10xi8>, "p_type": memref<25xi8>, "p_size": i32, "p_container": memref<10xi8>, "p_retailprice": f64, "p_comment": memref<23xi8>> -> i32
    %consti_42 = "arith.constant"() <{value = 49 : i32}> : () -> i32
    %cmpi_43 = "arith.cmpi"(%fieldnode_41, %consti_42) <{predicate = 0}> : (i32, i32) -> i1
    %if_44 = "scf.if"(%cmpi_43) ({
      %constv_45 = "arith.constant"() <{value = 1 : i1}> : () -> i1
      "scf.yield"(%constv_45) : (i1) -> ()
    }, {
      %fieldnode_46 = sdql.access_record %p "p_size" : record<"p_partkey": i32, "p_name": memref<55xi8>, "p_mfgr": memref<25xi8>, "p_brand": memref<10xi8>, "p_type": memref<25xi8>, "p_size": i32, "p_container": memref<10xi8>, "p_retailprice": f64, "p_comment": memref<23xi8>> -> i32
      %consti_47 = "arith.constant"() <{value = 14 : i32}> : () -> i32
      %cmpi_48 = "arith.cmpi"(%fieldnode_46, %consti_47) <{predicate = 0}> : (i32, i32) -> i1
      "scf.yield"(%cmpi_48) : (i1) -> ()
    }) : (i1) -> i1
    %if_49 = "scf.if"(%if_44) ({
      %constv_50 = "arith.constant"() <{value = 1 : i1}> : () -> i1
      "scf.yield"(%constv_50) : (i1) -> ()
    }, {
      %fieldnode_51 = sdql.access_record %p "p_size" : record<"p_partkey": i32, "p_name": memref<55xi8>, "p_mfgr": memref<25xi8>, "p_brand": memref<10xi8>, "p_type": memref<25xi8>, "p_size": i32, "p_container": memref<10xi8>, "p_retailprice": f64, "p_comment": memref<23xi8>> -> i32
      %consti_52 = "arith.constant"() <{value = 23 : i32}> : () -> i32
      %cmpi_53 = "arith.cmpi"(%fieldnode_51, %consti_52) <{predicate = 0}> : (i32, i32) -> i1
      "scf.yield"(%cmpi_53) : (i1) -> ()
    }) : (i1) -> i1
    %if_54 = "scf.if"(%if_49) ({
      %constv_55 = "arith.constant"() <{value = 1 : i1}> : () -> i1
      "scf.yield"(%constv_55) : (i1) -> ()
    }, {
      %fieldnode_56 = sdql.access_record %p "p_size" : record<"p_partkey": i32, "p_name": memref<55xi8>, "p_mfgr": memref<25xi8>, "p_brand": memref<10xi8>, "p_type": memref<25xi8>, "p_size": i32, "p_container": memref<10xi8>, "p_retailprice": f64, "p_comment": memref<23xi8>> -> i32
      %consti_57 = "arith.constant"() <{value = 45 : i32}> : () -> i32
      %cmpi_58 = "arith.cmpi"(%fieldnode_56, %consti_57) <{predicate = 0}> : (i32, i32) -> i1
      "scf.yield"(%cmpi_58) : (i1) -> ()
    }) : (i1) -> i1
    %if_59 = "scf.if"(%if_54) ({
      %constv_60 = "arith.constant"() <{value = 1 : i1}> : () -> i1
      "scf.yield"(%constv_60) : (i1) -> ()
    }, {
      %fieldnode_61 = sdql.access_record %p "p_size" : record<"p_partkey": i32, "p_name": memref<55xi8>, "p_mfgr": memref<25xi8>, "p_brand": memref<10xi8>, "p_type": memref<25xi8>, "p_size": i32, "p_container": memref<10xi8>, "p_retailprice": f64, "p_comment": memref<23xi8>> -> i32
      %consti_62 = "arith.constant"() <{value = 19 : i32}> : () -> i32
      %cmpi_63 = "arith.cmpi"(%fieldnode_61, %consti_62) <{predicate = 0}> : (i32, i32) -> i1
      "scf.yield"(%cmpi_63) : (i1) -> ()
    }) : (i1) -> i1
    %if_64 = "scf.if"(%if_59) ({
      %constv_65 = "arith.constant"() <{value = 1 : i1}> : () -> i1
      "scf.yield"(%constv_65) : (i1) -> ()
    }, {
      %fieldnode_66 = sdql.access_record %p "p_size" : record<"p_partkey": i32, "p_name": memref<55xi8>, "p_mfgr": memref<25xi8>, "p_brand": memref<10xi8>, "p_type": memref<25xi8>, "p_size": i32, "p_container": memref<10xi8>, "p_retailprice": f64, "p_comment": memref<23xi8>> -> i32
      %consti_67 = "arith.constant"() <{value = 3 : i32}> : () -> i32
      %cmpi_68 = "arith.cmpi"(%fieldnode_66, %consti_67) <{predicate = 0}> : (i32, i32) -> i1
      "scf.yield"(%cmpi_68) : (i1) -> ()
    }) : (i1) -> i1
    %if_69 = "scf.if"(%if_64) ({
      %constv_70 = "arith.constant"() <{value = 1 : i1}> : () -> i1
      "scf.yield"(%constv_70) : (i1) -> ()
    }, {
      %fieldnode_71 = sdql.access_record %p "p_size" : record<"p_partkey": i32, "p_name": memref<55xi8>, "p_mfgr": memref<25xi8>, "p_brand": memref<10xi8>, "p_type": memref<25xi8>, "p_size": i32, "p_container": memref<10xi8>, "p_retailprice": f64, "p_comment": memref<23xi8>> -> i32
      %consti_72 = "arith.constant"() <{value = 36 : i32}> : () -> i32
      %cmpi_73 = "arith.cmpi"(%fieldnode_71, %consti_72) <{predicate = 0}> : (i32, i32) -> i1
      "scf.yield"(%cmpi_73) : (i1) -> ()
    }) : (i1) -> i1
    %if_74 = "scf.if"(%if_69) ({
      %constv_75 = "arith.constant"() <{value = 1 : i1}> : () -> i1
      "scf.yield"(%constv_75) : (i1) -> ()
    }, {
      %fieldnode_76 = sdql.access_record %p "p_size" : record<"p_partkey": i32, "p_name": memref<55xi8>, "p_mfgr": memref<25xi8>, "p_brand": memref<10xi8>, "p_type": memref<25xi8>, "p_size": i32, "p_container": memref<10xi8>, "p_retailprice": f64, "p_comment": memref<23xi8>> -> i32
      %consti_77 = "arith.constant"() <{value = 9 : i32}> : () -> i32
      %cmpi_78 = "arith.cmpi"(%fieldnode_76, %consti_77) <{predicate = 0}> : (i32, i32) -> i1
      "scf.yield"(%cmpi_78) : (i1) -> ()
    }) : (i1) -> i1
    "scf.yield"(%if_74) : (i1) -> ()
  }, {
    %constv_79 = "arith.constant"() <{value = 0 : i1}> : () -> i1
    "scf.yield"(%constv_79) : (i1) -> ()
  }) : (i1) -> i1
  %if_80 = "scf.if"(%if_40) ({
    %fieldnode_83 = sdql.access_record %p "p_partkey" : record<"p_partkey": i32, "p_name": memref<55xi8>, "p_mfgr": memref<25xi8>, "p_brand": memref<10xi8>, "p_type": memref<25xi8>, "p_size": i32, "p_container": memref<10xi8>, "p_retailprice": f64, "p_comment": memref<23xi8>> -> i32
    %uniq_82 = sdql.unique %fieldnode_83 : i32 -> i32
    %fieldnode_84 = sdql.access_record %p "p_brand" : record<"p_partkey": i32, "p_name": memref<55xi8>, "p_mfgr": memref<25xi8>, "p_brand": memref<10xi8>, "p_type": memref<25xi8>, "p_size": i32, "p_container": memref<10xi8>, "p_retailprice": f64, "p_comment": memref<23xi8>> -> memref<10xi8>
    %fieldnode_85 = sdql.access_record %p "p_type" : record<"p_partkey": i32, "p_name": memref<55xi8>, "p_mfgr": memref<25xi8>, "p_brand": memref<10xi8>, "p_type": memref<25xi8>, "p_size": i32, "p_container": memref<10xi8>, "p_retailprice": f64, "p_comment": memref<23xi8>> -> memref<25xi8>
    %fieldnode_86 = sdql.access_record %p "p_size" : record<"p_partkey": i32, "p_name": memref<55xi8>, "p_mfgr": memref<25xi8>, "p_brand": memref<10xi8>, "p_type": memref<25xi8>, "p_size": i32, "p_container": memref<10xi8>, "p_retailprice": f64, "p_comment": memref<23xi8>> -> i32
    %recnode_87 = sdql.create_record {fields = ["_", "_", "_"]} %fieldnode_84, %fieldnode_85, %fieldnode_86 : memref<10xi8>, memref<25xi8>, i32 -> record<"_": memref<10xi8>, "_": memref<25xi8>, "_": i32>
    %dict_81 = sdql.create_dictionary %uniq_82, %recnode_87 : i32, record<"_" : memref<10xi8>, "_" : memref<25xi8>, "_" : i32> -> dictionary<i32, record<"_" : memref<10xi8>, "_" : memref<25xi8>, "_" : i32>>
    "scf.yield"(%dict_81) : (dictionary<i32, record<"_" : memref<10xi8>, "_" : memref<25xi8>, "_" : i32>>) -> ()
  }, {
    %dict_88 = sdql.empty_dictionary : dictionary<i32, record<"_" : memref<10xi8>, "_" : memref<25xi8>, "_" : i32>>
    "scf.yield"(%dict_88) : (dictionary<i32, record<"_" : memref<10xi8>, "_" : memref<25xi8>, "_" : i32>>) -> ()
  }) : (i1) -> dictionary<i32, record<"_" : memref<10xi8>, "_" : memref<25xi8>, "_" : i32>>
  sdql.yield %if_80 : dictionary<i32, record<"_" : memref<10xi8>, "_" : memref<25xi8>, "_" : i32>>
}
%fieldnode_89 = sdql.access_record %supplier "size" : record<"s_suppkey": dictionary<i32, i32>, "s_name": dictionary<i32, memref<25xi8>>, "s_address": dictionary<i32, memref<40xi8>>, "s_nationkey": dictionary<i32, i32>, "s_phone": dictionary<i32, memref<15xi8>>, "s_acctbal": dictionary<i32, f64>, "s_comment": dictionary<i32, memref<101xi8>>, "size": i32> -> i32
%rangenode_90 = "func.call"(%fieldnode_89) <{callee = @range_builtin}> : (i32) -> dictionary<i32, i32>
%s_h = sdql.sum %rangenode_90 : dictionary<i32, i32> -> dictionary<i32, record<"_" : i32>> {
^bb_91(%i: i32, %_: i32):
  %fieldnode_93 = sdql.access_record %supplier "s_suppkey" : record<"s_suppkey": dictionary<i32, i32>, "s_name": dictionary<i32, memref<25xi8>>, "s_address": dictionary<i32, memref<40xi8>>, "s_nationkey": dictionary<i32, i32>, "s_phone": dictionary<i32, memref<15xi8>>, "s_acctbal": dictionary<i32, f64>, "s_comment": dictionary<i32, memref<101xi8>>, "size": i32> -> dictionary<i32, i32>
  %get_92 = sdql.lookup_dictionary %fieldnode_93 [%i : i32] : dictionary<i32, i32> -> i32
  %fieldnode_95 = sdql.access_record %supplier "s_name" : record<"s_suppkey": dictionary<i32, i32>, "s_name": dictionary<i32, memref<25xi8>>, "s_address": dictionary<i32, memref<40xi8>>, "s_nationkey": dictionary<i32, i32>, "s_phone": dictionary<i32, memref<15xi8>>, "s_acctbal": dictionary<i32, f64>, "s_comment": dictionary<i32, memref<101xi8>>, "size": i32> -> dictionary<i32, memref<25xi8>>
  %get_94 = sdql.lookup_dictionary %fieldnode_95 [%i : i32] : dictionary<i32, memref<25xi8>> -> memref<25xi8>
  %fieldnode_97 = sdql.access_record %supplier "s_address" : record<"s_suppkey": dictionary<i32, i32>, "s_name": dictionary<i32, memref<25xi8>>, "s_address": dictionary<i32, memref<40xi8>>, "s_nationkey": dictionary<i32, i32>, "s_phone": dictionary<i32, memref<15xi8>>, "s_acctbal": dictionary<i32, f64>, "s_comment": dictionary<i32, memref<101xi8>>, "size": i32> -> dictionary<i32, memref<40xi8>>
  %get_96 = sdql.lookup_dictionary %fieldnode_97 [%i : i32] : dictionary<i32, memref<40xi8>> -> memref<40xi8>
  %fieldnode_99 = sdql.access_record %supplier "s_nationkey" : record<"s_suppkey": dictionary<i32, i32>, "s_name": dictionary<i32, memref<25xi8>>, "s_address": dictionary<i32, memref<40xi8>>, "s_nationkey": dictionary<i32, i32>, "s_phone": dictionary<i32, memref<15xi8>>, "s_acctbal": dictionary<i32, f64>, "s_comment": dictionary<i32, memref<101xi8>>, "size": i32> -> dictionary<i32, i32>
  %get_98 = sdql.lookup_dictionary %fieldnode_99 [%i : i32] : dictionary<i32, i32> -> i32
  %fieldnode_101 = sdql.access_record %supplier "s_phone" : record<"s_suppkey": dictionary<i32, i32>, "s_name": dictionary<i32, memref<25xi8>>, "s_address": dictionary<i32, memref<40xi8>>, "s_nationkey": dictionary<i32, i32>, "s_phone": dictionary<i32, memref<15xi8>>, "s_acctbal": dictionary<i32, f64>, "s_comment": dictionary<i32, memref<101xi8>>, "size": i32> -> dictionary<i32, memref<15xi8>>
  %get_100 = sdql.lookup_dictionary %fieldnode_101 [%i : i32] : dictionary<i32, memref<15xi8>> -> memref<15xi8>
  %fieldnode_103 = sdql.access_record %supplier "s_acctbal" : record<"s_suppkey": dictionary<i32, i32>, "s_name": dictionary<i32, memref<25xi8>>, "s_address": dictionary<i32, memref<40xi8>>, "s_nationkey": dictionary<i32, i32>, "s_phone": dictionary<i32, memref<15xi8>>, "s_acctbal": dictionary<i32, f64>, "s_comment": dictionary<i32, memref<101xi8>>, "size": i32> -> dictionary<i32, f64>
  %get_102 = sdql.lookup_dictionary %fieldnode_103 [%i : i32] : dictionary<i32, f64> -> f64
  %fieldnode_105 = sdql.access_record %supplier "s_comment" : record<"s_suppkey": dictionary<i32, i32>, "s_name": dictionary<i32, memref<25xi8>>, "s_address": dictionary<i32, memref<40xi8>>, "s_nationkey": dictionary<i32, i32>, "s_phone": dictionary<i32, memref<15xi8>>, "s_acctbal": dictionary<i32, f64>, "s_comment": dictionary<i32, memref<101xi8>>, "size": i32> -> dictionary<i32, memref<101xi8>>
  %get_104 = sdql.lookup_dictionary %fieldnode_105 [%i : i32] : dictionary<i32, memref<101xi8>> -> memref<101xi8>
  %s = sdql.create_record {fields = ["s_suppkey", "s_name", "s_address", "s_nationkey", "s_phone", "s_acctbal", "s_comment"]} %get_92, %get_94, %get_96, %get_98, %get_100, %get_102, %get_104 : i32, memref<25xi8>, memref<40xi8>, i32, memref<15xi8>, f64, memref<101xi8> -> record<"s_suppkey": i32, "s_name": memref<25xi8>, "s_address": memref<40xi8>, "s_nationkey": i32, "s_phone": memref<15xi8>, "s_acctbal": f64, "s_comment": memref<101xi8>>
  %fieldnode_106 = sdql.access_record %s "s_comment" : record<"s_suppkey": i32, "s_name": memref<25xi8>, "s_address": memref<40xi8>, "s_nationkey": i32, "s_phone": memref<15xi8>, "s_acctbal": f64, "s_comment": memref<101xi8>> -> memref<101xi8>
  %idx_customer = sdql.external "FirstIndex", %fieldnode_106, %_customer : memref<101xi8>, memref<8xi8> -> i32
  %consti_107 = "arith.constant"() <{value = -1 : i32}> : () -> i32
  %cmpi_108 = "arith.cmpi"(%idx_customer, %consti_107) <{predicate = 1}> : (i32, i32) -> i1
  %if_109 = "scf.if"(%cmpi_108) ({
    %addi_110 = "arith.addi"(%idx_customer, %nchars_customer) : (i32, i32) -> i32
    %fieldnode_111 = sdql.access_record %s "s_comment" : record<"s_suppkey": i32, "s_name": memref<25xi8>, "s_address": memref<40xi8>, "s_nationkey": i32, "s_phone": memref<15xi8>, "s_acctbal": f64, "s_comment": memref<101xi8>> -> memref<101xi8>
    %external_112 = sdql.external "FirstIndex", %fieldnode_111, %_complaints : memref<101xi8>, memref<10xi8> -> i32
    %cmpi_113 = "arith.cmpi"(%addi_110, %external_112) <{predicate = 3}> : (i32, i32) -> i1
    "scf.yield"(%cmpi_113) : (i1) -> ()
  }, {
    %constv_114 = "arith.constant"() <{value = 0 : i1}> : () -> i1
    "scf.yield"(%constv_114) : (i1) -> ()
  }) : (i1) -> i1
  %if_115 = "scf.if"(%if_109) ({
    %fieldnode_118 = sdql.access_record %s "s_suppkey" : record<"s_suppkey": i32, "s_name": memref<25xi8>, "s_address": memref<40xi8>, "s_nationkey": i32, "s_phone": memref<15xi8>, "s_acctbal": f64, "s_comment": memref<101xi8>> -> i32
    %uniq_117 = sdql.unique %fieldnode_118 : i32 -> i32
    %fieldnode_119 = sdql.access_record %s "s_suppkey" : record<"s_suppkey": i32, "s_name": memref<25xi8>, "s_address": memref<40xi8>, "s_nationkey": i32, "s_phone": memref<15xi8>, "s_acctbal": f64, "s_comment": memref<101xi8>> -> i32
    %recnode_120 = sdql.create_record {fields = ["_"]} %fieldnode_119 : i32 -> record<"_": i32>
    %dict_116 = sdql.create_dictionary %uniq_117, %recnode_120 : i32, record<"_" : i32> -> dictionary<i32, record<"_" : i32>>
    "scf.yield"(%dict_116) : (dictionary<i32, record<"_" : i32>>) -> ()
  }, {
    %dict_121 = sdql.empty_dictionary : dictionary<i32, record<"_" : i32>>
    "scf.yield"(%dict_121) : (dictionary<i32, record<"_" : i32>>) -> ()
  }) : (i1) -> dictionary<i32, record<"_" : i32>>
  sdql.yield %if_115 : dictionary<i32, record<"_" : i32>>
}
%fieldnode_122 = sdql.access_record %partsupp "size" : record<"ps_partkey": dictionary<i32, i32>, "ps_suppkey": dictionary<i32, i32>, "ps_availqty": dictionary<i32, f64>, "ps_supplycost": dictionary<i32, f64>, "ps_comment": dictionary<i32, memref<199xi8>>, "size": i32> -> i32
%rangenode_123 = "func.call"(%fieldnode_122) <{callee = @range_builtin}> : (i32) -> dictionary<i32, i32>
%ps_h = sdql.sum %rangenode_123 : dictionary<i32, i32> -> dictionary<record<"brand" : memref<10xi8>, "type" : memref<25xi8>, "size" : i32>, dictionary<i32, i32>> {
^bb_124(%i: i32, %_: i32):
  %fieldnode_126 = sdql.access_record %partsupp "ps_partkey" : record<"ps_partkey": dictionary<i32, i32>, "ps_suppkey": dictionary<i32, i32>, "ps_availqty": dictionary<i32, f64>, "ps_supplycost": dictionary<i32, f64>, "ps_comment": dictionary<i32, memref<199xi8>>, "size": i32> -> dictionary<i32, i32>
  %get_125 = sdql.lookup_dictionary %fieldnode_126 [%i : i32] : dictionary<i32, i32> -> i32
  %fieldnode_128 = sdql.access_record %partsupp "ps_suppkey" : record<"ps_partkey": dictionary<i32, i32>, "ps_suppkey": dictionary<i32, i32>, "ps_availqty": dictionary<i32, f64>, "ps_supplycost": dictionary<i32, f64>, "ps_comment": dictionary<i32, memref<199xi8>>, "size": i32> -> dictionary<i32, i32>
  %get_127 = sdql.lookup_dictionary %fieldnode_128 [%i : i32] : dictionary<i32, i32> -> i32
  %fieldnode_130 = sdql.access_record %partsupp "ps_availqty" : record<"ps_partkey": dictionary<i32, i32>, "ps_suppkey": dictionary<i32, i32>, "ps_availqty": dictionary<i32, f64>, "ps_supplycost": dictionary<i32, f64>, "ps_comment": dictionary<i32, memref<199xi8>>, "size": i32> -> dictionary<i32, f64>
  %get_129 = sdql.lookup_dictionary %fieldnode_130 [%i : i32] : dictionary<i32, f64> -> f64
  %fieldnode_132 = sdql.access_record %partsupp "ps_supplycost" : record<"ps_partkey": dictionary<i32, i32>, "ps_suppkey": dictionary<i32, i32>, "ps_availqty": dictionary<i32, f64>, "ps_supplycost": dictionary<i32, f64>, "ps_comment": dictionary<i32, memref<199xi8>>, "size": i32> -> dictionary<i32, f64>
  %get_131 = sdql.lookup_dictionary %fieldnode_132 [%i : i32] : dictionary<i32, f64> -> f64
  %fieldnode_134 = sdql.access_record %partsupp "ps_comment" : record<"ps_partkey": dictionary<i32, i32>, "ps_suppkey": dictionary<i32, i32>, "ps_availqty": dictionary<i32, f64>, "ps_supplycost": dictionary<i32, f64>, "ps_comment": dictionary<i32, memref<199xi8>>, "size": i32> -> dictionary<i32, memref<199xi8>>
  %get_133 = sdql.lookup_dictionary %fieldnode_134 [%i : i32] : dictionary<i32, memref<199xi8>> -> memref<199xi8>
  %ps = sdql.create_record {fields = ["ps_partkey", "ps_suppkey", "ps_availqty", "ps_supplycost", "ps_comment"]} %get_125, %get_127, %get_129, %get_131, %get_133 : i32, i32, f64, f64, memref<199xi8> -> record<"ps_partkey": i32, "ps_suppkey": i32, "ps_availqty": f64, "ps_supplycost": f64, "ps_comment": memref<199xi8>>
  %fieldnode_138 = sdql.access_record %ps "ps_partkey" : record<"ps_partkey": i32, "ps_suppkey": i32, "ps_availqty": f64, "ps_supplycost": f64, "ps_comment": memref<199xi8>> -> i32
  %get_137 = sdql.lookup_dictionary %p_h [%fieldnode_138 : i32] : dictionary<i32, record<"_" : memref<10xi8>, "_" : memref<25xi8>, "_" : i32>> -> record<"_" : memref<10xi8>, "_" : memref<25xi8>, "_" : i32>
  %dict_139 = sdql.empty_dictionary : record<"_" : memref<10xi8>, "_" : memref<25xi8>, "_" : i32>
  %cmp_140 = sdql.cmp %get_137, %dict_139 : record<"_" : memref<10xi8>, "_" : memref<25xi8>, "_" : i32>, record<"_" : memref<10xi8>, "_" : memref<25xi8>, "_" : i32> -> i1
  %zero_135 = "arith.constant"() <{value = 0 : i1}> : () -> i1
  %neg_136 = "arith.cmpi"(%cmp_140, %zero_135) <{predicate = 0}> : (i1, i1) -> i1
  %if_141 = "scf.if"(%neg_136) ({
    %fieldnode_145 = sdql.access_record %ps "ps_suppkey" : record<"ps_partkey": i32, "ps_suppkey": i32, "ps_availqty": f64, "ps_supplycost": f64, "ps_comment": memref<199xi8>> -> i32
    %get_144 = sdql.lookup_dictionary %s_h [%fieldnode_145 : i32] : dictionary<i32, record<"_" : i32>> -> record<"_" : i32>
    %dict_146 = sdql.empty_dictionary : record<"_" : i32>
    %cmp_147 = sdql.cmp %get_144, %dict_146 : record<"_" : i32>, record<"_" : i32> -> i1
    %zero_142 = "arith.constant"() <{value = 0 : i1}> : () -> i1
    %neg_143 = "arith.cmpi"(%cmp_147, %zero_142) <{predicate = 0}> : (i1, i1) -> i1
    %if_148 = "scf.if"(%neg_143) ({
      %constv_149 = "arith.constant"() <{value = 0 : i1}> : () -> i1
      "scf.yield"(%constv_149) : (i1) -> ()
    }, {
      %constv_150 = "arith.constant"() <{value = 1 : i1}> : () -> i1
      "scf.yield"(%constv_150) : (i1) -> ()
    }) : (i1) -> i1
    "scf.yield"(%if_148) : (i1) -> ()
  }, {
    %constv_151 = "arith.constant"() <{value = 0 : i1}> : () -> i1
    "scf.yield"(%constv_151) : (i1) -> ()
  }) : (i1) -> i1
  %if_152 = "scf.if"(%if_141) ({
    %fieldnode_156 = sdql.access_record %ps "ps_partkey" : record<"ps_partkey": i32, "ps_suppkey": i32, "ps_availqty": f64, "ps_supplycost": f64, "ps_comment": memref<199xi8>> -> i32
    %get_155 = sdql.lookup_dictionary %p_h [%fieldnode_156 : i32] : dictionary<i32, record<"_" : memref<10xi8>, "_" : memref<25xi8>, "_" : i32>> -> record<"_" : memref<10xi8>, "_" : memref<25xi8>, "_" : i32>
    %consti_157 = "arith.constant"() <{value = 0 : i32}> : () -> i32
    %get_154 = sdql.access_record %get_155 "%consti_157" : record<"_" : memref<10xi8>, "_" : memref<25xi8>, "_" : i32> -> memref<10xi8>
    %fieldnode_160 = sdql.access_record %ps "ps_partkey" : record<"ps_partkey": i32, "ps_suppkey": i32, "ps_availqty": f64, "ps_supplycost": f64, "ps_comment": memref<199xi8>> -> i32
    %get_159 = sdql.lookup_dictionary %p_h [%fieldnode_160 : i32] : dictionary<i32, record<"_" : memref<10xi8>, "_" : memref<25xi8>, "_" : i32>> -> record<"_" : memref<10xi8>, "_" : memref<25xi8>, "_" : i32>
    %consti_161 = "arith.constant"() <{value = 1 : i32}> : () -> i32
    %get_158 = sdql.access_record %get_159 "%consti_161" : record<"_" : memref<10xi8>, "_" : memref<25xi8>, "_" : i32> -> memref<25xi8>
    %fieldnode_164 = sdql.access_record %ps "ps_partkey" : record<"ps_partkey": i32, "ps_suppkey": i32, "ps_availqty": f64, "ps_supplycost": f64, "ps_comment": memref<199xi8>> -> i32
    %get_163 = sdql.lookup_dictionary %p_h [%fieldnode_164 : i32] : dictionary<i32, record<"_" : memref<10xi8>, "_" : memref<25xi8>, "_" : i32>> -> record<"_" : memref<10xi8>, "_" : memref<25xi8>, "_" : i32>
    %consti_165 = "arith.constant"() <{value = 2 : i32}> : () -> i32
    %get_162 = sdql.access_record %get_163 "%consti_165" : record<"_" : memref<10xi8>, "_" : memref<25xi8>, "_" : i32> -> i32
    %recnode_166 = sdql.create_record {fields = ["brand", "type", "size"]} %get_154, %get_158, %get_162 : memref<10xi8>, memref<25xi8>, i32 -> record<"brand": memref<10xi8>, "type": memref<25xi8>, "size": i32>
    %fieldnode_168 = sdql.access_record %ps "ps_suppkey" : record<"ps_partkey": i32, "ps_suppkey": i32, "ps_availqty": f64, "ps_supplycost": f64, "ps_comment": memref<199xi8>> -> i32
    %consti_169 = "arith.constant"() <{value = 1 : i32}> : () -> i32
    %dict_167 = sdql.create_dictionary %fieldnode_168, %consti_169 : i32, i32 -> dictionary<i32, i32>
    %dict_153 = sdql.create_dictionary %recnode_166, %dict_167 : record<"brand" : memref<10xi8>, "type" : memref<25xi8>, "size" : i32>, dictionary<i32, i32> -> dictionary<record<"brand" : memref<10xi8>, "type" : memref<25xi8>, "size" : i32>, dictionary<i32, i32>>
    "scf.yield"(%dict_153) : (dictionary<record<"brand" : memref<10xi8>, "type" : memref<25xi8>, "size" : i32>, dictionary<i32, i32>>) -> ()
  }, {
    %dict_170 = sdql.empty_dictionary : dictionary<record<"brand" : memref<10xi8>, "type" : memref<25xi8>, "size" : i32>, dictionary<i32, i32>>
    "scf.yield"(%dict_170) : (dictionary<record<"brand" : memref<10xi8>, "type" : memref<25xi8>, "size" : i32>, dictionary<i32, i32>>) -> ()
  }) : (i1) -> dictionary<record<"brand" : memref<10xi8>, "type" : memref<25xi8>, "size" : i32>, dictionary<i32, i32>>
  sdql.yield %if_152 : dictionary<record<"brand" : memref<10xi8>, "type" : memref<25xi8>, "size" : i32>, dictionary<i32, i32>>
}
%sum_171 = sdql.sum %ps_h : dictionary<record<"brand" : memref<10xi8>, "type" : memref<25xi8>, "size" : i32>, dictionary<i32, i32>> -> dictionary<record<"brand" : memref<10xi8>, "type" : memref<25xi8>, "size" : i32, "_" : i32>, i32> {
^bb_172(%k: record<"brand" : memref<10xi8>, "type" : memref<25xi8>, "size" : i32>, %v: dictionary<i32, i32>):
  %external_175 = sdql.external "Size", %v : dictionary<i32, i32> -> i32
  %recnode_176 = sdql.create_record {fields = ["_"]} %external_175 : i32 -> record<"_": i32>
  %concat_177 = sdql.concat %k, %recnode_176 : record<"brand" : memref<10xi8>, "type" : memref<25xi8>, "size" : i32>, record<"_" : i32> -> record<"brand" : memref<10xi8>, "type" : memref<25xi8>, "size" : i32, "_" : i32>
  %uniq_174 = sdql.unique %concat_177 : record<"brand" : memref<10xi8>, "type" : memref<25xi8>, "size" : i32, "_" : i32> -> record<"brand" : memref<10xi8>, "type" : memref<25xi8>, "size" : i32, "_" : i32>
  %consti_178 = "arith.constant"() <{value = 1 : i32}> : () -> i32
  %dict_173 = sdql.create_dictionary %uniq_174, %consti_178 : record<"brand" : memref<10xi8>, "type" : memref<25xi8>, "size" : i32, "_" : i32>, i32 -> dictionary<record<"brand" : memref<10xi8>, "type" : memref<25xi8>, "size" : i32, "_" : i32>, i32>
  sdql.yield %dict_173 : dictionary<record<"brand" : memref<10xi8>, "type" : memref<25xi8>, "size" : i32, "_" : i32>, i32>
}
