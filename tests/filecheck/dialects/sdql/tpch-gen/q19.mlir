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
%lineitem = sdql.load "datasets/tpch/lineitem.tbl" : record<"l_orderkey" : dictionary<i32, i32>, "l_partkey" : dictionary<i32, i32>, "l_suppkey" : dictionary<i32, i32>, "l_linenumber" : dictionary<i32, i32>, "l_quantity" : dictionary<i32, f64>, "l_extendedprice" : dictionary<i32, f64>, "l_discount" : dictionary<i32, f64>, "l_tax" : dictionary<i32, f64>, "l_returnflag" : dictionary<i32, memref<1xi8>>, "l_linestatus" : dictionary<i32, memref<1xi8>>, "l_shipdate" : dictionary<i32, i32>, "l_commitdate" : dictionary<i32, i32>, "l_receiptdate" : dictionary<i32, i32>, "l_shipinstruct" : dictionary<i32, memref<25xi8>>, "l_shipmode" : dictionary<i32, memref<10xi8>>, "l_comment" : dictionary<i32, memref<44xi8>>, "size" : i32>
%part = sdql.load "datasets/tpch/part.tbl" : record<"p_partkey" : dictionary<i32, i32>, "p_name" : dictionary<i32, memref<55xi8>>, "p_mfgr" : dictionary<i32, memref<25xi8>>, "p_brand" : dictionary<i32, memref<10xi8>>, "p_type" : dictionary<i32, memref<25xi8>>, "p_size" : dictionary<i32, i32>, "p_container" : dictionary<i32, memref<10xi8>>, "p_retailprice" : dictionary<i32, f64>, "p_comment" : dictionary<i32, memref<23xi8>>, "size" : i32>
// Brand#12
%constds_0 = "arith.constant"() <{value = dense<[66, 114, 97, 110, 100, 35, 49, 50]> : memref<?xi8>}> : () -> memref<?xi8>
%consti_1 = "arith.constant"() <{value = 9 : i32}> : () -> i32
%brand12 = sdql.external "ConstantString", %constds_0, %consti_1 : memref<?xi8>, i32 -> memref<8xi8>
// Brand#23
%constds_2 = "arith.constant"() <{value = dense<[66, 114, 97, 110, 100, 35, 50, 51]> : memref<?xi8>}> : () -> memref<?xi8>
%consti_3 = "arith.constant"() <{value = 9 : i32}> : () -> i32
%brand23 = sdql.external "ConstantString", %constds_2, %consti_3 : memref<?xi8>, i32 -> memref<8xi8>
// Brand#34
%constds_4 = "arith.constant"() <{value = dense<[66, 114, 97, 110, 100, 35, 51, 52]> : memref<?xi8>}> : () -> memref<?xi8>
%consti_5 = "arith.constant"() <{value = 9 : i32}> : () -> i32
%brand34 = sdql.external "ConstantString", %constds_4, %consti_5 : memref<?xi8>, i32 -> memref<8xi8>
// SM CASE
%constds_6 = "arith.constant"() <{value = dense<[83, 77, 32, 67, 65, 83, 69]> : memref<?xi8>}> : () -> memref<?xi8>
%consti_7 = "arith.constant"() <{value = 8 : i32}> : () -> i32
%smcase = sdql.external "ConstantString", %constds_6, %consti_7 : memref<?xi8>, i32 -> memref<7xi8>
// SM BOX
%constds_8 = "arith.constant"() <{value = dense<[83, 77, 32, 66, 79, 88]> : memref<?xi8>}> : () -> memref<?xi8>
%consti_9 = "arith.constant"() <{value = 7 : i32}> : () -> i32
%smbox = sdql.external "ConstantString", %constds_8, %consti_9 : memref<?xi8>, i32 -> memref<6xi8>
// SM PACK
%constds_10 = "arith.constant"() <{value = dense<[83, 77, 32, 80, 65, 67, 75]> : memref<?xi8>}> : () -> memref<?xi8>
%consti_11 = "arith.constant"() <{value = 8 : i32}> : () -> i32
%smpack = sdql.external "ConstantString", %constds_10, %consti_11 : memref<?xi8>, i32 -> memref<7xi8>
// SM PKG
%constds_12 = "arith.constant"() <{value = dense<[83, 77, 32, 80, 75, 71]> : memref<?xi8>}> : () -> memref<?xi8>
%consti_13 = "arith.constant"() <{value = 7 : i32}> : () -> i32
%smpkg = sdql.external "ConstantString", %constds_12, %consti_13 : memref<?xi8>, i32 -> memref<6xi8>
// MED BAG
%constds_14 = "arith.constant"() <{value = dense<[77, 69, 68, 32, 66, 65, 71]> : memref<?xi8>}> : () -> memref<?xi8>
%consti_15 = "arith.constant"() <{value = 8 : i32}> : () -> i32
%mdbag = sdql.external "ConstantString", %constds_14, %consti_15 : memref<?xi8>, i32 -> memref<7xi8>
// MED BOX
%constds_16 = "arith.constant"() <{value = dense<[77, 69, 68, 32, 66, 79, 88]> : memref<?xi8>}> : () -> memref<?xi8>
%consti_17 = "arith.constant"() <{value = 8 : i32}> : () -> i32
%mdbox = sdql.external "ConstantString", %constds_16, %consti_17 : memref<?xi8>, i32 -> memref<7xi8>
// MED PACK
%constds_18 = "arith.constant"() <{value = dense<[77, 69, 68, 32, 80, 65, 67, 75]> : memref<?xi8>}> : () -> memref<?xi8>
%consti_19 = "arith.constant"() <{value = 9 : i32}> : () -> i32
%mdpack = sdql.external "ConstantString", %constds_18, %consti_19 : memref<?xi8>, i32 -> memref<8xi8>
// MED PKG
%constds_20 = "arith.constant"() <{value = dense<[77, 69, 68, 32, 80, 75, 71]> : memref<?xi8>}> : () -> memref<?xi8>
%consti_21 = "arith.constant"() <{value = 8 : i32}> : () -> i32
%mdpkg = sdql.external "ConstantString", %constds_20, %consti_21 : memref<?xi8>, i32 -> memref<7xi8>
// LG CASE
%constds_22 = "arith.constant"() <{value = dense<[76, 71, 32, 67, 65, 83, 69]> : memref<?xi8>}> : () -> memref<?xi8>
%consti_23 = "arith.constant"() <{value = 8 : i32}> : () -> i32
%lgcase = sdql.external "ConstantString", %constds_22, %consti_23 : memref<?xi8>, i32 -> memref<7xi8>
// LG BOX
%constds_24 = "arith.constant"() <{value = dense<[76, 71, 32, 66, 79, 88]> : memref<?xi8>}> : () -> memref<?xi8>
%consti_25 = "arith.constant"() <{value = 7 : i32}> : () -> i32
%lgbox = sdql.external "ConstantString", %constds_24, %consti_25 : memref<?xi8>, i32 -> memref<6xi8>
// LG PACK
%constds_26 = "arith.constant"() <{value = dense<[76, 71, 32, 80, 65, 67, 75]> : memref<?xi8>}> : () -> memref<?xi8>
%consti_27 = "arith.constant"() <{value = 8 : i32}> : () -> i32
%lgpack = sdql.external "ConstantString", %constds_26, %consti_27 : memref<?xi8>, i32 -> memref<7xi8>
// LG PKG
%constds_28 = "arith.constant"() <{value = dense<[76, 71, 32, 80, 75, 71]> : memref<?xi8>}> : () -> memref<?xi8>
%consti_29 = "arith.constant"() <{value = 7 : i32}> : () -> i32
%lgpkg = sdql.external "ConstantString", %constds_28, %consti_29 : memref<?xi8>, i32 -> memref<6xi8>
// AIR
%constds_30 = "arith.constant"() <{value = dense<[65, 73, 82]> : memref<?xi8>}> : () -> memref<?xi8>
%consti_31 = "arith.constant"() <{value = 4 : i32}> : () -> i32
%air = sdql.external "ConstantString", %constds_30, %consti_31 : memref<?xi8>, i32 -> memref<3xi8>
// AIR REG
%constds_32 = "arith.constant"() <{value = dense<[65, 73, 82, 32, 82, 69, 71]> : memref<?xi8>}> : () -> memref<?xi8>
%consti_33 = "arith.constant"() <{value = 8 : i32}> : () -> i32
%airreg = sdql.external "ConstantString", %constds_32, %consti_33 : memref<?xi8>, i32 -> memref<7xi8>
// DELIVER IN PERSON
%constds_34 = "arith.constant"() <{value = dense<[68, 69, 76, 73, 86, 69, 82, 32, 73, 78, 32, 80, 69, 82, 83, 79, 78]> : memref<?xi8>}> : () -> memref<?xi8>
%consti_35 = "arith.constant"() <{value = 18 : i32}> : () -> i32
%deliverinperson = sdql.external "ConstantString", %constds_34, %consti_35 : memref<?xi8>, i32 -> memref<17xi8>
%fieldnode_36 = sdql.access_record %part "size" : record<"p_partkey": dictionary<i32, i32>, "p_name": dictionary<i32, memref<55xi8>>, "p_mfgr": dictionary<i32, memref<25xi8>>, "p_brand": dictionary<i32, memref<10xi8>>, "p_type": dictionary<i32, memref<25xi8>>, "p_size": dictionary<i32, i32>, "p_container": dictionary<i32, memref<10xi8>>, "p_retailprice": dictionary<i32, f64>, "p_comment": dictionary<i32, memref<23xi8>>, "size": i32> -> i32
%rangenode_37 = "func.call"(%fieldnode_36) <{callee = @range_builtin}> : (i32) -> dictionary<i32, i32>
%p_h = sdql.sum %rangenode_37 : dictionary<i32, i32> -> dictionary<i32, record<"p_brand" : memref<10xi8>, "p_size" : i32, "p_container" : memref<10xi8>>> {
^bb_38(%i: i32, %_: i32):
  %fieldnode_40 = sdql.access_record %part "p_partkey" : record<"p_partkey": dictionary<i32, i32>, "p_name": dictionary<i32, memref<55xi8>>, "p_mfgr": dictionary<i32, memref<25xi8>>, "p_brand": dictionary<i32, memref<10xi8>>, "p_type": dictionary<i32, memref<25xi8>>, "p_size": dictionary<i32, i32>, "p_container": dictionary<i32, memref<10xi8>>, "p_retailprice": dictionary<i32, f64>, "p_comment": dictionary<i32, memref<23xi8>>, "size": i32> -> dictionary<i32, i32>
  %get_39 = sdql.lookup_dictionary %fieldnode_40 [%i : i32] : dictionary<i32, i32> -> i32
  %fieldnode_42 = sdql.access_record %part "p_name" : record<"p_partkey": dictionary<i32, i32>, "p_name": dictionary<i32, memref<55xi8>>, "p_mfgr": dictionary<i32, memref<25xi8>>, "p_brand": dictionary<i32, memref<10xi8>>, "p_type": dictionary<i32, memref<25xi8>>, "p_size": dictionary<i32, i32>, "p_container": dictionary<i32, memref<10xi8>>, "p_retailprice": dictionary<i32, f64>, "p_comment": dictionary<i32, memref<23xi8>>, "size": i32> -> dictionary<i32, memref<55xi8>>
  %get_41 = sdql.lookup_dictionary %fieldnode_42 [%i : i32] : dictionary<i32, memref<55xi8>> -> memref<55xi8>
  %fieldnode_44 = sdql.access_record %part "p_mfgr" : record<"p_partkey": dictionary<i32, i32>, "p_name": dictionary<i32, memref<55xi8>>, "p_mfgr": dictionary<i32, memref<25xi8>>, "p_brand": dictionary<i32, memref<10xi8>>, "p_type": dictionary<i32, memref<25xi8>>, "p_size": dictionary<i32, i32>, "p_container": dictionary<i32, memref<10xi8>>, "p_retailprice": dictionary<i32, f64>, "p_comment": dictionary<i32, memref<23xi8>>, "size": i32> -> dictionary<i32, memref<25xi8>>
  %get_43 = sdql.lookup_dictionary %fieldnode_44 [%i : i32] : dictionary<i32, memref<25xi8>> -> memref<25xi8>
  %fieldnode_46 = sdql.access_record %part "p_brand" : record<"p_partkey": dictionary<i32, i32>, "p_name": dictionary<i32, memref<55xi8>>, "p_mfgr": dictionary<i32, memref<25xi8>>, "p_brand": dictionary<i32, memref<10xi8>>, "p_type": dictionary<i32, memref<25xi8>>, "p_size": dictionary<i32, i32>, "p_container": dictionary<i32, memref<10xi8>>, "p_retailprice": dictionary<i32, f64>, "p_comment": dictionary<i32, memref<23xi8>>, "size": i32> -> dictionary<i32, memref<10xi8>>
  %get_45 = sdql.lookup_dictionary %fieldnode_46 [%i : i32] : dictionary<i32, memref<10xi8>> -> memref<10xi8>
  %fieldnode_48 = sdql.access_record %part "p_type" : record<"p_partkey": dictionary<i32, i32>, "p_name": dictionary<i32, memref<55xi8>>, "p_mfgr": dictionary<i32, memref<25xi8>>, "p_brand": dictionary<i32, memref<10xi8>>, "p_type": dictionary<i32, memref<25xi8>>, "p_size": dictionary<i32, i32>, "p_container": dictionary<i32, memref<10xi8>>, "p_retailprice": dictionary<i32, f64>, "p_comment": dictionary<i32, memref<23xi8>>, "size": i32> -> dictionary<i32, memref<25xi8>>
  %get_47 = sdql.lookup_dictionary %fieldnode_48 [%i : i32] : dictionary<i32, memref<25xi8>> -> memref<25xi8>
  %fieldnode_50 = sdql.access_record %part "p_size" : record<"p_partkey": dictionary<i32, i32>, "p_name": dictionary<i32, memref<55xi8>>, "p_mfgr": dictionary<i32, memref<25xi8>>, "p_brand": dictionary<i32, memref<10xi8>>, "p_type": dictionary<i32, memref<25xi8>>, "p_size": dictionary<i32, i32>, "p_container": dictionary<i32, memref<10xi8>>, "p_retailprice": dictionary<i32, f64>, "p_comment": dictionary<i32, memref<23xi8>>, "size": i32> -> dictionary<i32, i32>
  %get_49 = sdql.lookup_dictionary %fieldnode_50 [%i : i32] : dictionary<i32, i32> -> i32
  %fieldnode_52 = sdql.access_record %part "p_container" : record<"p_partkey": dictionary<i32, i32>, "p_name": dictionary<i32, memref<55xi8>>, "p_mfgr": dictionary<i32, memref<25xi8>>, "p_brand": dictionary<i32, memref<10xi8>>, "p_type": dictionary<i32, memref<25xi8>>, "p_size": dictionary<i32, i32>, "p_container": dictionary<i32, memref<10xi8>>, "p_retailprice": dictionary<i32, f64>, "p_comment": dictionary<i32, memref<23xi8>>, "size": i32> -> dictionary<i32, memref<10xi8>>
  %get_51 = sdql.lookup_dictionary %fieldnode_52 [%i : i32] : dictionary<i32, memref<10xi8>> -> memref<10xi8>
  %fieldnode_54 = sdql.access_record %part "p_retailprice" : record<"p_partkey": dictionary<i32, i32>, "p_name": dictionary<i32, memref<55xi8>>, "p_mfgr": dictionary<i32, memref<25xi8>>, "p_brand": dictionary<i32, memref<10xi8>>, "p_type": dictionary<i32, memref<25xi8>>, "p_size": dictionary<i32, i32>, "p_container": dictionary<i32, memref<10xi8>>, "p_retailprice": dictionary<i32, f64>, "p_comment": dictionary<i32, memref<23xi8>>, "size": i32> -> dictionary<i32, f64>
  %get_53 = sdql.lookup_dictionary %fieldnode_54 [%i : i32] : dictionary<i32, f64> -> f64
  %fieldnode_56 = sdql.access_record %part "p_comment" : record<"p_partkey": dictionary<i32, i32>, "p_name": dictionary<i32, memref<55xi8>>, "p_mfgr": dictionary<i32, memref<25xi8>>, "p_brand": dictionary<i32, memref<10xi8>>, "p_type": dictionary<i32, memref<25xi8>>, "p_size": dictionary<i32, i32>, "p_container": dictionary<i32, memref<10xi8>>, "p_retailprice": dictionary<i32, f64>, "p_comment": dictionary<i32, memref<23xi8>>, "size": i32> -> dictionary<i32, memref<23xi8>>
  %get_55 = sdql.lookup_dictionary %fieldnode_56 [%i : i32] : dictionary<i32, memref<23xi8>> -> memref<23xi8>
  %p = sdql.create_record {fields = ["p_partkey", "p_name", "p_mfgr", "p_brand", "p_type", "p_size", "p_container", "p_retailprice", "p_comment"]} %get_39, %get_41, %get_43, %get_45, %get_47, %get_49, %get_51, %get_53, %get_55 : i32, memref<55xi8>, memref<25xi8>, memref<10xi8>, memref<25xi8>, i32, memref<10xi8>, f64, memref<23xi8> -> record<"p_partkey": i32, "p_name": memref<55xi8>, "p_mfgr": memref<25xi8>, "p_brand": memref<10xi8>, "p_type": memref<25xi8>, "p_size": i32, "p_container": memref<10xi8>, "p_retailprice": f64, "p_comment": memref<23xi8>>
  %fieldnode_57 = sdql.access_record %p "p_brand" : record<"p_partkey": i32, "p_name": memref<55xi8>, "p_mfgr": memref<25xi8>, "p_brand": memref<10xi8>, "p_type": memref<25xi8>, "p_size": i32, "p_container": memref<10xi8>, "p_retailprice": f64, "p_comment": memref<23xi8>> -> memref<10xi8>
  %cmp_58 = sdql.cmp %fieldnode_57, %brand12 : memref<10xi8>, memref<8xi8> -> i1
  %if_59 = "scf.if"(%cmp_58) ({
    %fieldnode_60 = sdql.access_record %p "p_container" : record<"p_partkey": i32, "p_name": memref<55xi8>, "p_mfgr": memref<25xi8>, "p_brand": memref<10xi8>, "p_type": memref<25xi8>, "p_size": i32, "p_container": memref<10xi8>, "p_retailprice": f64, "p_comment": memref<23xi8>> -> memref<10xi8>
    %cmp_61 = sdql.cmp %fieldnode_60, %smcase : memref<10xi8>, memref<7xi8> -> i1
    %if_62 = "scf.if"(%cmp_61) ({
      %constv_63 = "arith.constant"() <{value = 1 : i1}> : () -> i1
      "scf.yield"(%constv_63) : (i1) -> ()
    }, {
      %fieldnode_64 = sdql.access_record %p "p_container" : record<"p_partkey": i32, "p_name": memref<55xi8>, "p_mfgr": memref<25xi8>, "p_brand": memref<10xi8>, "p_type": memref<25xi8>, "p_size": i32, "p_container": memref<10xi8>, "p_retailprice": f64, "p_comment": memref<23xi8>> -> memref<10xi8>
      %cmp_65 = sdql.cmp %fieldnode_64, %smbox : memref<10xi8>, memref<6xi8> -> i1
      "scf.yield"(%cmp_65) : (i1) -> ()
    }) : (i1) -> i1
    %if_66 = "scf.if"(%if_62) ({
      %constv_67 = "arith.constant"() <{value = 1 : i1}> : () -> i1
      "scf.yield"(%constv_67) : (i1) -> ()
    }, {
      %fieldnode_68 = sdql.access_record %p "p_container" : record<"p_partkey": i32, "p_name": memref<55xi8>, "p_mfgr": memref<25xi8>, "p_brand": memref<10xi8>, "p_type": memref<25xi8>, "p_size": i32, "p_container": memref<10xi8>, "p_retailprice": f64, "p_comment": memref<23xi8>> -> memref<10xi8>
      %cmp_69 = sdql.cmp %fieldnode_68, %smpack : memref<10xi8>, memref<7xi8> -> i1
      "scf.yield"(%cmp_69) : (i1) -> ()
    }) : (i1) -> i1
    %if_70 = "scf.if"(%if_66) ({
      %constv_71 = "arith.constant"() <{value = 1 : i1}> : () -> i1
      "scf.yield"(%constv_71) : (i1) -> ()
    }, {
      %fieldnode_72 = sdql.access_record %p "p_container" : record<"p_partkey": i32, "p_name": memref<55xi8>, "p_mfgr": memref<25xi8>, "p_brand": memref<10xi8>, "p_type": memref<25xi8>, "p_size": i32, "p_container": memref<10xi8>, "p_retailprice": f64, "p_comment": memref<23xi8>> -> memref<10xi8>
      %cmp_73 = sdql.cmp %fieldnode_72, %smpkg : memref<10xi8>, memref<6xi8> -> i1
      "scf.yield"(%cmp_73) : (i1) -> ()
    }) : (i1) -> i1
    "scf.yield"(%if_70) : (i1) -> ()
  }, {
    %constv_74 = "arith.constant"() <{value = 0 : i1}> : () -> i1
    "scf.yield"(%constv_74) : (i1) -> ()
  }) : (i1) -> i1
  %if_75 = "scf.if"(%if_59) ({
    %consti_76 = "arith.constant"() <{value = 1 : i32}> : () -> i32
    %fieldnode_77 = sdql.access_record %p "p_size" : record<"p_partkey": i32, "p_name": memref<55xi8>, "p_mfgr": memref<25xi8>, "p_brand": memref<10xi8>, "p_type": memref<25xi8>, "p_size": i32, "p_container": memref<10xi8>, "p_retailprice": f64, "p_comment": memref<23xi8>> -> i32
    %cmpi_78 = "arith.cmpi"(%consti_76, %fieldnode_77) <{predicate = 3}> : (i32, i32) -> i1
    "scf.yield"(%cmpi_78) : (i1) -> ()
  }, {
    %constv_79 = "arith.constant"() <{value = 0 : i1}> : () -> i1
    "scf.yield"(%constv_79) : (i1) -> ()
  }) : (i1) -> i1
  %if_80 = "scf.if"(%if_75) ({
    %fieldnode_81 = sdql.access_record %p "p_size" : record<"p_partkey": i32, "p_name": memref<55xi8>, "p_mfgr": memref<25xi8>, "p_brand": memref<10xi8>, "p_type": memref<25xi8>, "p_size": i32, "p_container": memref<10xi8>, "p_retailprice": f64, "p_comment": memref<23xi8>> -> i32
    %consti_82 = "arith.constant"() <{value = 5 : i32}> : () -> i32
    %cmpi_83 = "arith.cmpi"(%fieldnode_81, %consti_82) <{predicate = 3}> : (i32, i32) -> i1
    "scf.yield"(%cmpi_83) : (i1) -> ()
  }, {
    %constv_84 = "arith.constant"() <{value = 0 : i1}> : () -> i1
    "scf.yield"(%constv_84) : (i1) -> ()
  }) : (i1) -> i1
  %if_85 = "scf.if"(%if_80) ({
    %constv_86 = "arith.constant"() <{value = 1 : i1}> : () -> i1
    "scf.yield"(%constv_86) : (i1) -> ()
  }, {
    %fieldnode_87 = sdql.access_record %p "p_brand" : record<"p_partkey": i32, "p_name": memref<55xi8>, "p_mfgr": memref<25xi8>, "p_brand": memref<10xi8>, "p_type": memref<25xi8>, "p_size": i32, "p_container": memref<10xi8>, "p_retailprice": f64, "p_comment": memref<23xi8>> -> memref<10xi8>
    %cmp_88 = sdql.cmp %fieldnode_87, %brand23 : memref<10xi8>, memref<8xi8> -> i1
    %if_89 = "scf.if"(%cmp_88) ({
      %fieldnode_90 = sdql.access_record %p "p_container" : record<"p_partkey": i32, "p_name": memref<55xi8>, "p_mfgr": memref<25xi8>, "p_brand": memref<10xi8>, "p_type": memref<25xi8>, "p_size": i32, "p_container": memref<10xi8>, "p_retailprice": f64, "p_comment": memref<23xi8>> -> memref<10xi8>
      %cmp_91 = sdql.cmp %fieldnode_90, %mdbag : memref<10xi8>, memref<7xi8> -> i1
      %if_92 = "scf.if"(%cmp_91) ({
        %constv_93 = "arith.constant"() <{value = 1 : i1}> : () -> i1
        "scf.yield"(%constv_93) : (i1) -> ()
      }, {
        %fieldnode_94 = sdql.access_record %p "p_container" : record<"p_partkey": i32, "p_name": memref<55xi8>, "p_mfgr": memref<25xi8>, "p_brand": memref<10xi8>, "p_type": memref<25xi8>, "p_size": i32, "p_container": memref<10xi8>, "p_retailprice": f64, "p_comment": memref<23xi8>> -> memref<10xi8>
        %cmp_95 = sdql.cmp %fieldnode_94, %mdbox : memref<10xi8>, memref<7xi8> -> i1
        "scf.yield"(%cmp_95) : (i1) -> ()
      }) : (i1) -> i1
      %if_96 = "scf.if"(%if_92) ({
        %constv_97 = "arith.constant"() <{value = 1 : i1}> : () -> i1
        "scf.yield"(%constv_97) : (i1) -> ()
      }, {
        %fieldnode_98 = sdql.access_record %p "p_container" : record<"p_partkey": i32, "p_name": memref<55xi8>, "p_mfgr": memref<25xi8>, "p_brand": memref<10xi8>, "p_type": memref<25xi8>, "p_size": i32, "p_container": memref<10xi8>, "p_retailprice": f64, "p_comment": memref<23xi8>> -> memref<10xi8>
        %cmp_99 = sdql.cmp %fieldnode_98, %mdpack : memref<10xi8>, memref<8xi8> -> i1
        "scf.yield"(%cmp_99) : (i1) -> ()
      }) : (i1) -> i1
      %if_100 = "scf.if"(%if_96) ({
        %constv_101 = "arith.constant"() <{value = 1 : i1}> : () -> i1
        "scf.yield"(%constv_101) : (i1) -> ()
      }, {
        %fieldnode_102 = sdql.access_record %p "p_container" : record<"p_partkey": i32, "p_name": memref<55xi8>, "p_mfgr": memref<25xi8>, "p_brand": memref<10xi8>, "p_type": memref<25xi8>, "p_size": i32, "p_container": memref<10xi8>, "p_retailprice": f64, "p_comment": memref<23xi8>> -> memref<10xi8>
        %cmp_103 = sdql.cmp %fieldnode_102, %mdpkg : memref<10xi8>, memref<7xi8> -> i1
        "scf.yield"(%cmp_103) : (i1) -> ()
      }) : (i1) -> i1
      "scf.yield"(%if_100) : (i1) -> ()
    }, {
      %constv_104 = "arith.constant"() <{value = 0 : i1}> : () -> i1
      "scf.yield"(%constv_104) : (i1) -> ()
    }) : (i1) -> i1
    %if_105 = "scf.if"(%if_89) ({
      %consti_106 = "arith.constant"() <{value = 1 : i32}> : () -> i32
      %fieldnode_107 = sdql.access_record %p "p_size" : record<"p_partkey": i32, "p_name": memref<55xi8>, "p_mfgr": memref<25xi8>, "p_brand": memref<10xi8>, "p_type": memref<25xi8>, "p_size": i32, "p_container": memref<10xi8>, "p_retailprice": f64, "p_comment": memref<23xi8>> -> i32
      %cmpi_108 = "arith.cmpi"(%consti_106, %fieldnode_107) <{predicate = 3}> : (i32, i32) -> i1
      "scf.yield"(%cmpi_108) : (i1) -> ()
    }, {
      %constv_109 = "arith.constant"() <{value = 0 : i1}> : () -> i1
      "scf.yield"(%constv_109) : (i1) -> ()
    }) : (i1) -> i1
    %if_110 = "scf.if"(%if_105) ({
      %fieldnode_111 = sdql.access_record %p "p_size" : record<"p_partkey": i32, "p_name": memref<55xi8>, "p_mfgr": memref<25xi8>, "p_brand": memref<10xi8>, "p_type": memref<25xi8>, "p_size": i32, "p_container": memref<10xi8>, "p_retailprice": f64, "p_comment": memref<23xi8>> -> i32
      %consti_112 = "arith.constant"() <{value = 10 : i32}> : () -> i32
      %cmpi_113 = "arith.cmpi"(%fieldnode_111, %consti_112) <{predicate = 3}> : (i32, i32) -> i1
      "scf.yield"(%cmpi_113) : (i1) -> ()
    }, {
      %constv_114 = "arith.constant"() <{value = 0 : i1}> : () -> i1
      "scf.yield"(%constv_114) : (i1) -> ()
    }) : (i1) -> i1
    "scf.yield"(%if_110) : (i1) -> ()
  }) : (i1) -> i1
  %if_115 = "scf.if"(%if_85) ({
    %constv_116 = "arith.constant"() <{value = 1 : i1}> : () -> i1
    "scf.yield"(%constv_116) : (i1) -> ()
  }, {
    %fieldnode_117 = sdql.access_record %p "p_brand" : record<"p_partkey": i32, "p_name": memref<55xi8>, "p_mfgr": memref<25xi8>, "p_brand": memref<10xi8>, "p_type": memref<25xi8>, "p_size": i32, "p_container": memref<10xi8>, "p_retailprice": f64, "p_comment": memref<23xi8>> -> memref<10xi8>
    %cmp_118 = sdql.cmp %fieldnode_117, %brand34 : memref<10xi8>, memref<8xi8> -> i1
    %if_119 = "scf.if"(%cmp_118) ({
      %fieldnode_120 = sdql.access_record %p "p_container" : record<"p_partkey": i32, "p_name": memref<55xi8>, "p_mfgr": memref<25xi8>, "p_brand": memref<10xi8>, "p_type": memref<25xi8>, "p_size": i32, "p_container": memref<10xi8>, "p_retailprice": f64, "p_comment": memref<23xi8>> -> memref<10xi8>
      %cmp_121 = sdql.cmp %fieldnode_120, %lgcase : memref<10xi8>, memref<7xi8> -> i1
      %if_122 = "scf.if"(%cmp_121) ({
        %constv_123 = "arith.constant"() <{value = 1 : i1}> : () -> i1
        "scf.yield"(%constv_123) : (i1) -> ()
      }, {
        %fieldnode_124 = sdql.access_record %p "p_container" : record<"p_partkey": i32, "p_name": memref<55xi8>, "p_mfgr": memref<25xi8>, "p_brand": memref<10xi8>, "p_type": memref<25xi8>, "p_size": i32, "p_container": memref<10xi8>, "p_retailprice": f64, "p_comment": memref<23xi8>> -> memref<10xi8>
        %cmp_125 = sdql.cmp %fieldnode_124, %lgbox : memref<10xi8>, memref<6xi8> -> i1
        "scf.yield"(%cmp_125) : (i1) -> ()
      }) : (i1) -> i1
      %if_126 = "scf.if"(%if_122) ({
        %constv_127 = "arith.constant"() <{value = 1 : i1}> : () -> i1
        "scf.yield"(%constv_127) : (i1) -> ()
      }, {
        %fieldnode_128 = sdql.access_record %p "p_container" : record<"p_partkey": i32, "p_name": memref<55xi8>, "p_mfgr": memref<25xi8>, "p_brand": memref<10xi8>, "p_type": memref<25xi8>, "p_size": i32, "p_container": memref<10xi8>, "p_retailprice": f64, "p_comment": memref<23xi8>> -> memref<10xi8>
        %cmp_129 = sdql.cmp %fieldnode_128, %lgpack : memref<10xi8>, memref<7xi8> -> i1
        "scf.yield"(%cmp_129) : (i1) -> ()
      }) : (i1) -> i1
      %if_130 = "scf.if"(%if_126) ({
        %constv_131 = "arith.constant"() <{value = 1 : i1}> : () -> i1
        "scf.yield"(%constv_131) : (i1) -> ()
      }, {
        %fieldnode_132 = sdql.access_record %p "p_container" : record<"p_partkey": i32, "p_name": memref<55xi8>, "p_mfgr": memref<25xi8>, "p_brand": memref<10xi8>, "p_type": memref<25xi8>, "p_size": i32, "p_container": memref<10xi8>, "p_retailprice": f64, "p_comment": memref<23xi8>> -> memref<10xi8>
        %cmp_133 = sdql.cmp %fieldnode_132, %lgpkg : memref<10xi8>, memref<6xi8> -> i1
        "scf.yield"(%cmp_133) : (i1) -> ()
      }) : (i1) -> i1
      "scf.yield"(%if_130) : (i1) -> ()
    }, {
      %constv_134 = "arith.constant"() <{value = 0 : i1}> : () -> i1
      "scf.yield"(%constv_134) : (i1) -> ()
    }) : (i1) -> i1
    %if_135 = "scf.if"(%if_119) ({
      %consti_136 = "arith.constant"() <{value = 1 : i32}> : () -> i32
      %fieldnode_137 = sdql.access_record %p "p_size" : record<"p_partkey": i32, "p_name": memref<55xi8>, "p_mfgr": memref<25xi8>, "p_brand": memref<10xi8>, "p_type": memref<25xi8>, "p_size": i32, "p_container": memref<10xi8>, "p_retailprice": f64, "p_comment": memref<23xi8>> -> i32
      %cmpi_138 = "arith.cmpi"(%consti_136, %fieldnode_137) <{predicate = 3}> : (i32, i32) -> i1
      "scf.yield"(%cmpi_138) : (i1) -> ()
    }, {
      %constv_139 = "arith.constant"() <{value = 0 : i1}> : () -> i1
      "scf.yield"(%constv_139) : (i1) -> ()
    }) : (i1) -> i1
    %if_140 = "scf.if"(%if_135) ({
      %fieldnode_141 = sdql.access_record %p "p_size" : record<"p_partkey": i32, "p_name": memref<55xi8>, "p_mfgr": memref<25xi8>, "p_brand": memref<10xi8>, "p_type": memref<25xi8>, "p_size": i32, "p_container": memref<10xi8>, "p_retailprice": f64, "p_comment": memref<23xi8>> -> i32
      %consti_142 = "arith.constant"() <{value = 15 : i32}> : () -> i32
      %cmpi_143 = "arith.cmpi"(%fieldnode_141, %consti_142) <{predicate = 3}> : (i32, i32) -> i1
      "scf.yield"(%cmpi_143) : (i1) -> ()
    }, {
      %constv_144 = "arith.constant"() <{value = 0 : i1}> : () -> i1
      "scf.yield"(%constv_144) : (i1) -> ()
    }) : (i1) -> i1
    "scf.yield"(%if_140) : (i1) -> ()
  }) : (i1) -> i1
  %if_145 = "scf.if"(%if_115) ({
    %fieldnode_148 = sdql.access_record %p "p_partkey" : record<"p_partkey": i32, "p_name": memref<55xi8>, "p_mfgr": memref<25xi8>, "p_brand": memref<10xi8>, "p_type": memref<25xi8>, "p_size": i32, "p_container": memref<10xi8>, "p_retailprice": f64, "p_comment": memref<23xi8>> -> i32
    %uniq_147 = sdql.unique %fieldnode_148 : i32 -> i32
    %fieldnode_149 = sdql.access_record %p "p_brand" : record<"p_partkey": i32, "p_name": memref<55xi8>, "p_mfgr": memref<25xi8>, "p_brand": memref<10xi8>, "p_type": memref<25xi8>, "p_size": i32, "p_container": memref<10xi8>, "p_retailprice": f64, "p_comment": memref<23xi8>> -> memref<10xi8>
    %fieldnode_150 = sdql.access_record %p "p_size" : record<"p_partkey": i32, "p_name": memref<55xi8>, "p_mfgr": memref<25xi8>, "p_brand": memref<10xi8>, "p_type": memref<25xi8>, "p_size": i32, "p_container": memref<10xi8>, "p_retailprice": f64, "p_comment": memref<23xi8>> -> i32
    %fieldnode_151 = sdql.access_record %p "p_container" : record<"p_partkey": i32, "p_name": memref<55xi8>, "p_mfgr": memref<25xi8>, "p_brand": memref<10xi8>, "p_type": memref<25xi8>, "p_size": i32, "p_container": memref<10xi8>, "p_retailprice": f64, "p_comment": memref<23xi8>> -> memref<10xi8>
    %recnode_152 = sdql.create_record {fields = ["p_brand", "p_size", "p_container"]} %fieldnode_149, %fieldnode_150, %fieldnode_151 : memref<10xi8>, i32, memref<10xi8> -> record<"p_brand": memref<10xi8>, "p_size": i32, "p_container": memref<10xi8>>
    %dict_146 = sdql.create_dictionary %uniq_147, %recnode_152 : i32, record<"p_brand" : memref<10xi8>, "p_size" : i32, "p_container" : memref<10xi8>> -> dictionary<i32, record<"p_brand" : memref<10xi8>, "p_size" : i32, "p_container" : memref<10xi8>>>
    "scf.yield"(%dict_146) : (dictionary<i32, record<"p_brand" : memref<10xi8>, "p_size" : i32, "p_container" : memref<10xi8>>>) -> ()
  }, {
    %dict_153 = sdql.empty_dictionary : dictionary<i32, record<"p_brand" : memref<10xi8>, "p_size" : i32, "p_container" : memref<10xi8>>>
    "scf.yield"(%dict_153) : (dictionary<i32, record<"p_brand" : memref<10xi8>, "p_size" : i32, "p_container" : memref<10xi8>>>) -> ()
  }) : (i1) -> dictionary<i32, record<"p_brand" : memref<10xi8>, "p_size" : i32, "p_container" : memref<10xi8>>>
  sdql.yield %if_145 : dictionary<i32, record<"p_brand" : memref<10xi8>, "p_size" : i32, "p_container" : memref<10xi8>>>
}
%fieldnode_154 = sdql.access_record %lineitem "size" : record<"l_orderkey": dictionary<i32, i32>, "l_partkey": dictionary<i32, i32>, "l_suppkey": dictionary<i32, i32>, "l_linenumber": dictionary<i32, i32>, "l_quantity": dictionary<i32, f64>, "l_extendedprice": dictionary<i32, f64>, "l_discount": dictionary<i32, f64>, "l_tax": dictionary<i32, f64>, "l_returnflag": dictionary<i32, memref<1xi8>>, "l_linestatus": dictionary<i32, memref<1xi8>>, "l_shipdate": dictionary<i32, i32>, "l_commitdate": dictionary<i32, i32>, "l_receiptdate": dictionary<i32, i32>, "l_shipinstruct": dictionary<i32, memref<25xi8>>, "l_shipmode": dictionary<i32, memref<10xi8>>, "l_comment": dictionary<i32, memref<44xi8>>, "size": i32> -> i32
%rangenode_155 = "func.call"(%fieldnode_154) <{callee = @range_builtin}> : (i32) -> dictionary<i32, i32>
%res = sdql.sum %rangenode_155 : dictionary<i32, i32> -> f64 {
^bb_156(%i: i32, %_: i32):
  %fieldnode_158 = sdql.access_record %lineitem "l_orderkey" : record<"l_orderkey": dictionary<i32, i32>, "l_partkey": dictionary<i32, i32>, "l_suppkey": dictionary<i32, i32>, "l_linenumber": dictionary<i32, i32>, "l_quantity": dictionary<i32, f64>, "l_extendedprice": dictionary<i32, f64>, "l_discount": dictionary<i32, f64>, "l_tax": dictionary<i32, f64>, "l_returnflag": dictionary<i32, memref<1xi8>>, "l_linestatus": dictionary<i32, memref<1xi8>>, "l_shipdate": dictionary<i32, i32>, "l_commitdate": dictionary<i32, i32>, "l_receiptdate": dictionary<i32, i32>, "l_shipinstruct": dictionary<i32, memref<25xi8>>, "l_shipmode": dictionary<i32, memref<10xi8>>, "l_comment": dictionary<i32, memref<44xi8>>, "size": i32> -> dictionary<i32, i32>
  %get_157 = sdql.lookup_dictionary %fieldnode_158 [%i : i32] : dictionary<i32, i32> -> i32
  %fieldnode_160 = sdql.access_record %lineitem "l_partkey" : record<"l_orderkey": dictionary<i32, i32>, "l_partkey": dictionary<i32, i32>, "l_suppkey": dictionary<i32, i32>, "l_linenumber": dictionary<i32, i32>, "l_quantity": dictionary<i32, f64>, "l_extendedprice": dictionary<i32, f64>, "l_discount": dictionary<i32, f64>, "l_tax": dictionary<i32, f64>, "l_returnflag": dictionary<i32, memref<1xi8>>, "l_linestatus": dictionary<i32, memref<1xi8>>, "l_shipdate": dictionary<i32, i32>, "l_commitdate": dictionary<i32, i32>, "l_receiptdate": dictionary<i32, i32>, "l_shipinstruct": dictionary<i32, memref<25xi8>>, "l_shipmode": dictionary<i32, memref<10xi8>>, "l_comment": dictionary<i32, memref<44xi8>>, "size": i32> -> dictionary<i32, i32>
  %get_159 = sdql.lookup_dictionary %fieldnode_160 [%i : i32] : dictionary<i32, i32> -> i32
  %fieldnode_162 = sdql.access_record %lineitem "l_suppkey" : record<"l_orderkey": dictionary<i32, i32>, "l_partkey": dictionary<i32, i32>, "l_suppkey": dictionary<i32, i32>, "l_linenumber": dictionary<i32, i32>, "l_quantity": dictionary<i32, f64>, "l_extendedprice": dictionary<i32, f64>, "l_discount": dictionary<i32, f64>, "l_tax": dictionary<i32, f64>, "l_returnflag": dictionary<i32, memref<1xi8>>, "l_linestatus": dictionary<i32, memref<1xi8>>, "l_shipdate": dictionary<i32, i32>, "l_commitdate": dictionary<i32, i32>, "l_receiptdate": dictionary<i32, i32>, "l_shipinstruct": dictionary<i32, memref<25xi8>>, "l_shipmode": dictionary<i32, memref<10xi8>>, "l_comment": dictionary<i32, memref<44xi8>>, "size": i32> -> dictionary<i32, i32>
  %get_161 = sdql.lookup_dictionary %fieldnode_162 [%i : i32] : dictionary<i32, i32> -> i32
  %fieldnode_164 = sdql.access_record %lineitem "l_linenumber" : record<"l_orderkey": dictionary<i32, i32>, "l_partkey": dictionary<i32, i32>, "l_suppkey": dictionary<i32, i32>, "l_linenumber": dictionary<i32, i32>, "l_quantity": dictionary<i32, f64>, "l_extendedprice": dictionary<i32, f64>, "l_discount": dictionary<i32, f64>, "l_tax": dictionary<i32, f64>, "l_returnflag": dictionary<i32, memref<1xi8>>, "l_linestatus": dictionary<i32, memref<1xi8>>, "l_shipdate": dictionary<i32, i32>, "l_commitdate": dictionary<i32, i32>, "l_receiptdate": dictionary<i32, i32>, "l_shipinstruct": dictionary<i32, memref<25xi8>>, "l_shipmode": dictionary<i32, memref<10xi8>>, "l_comment": dictionary<i32, memref<44xi8>>, "size": i32> -> dictionary<i32, i32>
  %get_163 = sdql.lookup_dictionary %fieldnode_164 [%i : i32] : dictionary<i32, i32> -> i32
  %fieldnode_166 = sdql.access_record %lineitem "l_quantity" : record<"l_orderkey": dictionary<i32, i32>, "l_partkey": dictionary<i32, i32>, "l_suppkey": dictionary<i32, i32>, "l_linenumber": dictionary<i32, i32>, "l_quantity": dictionary<i32, f64>, "l_extendedprice": dictionary<i32, f64>, "l_discount": dictionary<i32, f64>, "l_tax": dictionary<i32, f64>, "l_returnflag": dictionary<i32, memref<1xi8>>, "l_linestatus": dictionary<i32, memref<1xi8>>, "l_shipdate": dictionary<i32, i32>, "l_commitdate": dictionary<i32, i32>, "l_receiptdate": dictionary<i32, i32>, "l_shipinstruct": dictionary<i32, memref<25xi8>>, "l_shipmode": dictionary<i32, memref<10xi8>>, "l_comment": dictionary<i32, memref<44xi8>>, "size": i32> -> dictionary<i32, f64>
  %get_165 = sdql.lookup_dictionary %fieldnode_166 [%i : i32] : dictionary<i32, f64> -> f64
  %fieldnode_168 = sdql.access_record %lineitem "l_extendedprice" : record<"l_orderkey": dictionary<i32, i32>, "l_partkey": dictionary<i32, i32>, "l_suppkey": dictionary<i32, i32>, "l_linenumber": dictionary<i32, i32>, "l_quantity": dictionary<i32, f64>, "l_extendedprice": dictionary<i32, f64>, "l_discount": dictionary<i32, f64>, "l_tax": dictionary<i32, f64>, "l_returnflag": dictionary<i32, memref<1xi8>>, "l_linestatus": dictionary<i32, memref<1xi8>>, "l_shipdate": dictionary<i32, i32>, "l_commitdate": dictionary<i32, i32>, "l_receiptdate": dictionary<i32, i32>, "l_shipinstruct": dictionary<i32, memref<25xi8>>, "l_shipmode": dictionary<i32, memref<10xi8>>, "l_comment": dictionary<i32, memref<44xi8>>, "size": i32> -> dictionary<i32, f64>
  %get_167 = sdql.lookup_dictionary %fieldnode_168 [%i : i32] : dictionary<i32, f64> -> f64
  %fieldnode_170 = sdql.access_record %lineitem "l_discount" : record<"l_orderkey": dictionary<i32, i32>, "l_partkey": dictionary<i32, i32>, "l_suppkey": dictionary<i32, i32>, "l_linenumber": dictionary<i32, i32>, "l_quantity": dictionary<i32, f64>, "l_extendedprice": dictionary<i32, f64>, "l_discount": dictionary<i32, f64>, "l_tax": dictionary<i32, f64>, "l_returnflag": dictionary<i32, memref<1xi8>>, "l_linestatus": dictionary<i32, memref<1xi8>>, "l_shipdate": dictionary<i32, i32>, "l_commitdate": dictionary<i32, i32>, "l_receiptdate": dictionary<i32, i32>, "l_shipinstruct": dictionary<i32, memref<25xi8>>, "l_shipmode": dictionary<i32, memref<10xi8>>, "l_comment": dictionary<i32, memref<44xi8>>, "size": i32> -> dictionary<i32, f64>
  %get_169 = sdql.lookup_dictionary %fieldnode_170 [%i : i32] : dictionary<i32, f64> -> f64
  %fieldnode_172 = sdql.access_record %lineitem "l_tax" : record<"l_orderkey": dictionary<i32, i32>, "l_partkey": dictionary<i32, i32>, "l_suppkey": dictionary<i32, i32>, "l_linenumber": dictionary<i32, i32>, "l_quantity": dictionary<i32, f64>, "l_extendedprice": dictionary<i32, f64>, "l_discount": dictionary<i32, f64>, "l_tax": dictionary<i32, f64>, "l_returnflag": dictionary<i32, memref<1xi8>>, "l_linestatus": dictionary<i32, memref<1xi8>>, "l_shipdate": dictionary<i32, i32>, "l_commitdate": dictionary<i32, i32>, "l_receiptdate": dictionary<i32, i32>, "l_shipinstruct": dictionary<i32, memref<25xi8>>, "l_shipmode": dictionary<i32, memref<10xi8>>, "l_comment": dictionary<i32, memref<44xi8>>, "size": i32> -> dictionary<i32, f64>
  %get_171 = sdql.lookup_dictionary %fieldnode_172 [%i : i32] : dictionary<i32, f64> -> f64
  %fieldnode_174 = sdql.access_record %lineitem "l_returnflag" : record<"l_orderkey": dictionary<i32, i32>, "l_partkey": dictionary<i32, i32>, "l_suppkey": dictionary<i32, i32>, "l_linenumber": dictionary<i32, i32>, "l_quantity": dictionary<i32, f64>, "l_extendedprice": dictionary<i32, f64>, "l_discount": dictionary<i32, f64>, "l_tax": dictionary<i32, f64>, "l_returnflag": dictionary<i32, memref<1xi8>>, "l_linestatus": dictionary<i32, memref<1xi8>>, "l_shipdate": dictionary<i32, i32>, "l_commitdate": dictionary<i32, i32>, "l_receiptdate": dictionary<i32, i32>, "l_shipinstruct": dictionary<i32, memref<25xi8>>, "l_shipmode": dictionary<i32, memref<10xi8>>, "l_comment": dictionary<i32, memref<44xi8>>, "size": i32> -> dictionary<i32, memref<1xi8>>
  %get_173 = sdql.lookup_dictionary %fieldnode_174 [%i : i32] : dictionary<i32, memref<1xi8>> -> memref<1xi8>
  %fieldnode_176 = sdql.access_record %lineitem "l_linestatus" : record<"l_orderkey": dictionary<i32, i32>, "l_partkey": dictionary<i32, i32>, "l_suppkey": dictionary<i32, i32>, "l_linenumber": dictionary<i32, i32>, "l_quantity": dictionary<i32, f64>, "l_extendedprice": dictionary<i32, f64>, "l_discount": dictionary<i32, f64>, "l_tax": dictionary<i32, f64>, "l_returnflag": dictionary<i32, memref<1xi8>>, "l_linestatus": dictionary<i32, memref<1xi8>>, "l_shipdate": dictionary<i32, i32>, "l_commitdate": dictionary<i32, i32>, "l_receiptdate": dictionary<i32, i32>, "l_shipinstruct": dictionary<i32, memref<25xi8>>, "l_shipmode": dictionary<i32, memref<10xi8>>, "l_comment": dictionary<i32, memref<44xi8>>, "size": i32> -> dictionary<i32, memref<1xi8>>
  %get_175 = sdql.lookup_dictionary %fieldnode_176 [%i : i32] : dictionary<i32, memref<1xi8>> -> memref<1xi8>
  %fieldnode_178 = sdql.access_record %lineitem "l_shipdate" : record<"l_orderkey": dictionary<i32, i32>, "l_partkey": dictionary<i32, i32>, "l_suppkey": dictionary<i32, i32>, "l_linenumber": dictionary<i32, i32>, "l_quantity": dictionary<i32, f64>, "l_extendedprice": dictionary<i32, f64>, "l_discount": dictionary<i32, f64>, "l_tax": dictionary<i32, f64>, "l_returnflag": dictionary<i32, memref<1xi8>>, "l_linestatus": dictionary<i32, memref<1xi8>>, "l_shipdate": dictionary<i32, i32>, "l_commitdate": dictionary<i32, i32>, "l_receiptdate": dictionary<i32, i32>, "l_shipinstruct": dictionary<i32, memref<25xi8>>, "l_shipmode": dictionary<i32, memref<10xi8>>, "l_comment": dictionary<i32, memref<44xi8>>, "size": i32> -> dictionary<i32, i32>
  %get_177 = sdql.lookup_dictionary %fieldnode_178 [%i : i32] : dictionary<i32, i32> -> i32
  %fieldnode_180 = sdql.access_record %lineitem "l_commitdate" : record<"l_orderkey": dictionary<i32, i32>, "l_partkey": dictionary<i32, i32>, "l_suppkey": dictionary<i32, i32>, "l_linenumber": dictionary<i32, i32>, "l_quantity": dictionary<i32, f64>, "l_extendedprice": dictionary<i32, f64>, "l_discount": dictionary<i32, f64>, "l_tax": dictionary<i32, f64>, "l_returnflag": dictionary<i32, memref<1xi8>>, "l_linestatus": dictionary<i32, memref<1xi8>>, "l_shipdate": dictionary<i32, i32>, "l_commitdate": dictionary<i32, i32>, "l_receiptdate": dictionary<i32, i32>, "l_shipinstruct": dictionary<i32, memref<25xi8>>, "l_shipmode": dictionary<i32, memref<10xi8>>, "l_comment": dictionary<i32, memref<44xi8>>, "size": i32> -> dictionary<i32, i32>
  %get_179 = sdql.lookup_dictionary %fieldnode_180 [%i : i32] : dictionary<i32, i32> -> i32
  %fieldnode_182 = sdql.access_record %lineitem "l_receiptdate" : record<"l_orderkey": dictionary<i32, i32>, "l_partkey": dictionary<i32, i32>, "l_suppkey": dictionary<i32, i32>, "l_linenumber": dictionary<i32, i32>, "l_quantity": dictionary<i32, f64>, "l_extendedprice": dictionary<i32, f64>, "l_discount": dictionary<i32, f64>, "l_tax": dictionary<i32, f64>, "l_returnflag": dictionary<i32, memref<1xi8>>, "l_linestatus": dictionary<i32, memref<1xi8>>, "l_shipdate": dictionary<i32, i32>, "l_commitdate": dictionary<i32, i32>, "l_receiptdate": dictionary<i32, i32>, "l_shipinstruct": dictionary<i32, memref<25xi8>>, "l_shipmode": dictionary<i32, memref<10xi8>>, "l_comment": dictionary<i32, memref<44xi8>>, "size": i32> -> dictionary<i32, i32>
  %get_181 = sdql.lookup_dictionary %fieldnode_182 [%i : i32] : dictionary<i32, i32> -> i32
  %fieldnode_184 = sdql.access_record %lineitem "l_shipinstruct" : record<"l_orderkey": dictionary<i32, i32>, "l_partkey": dictionary<i32, i32>, "l_suppkey": dictionary<i32, i32>, "l_linenumber": dictionary<i32, i32>, "l_quantity": dictionary<i32, f64>, "l_extendedprice": dictionary<i32, f64>, "l_discount": dictionary<i32, f64>, "l_tax": dictionary<i32, f64>, "l_returnflag": dictionary<i32, memref<1xi8>>, "l_linestatus": dictionary<i32, memref<1xi8>>, "l_shipdate": dictionary<i32, i32>, "l_commitdate": dictionary<i32, i32>, "l_receiptdate": dictionary<i32, i32>, "l_shipinstruct": dictionary<i32, memref<25xi8>>, "l_shipmode": dictionary<i32, memref<10xi8>>, "l_comment": dictionary<i32, memref<44xi8>>, "size": i32> -> dictionary<i32, memref<25xi8>>
  %get_183 = sdql.lookup_dictionary %fieldnode_184 [%i : i32] : dictionary<i32, memref<25xi8>> -> memref<25xi8>
  %fieldnode_186 = sdql.access_record %lineitem "l_shipmode" : record<"l_orderkey": dictionary<i32, i32>, "l_partkey": dictionary<i32, i32>, "l_suppkey": dictionary<i32, i32>, "l_linenumber": dictionary<i32, i32>, "l_quantity": dictionary<i32, f64>, "l_extendedprice": dictionary<i32, f64>, "l_discount": dictionary<i32, f64>, "l_tax": dictionary<i32, f64>, "l_returnflag": dictionary<i32, memref<1xi8>>, "l_linestatus": dictionary<i32, memref<1xi8>>, "l_shipdate": dictionary<i32, i32>, "l_commitdate": dictionary<i32, i32>, "l_receiptdate": dictionary<i32, i32>, "l_shipinstruct": dictionary<i32, memref<25xi8>>, "l_shipmode": dictionary<i32, memref<10xi8>>, "l_comment": dictionary<i32, memref<44xi8>>, "size": i32> -> dictionary<i32, memref<10xi8>>
  %get_185 = sdql.lookup_dictionary %fieldnode_186 [%i : i32] : dictionary<i32, memref<10xi8>> -> memref<10xi8>
  %fieldnode_188 = sdql.access_record %lineitem "l_comment" : record<"l_orderkey": dictionary<i32, i32>, "l_partkey": dictionary<i32, i32>, "l_suppkey": dictionary<i32, i32>, "l_linenumber": dictionary<i32, i32>, "l_quantity": dictionary<i32, f64>, "l_extendedprice": dictionary<i32, f64>, "l_discount": dictionary<i32, f64>, "l_tax": dictionary<i32, f64>, "l_returnflag": dictionary<i32, memref<1xi8>>, "l_linestatus": dictionary<i32, memref<1xi8>>, "l_shipdate": dictionary<i32, i32>, "l_commitdate": dictionary<i32, i32>, "l_receiptdate": dictionary<i32, i32>, "l_shipinstruct": dictionary<i32, memref<25xi8>>, "l_shipmode": dictionary<i32, memref<10xi8>>, "l_comment": dictionary<i32, memref<44xi8>>, "size": i32> -> dictionary<i32, memref<44xi8>>
  %get_187 = sdql.lookup_dictionary %fieldnode_188 [%i : i32] : dictionary<i32, memref<44xi8>> -> memref<44xi8>
  %l = sdql.create_record {fields = ["l_orderkey", "l_partkey", "l_suppkey", "l_linenumber", "l_quantity", "l_extendedprice", "l_discount", "l_tax", "l_returnflag", "l_linestatus", "l_shipdate", "l_commitdate", "l_receiptdate", "l_shipinstruct", "l_shipmode", "l_comment"]} %get_157, %get_159, %get_161, %get_163, %get_165, %get_167, %get_169, %get_171, %get_173, %get_175, %get_177, %get_179, %get_181, %get_183, %get_185, %get_187 : i32, i32, i32, i32, f64, f64, f64, f64, memref<1xi8>, memref<1xi8>, i32, i32, i32, memref<25xi8>, memref<10xi8>, memref<44xi8> -> record<"l_orderkey": i32, "l_partkey": i32, "l_suppkey": i32, "l_linenumber": i32, "l_quantity": f64, "l_extendedprice": f64, "l_discount": f64, "l_tax": f64, "l_returnflag": memref<1xi8>, "l_linestatus": memref<1xi8>, "l_shipdate": i32, "l_commitdate": i32, "l_receiptdate": i32, "l_shipinstruct": memref<25xi8>, "l_shipmode": memref<10xi8>, "l_comment": memref<44xi8>>
  %fieldnode_190 = sdql.access_record %l "l_partkey" : record<"l_orderkey": i32, "l_partkey": i32, "l_suppkey": i32, "l_linenumber": i32, "l_quantity": f64, "l_extendedprice": f64, "l_discount": f64, "l_tax": f64, "l_returnflag": memref<1xi8>, "l_linestatus": memref<1xi8>, "l_shipdate": i32, "l_commitdate": i32, "l_receiptdate": i32, "l_shipinstruct": memref<25xi8>, "l_shipmode": memref<10xi8>, "l_comment": memref<44xi8>> -> i32
  %get_189 = sdql.lookup_dictionary %p_h [%fieldnode_190 : i32] : dictionary<i32, record<"p_brand" : memref<10xi8>, "p_size" : i32, "p_container" : memref<10xi8>>> -> record<"p_brand" : memref<10xi8>, "p_size" : i32, "p_container" : memref<10xi8>>
  %consti_191 = "arith.constant"() <{value = 0 : i32}> : () -> i32
  %p_brand = sdql.access_record %get_189 "%consti_191" : record<"p_brand" : memref<10xi8>, "p_size" : i32, "p_container" : memref<10xi8>> -> memref<10xi8>
  %fieldnode_195 = sdql.access_record %l "l_partkey" : record<"l_orderkey": i32, "l_partkey": i32, "l_suppkey": i32, "l_linenumber": i32, "l_quantity": f64, "l_extendedprice": f64, "l_discount": f64, "l_tax": f64, "l_returnflag": memref<1xi8>, "l_linestatus": memref<1xi8>, "l_shipdate": i32, "l_commitdate": i32, "l_receiptdate": i32, "l_shipinstruct": memref<25xi8>, "l_shipmode": memref<10xi8>, "l_comment": memref<44xi8>> -> i32
  %get_194 = sdql.lookup_dictionary %p_h [%fieldnode_195 : i32] : dictionary<i32, record<"p_brand" : memref<10xi8>, "p_size" : i32, "p_container" : memref<10xi8>>> -> record<"p_brand" : memref<10xi8>, "p_size" : i32, "p_container" : memref<10xi8>>
  %dict_196 = sdql.empty_dictionary : record<"p_brand" : memref<10xi8>, "p_size" : i32, "p_container" : memref<10xi8>>
  %cmp_197 = sdql.cmp %get_194, %dict_196 : record<"p_brand" : memref<10xi8>, "p_size" : i32, "p_container" : memref<10xi8>>, record<"p_brand" : memref<10xi8>, "p_size" : i32, "p_container" : memref<10xi8>> -> i1
  %zero_192 = "arith.constant"() <{value = 0 : i1}> : () -> i1
  %neg_193 = "arith.cmpi"(%cmp_197, %zero_192) <{predicate = 0}> : (i1, i1) -> i1
  %if_198 = "scf.if"(%neg_193) ({
    %fieldnode_199 = sdql.access_record %l "l_shipmode" : record<"l_orderkey": i32, "l_partkey": i32, "l_suppkey": i32, "l_linenumber": i32, "l_quantity": f64, "l_extendedprice": f64, "l_discount": f64, "l_tax": f64, "l_returnflag": memref<1xi8>, "l_linestatus": memref<1xi8>, "l_shipdate": i32, "l_commitdate": i32, "l_receiptdate": i32, "l_shipinstruct": memref<25xi8>, "l_shipmode": memref<10xi8>, "l_comment": memref<44xi8>> -> memref<10xi8>
    %cmp_200 = sdql.cmp %fieldnode_199, %air : memref<10xi8>, memref<3xi8> -> i1
    %if_201 = "scf.if"(%cmp_200) ({
      %constv_202 = "arith.constant"() <{value = 1 : i1}> : () -> i1
      "scf.yield"(%constv_202) : (i1) -> ()
    }, {
      %fieldnode_203 = sdql.access_record %l "l_shipmode" : record<"l_orderkey": i32, "l_partkey": i32, "l_suppkey": i32, "l_linenumber": i32, "l_quantity": f64, "l_extendedprice": f64, "l_discount": f64, "l_tax": f64, "l_returnflag": memref<1xi8>, "l_linestatus": memref<1xi8>, "l_shipdate": i32, "l_commitdate": i32, "l_receiptdate": i32, "l_shipinstruct": memref<25xi8>, "l_shipmode": memref<10xi8>, "l_comment": memref<44xi8>> -> memref<10xi8>
      %cmp_204 = sdql.cmp %fieldnode_203, %airreg : memref<10xi8>, memref<7xi8> -> i1
      "scf.yield"(%cmp_204) : (i1) -> ()
    }) : (i1) -> i1
    "scf.yield"(%if_201) : (i1) -> ()
  }, {
    %constv_205 = "arith.constant"() <{value = 0 : i1}> : () -> i1
    "scf.yield"(%constv_205) : (i1) -> ()
  }) : (i1) -> i1
  %if_206 = "scf.if"(%if_198) ({
    %fieldnode_207 = sdql.access_record %l "l_shipinstruct" : record<"l_orderkey": i32, "l_partkey": i32, "l_suppkey": i32, "l_linenumber": i32, "l_quantity": f64, "l_extendedprice": f64, "l_discount": f64, "l_tax": f64, "l_returnflag": memref<1xi8>, "l_linestatus": memref<1xi8>, "l_shipdate": i32, "l_commitdate": i32, "l_receiptdate": i32, "l_shipinstruct": memref<25xi8>, "l_shipmode": memref<10xi8>, "l_comment": memref<44xi8>> -> memref<25xi8>
    %cmp_208 = sdql.cmp %fieldnode_207, %deliverinperson : memref<25xi8>, memref<17xi8> -> i1
    "scf.yield"(%cmp_208) : (i1) -> ()
  }, {
    %constv_209 = "arith.constant"() <{value = 0 : i1}> : () -> i1
    "scf.yield"(%constv_209) : (i1) -> ()
  }) : (i1) -> i1
  %if_210 = "scf.if"(%if_206) ({
    %cmp_211 = sdql.cmp %p_brand, %brand12 : memref<10xi8>, memref<8xi8> -> i1
    %if_212 = "scf.if"(%cmp_211) ({
      %consti_213 = "arith.constant"() <{value = 1 : i32}> : () -> i32
      %fieldnode_214 = sdql.access_record %l "l_quantity" : record<"l_orderkey": i32, "l_partkey": i32, "l_suppkey": i32, "l_linenumber": i32, "l_quantity": f64, "l_extendedprice": f64, "l_discount": f64, "l_tax": f64, "l_returnflag": memref<1xi8>, "l_linestatus": memref<1xi8>, "l_shipdate": i32, "l_commitdate": i32, "l_receiptdate": i32, "l_shipinstruct": memref<25xi8>, "l_shipmode": memref<10xi8>, "l_comment": memref<44xi8>> -> f64
      %cmpi_215 = "arith.cmpi"(%consti_213, %fieldnode_214) <{predicate = 3}> : (i32, f64) -> i1
      "scf.yield"(%cmpi_215) : (i1) -> ()
    }, {
      %constv_216 = "arith.constant"() <{value = 0 : i1}> : () -> i1
      "scf.yield"(%constv_216) : (i1) -> ()
    }) : (i1) -> i1
    %if_217 = "scf.if"(%if_212) ({
      %fieldnode_218 = sdql.access_record %l "l_quantity" : record<"l_orderkey": i32, "l_partkey": i32, "l_suppkey": i32, "l_linenumber": i32, "l_quantity": f64, "l_extendedprice": f64, "l_discount": f64, "l_tax": f64, "l_returnflag": memref<1xi8>, "l_linestatus": memref<1xi8>, "l_shipdate": i32, "l_commitdate": i32, "l_receiptdate": i32, "l_shipinstruct": memref<25xi8>, "l_shipmode": memref<10xi8>, "l_comment": memref<44xi8>> -> f64
      %consti_219 = "arith.constant"() <{value = 11 : i32}> : () -> i32
      %cmpf_220 = "arith.cmpf"(%fieldnode_218, %consti_219) <{fastmath = #arith.fastmath<none>, predicate = 5}> : (f64, i32) -> i1
      "scf.yield"(%cmpf_220) : (i1) -> ()
    }, {
      %constv_221 = "arith.constant"() <{value = 0 : i1}> : () -> i1
      "scf.yield"(%constv_221) : (i1) -> ()
    }) : (i1) -> i1
    %if_222 = "scf.if"(%if_217) ({
      %constv_223 = "arith.constant"() <{value = 1 : i1}> : () -> i1
      "scf.yield"(%constv_223) : (i1) -> ()
    }, {
      %cmp_224 = sdql.cmp %p_brand, %brand23 : memref<10xi8>, memref<8xi8> -> i1
      %if_225 = "scf.if"(%cmp_224) ({
        %consti_226 = "arith.constant"() <{value = 10 : i32}> : () -> i32
        %fieldnode_227 = sdql.access_record %l "l_quantity" : record<"l_orderkey": i32, "l_partkey": i32, "l_suppkey": i32, "l_linenumber": i32, "l_quantity": f64, "l_extendedprice": f64, "l_discount": f64, "l_tax": f64, "l_returnflag": memref<1xi8>, "l_linestatus": memref<1xi8>, "l_shipdate": i32, "l_commitdate": i32, "l_receiptdate": i32, "l_shipinstruct": memref<25xi8>, "l_shipmode": memref<10xi8>, "l_comment": memref<44xi8>> -> f64
        %cmpi_228 = "arith.cmpi"(%consti_226, %fieldnode_227) <{predicate = 3}> : (i32, f64) -> i1
        "scf.yield"(%cmpi_228) : (i1) -> ()
      }, {
        %constv_229 = "arith.constant"() <{value = 0 : i1}> : () -> i1
        "scf.yield"(%constv_229) : (i1) -> ()
      }) : (i1) -> i1
      %if_230 = "scf.if"(%if_225) ({
        %fieldnode_231 = sdql.access_record %l "l_quantity" : record<"l_orderkey": i32, "l_partkey": i32, "l_suppkey": i32, "l_linenumber": i32, "l_quantity": f64, "l_extendedprice": f64, "l_discount": f64, "l_tax": f64, "l_returnflag": memref<1xi8>, "l_linestatus": memref<1xi8>, "l_shipdate": i32, "l_commitdate": i32, "l_receiptdate": i32, "l_shipinstruct": memref<25xi8>, "l_shipmode": memref<10xi8>, "l_comment": memref<44xi8>> -> f64
        %consti_232 = "arith.constant"() <{value = 20 : i32}> : () -> i32
        %cmpf_233 = "arith.cmpf"(%fieldnode_231, %consti_232) <{fastmath = #arith.fastmath<none>, predicate = 5}> : (f64, i32) -> i1
        "scf.yield"(%cmpf_233) : (i1) -> ()
      }, {
        %constv_234 = "arith.constant"() <{value = 0 : i1}> : () -> i1
        "scf.yield"(%constv_234) : (i1) -> ()
      }) : (i1) -> i1
      "scf.yield"(%if_230) : (i1) -> ()
    }) : (i1) -> i1
    %if_235 = "scf.if"(%if_222) ({
      %constv_236 = "arith.constant"() <{value = 1 : i1}> : () -> i1
      "scf.yield"(%constv_236) : (i1) -> ()
    }, {
      %cmp_237 = sdql.cmp %p_brand, %brand34 : memref<10xi8>, memref<8xi8> -> i1
      %if_238 = "scf.if"(%cmp_237) ({
        %consti_239 = "arith.constant"() <{value = 20 : i32}> : () -> i32
        %fieldnode_240 = sdql.access_record %l "l_quantity" : record<"l_orderkey": i32, "l_partkey": i32, "l_suppkey": i32, "l_linenumber": i32, "l_quantity": f64, "l_extendedprice": f64, "l_discount": f64, "l_tax": f64, "l_returnflag": memref<1xi8>, "l_linestatus": memref<1xi8>, "l_shipdate": i32, "l_commitdate": i32, "l_receiptdate": i32, "l_shipinstruct": memref<25xi8>, "l_shipmode": memref<10xi8>, "l_comment": memref<44xi8>> -> f64
        %cmpi_241 = "arith.cmpi"(%consti_239, %fieldnode_240) <{predicate = 3}> : (i32, f64) -> i1
        "scf.yield"(%cmpi_241) : (i1) -> ()
      }, {
        %constv_242 = "arith.constant"() <{value = 0 : i1}> : () -> i1
        "scf.yield"(%constv_242) : (i1) -> ()
      }) : (i1) -> i1
      %if_243 = "scf.if"(%if_238) ({
        %fieldnode_244 = sdql.access_record %l "l_quantity" : record<"l_orderkey": i32, "l_partkey": i32, "l_suppkey": i32, "l_linenumber": i32, "l_quantity": f64, "l_extendedprice": f64, "l_discount": f64, "l_tax": f64, "l_returnflag": memref<1xi8>, "l_linestatus": memref<1xi8>, "l_shipdate": i32, "l_commitdate": i32, "l_receiptdate": i32, "l_shipinstruct": memref<25xi8>, "l_shipmode": memref<10xi8>, "l_comment": memref<44xi8>> -> f64
        %consti_245 = "arith.constant"() <{value = 30 : i32}> : () -> i32
        %cmpf_246 = "arith.cmpf"(%fieldnode_244, %consti_245) <{fastmath = #arith.fastmath<none>, predicate = 5}> : (f64, i32) -> i1
        "scf.yield"(%cmpf_246) : (i1) -> ()
      }, {
        %constv_247 = "arith.constant"() <{value = 0 : i1}> : () -> i1
        "scf.yield"(%constv_247) : (i1) -> ()
      }) : (i1) -> i1
      "scf.yield"(%if_243) : (i1) -> ()
    }) : (i1) -> i1
    "scf.yield"(%if_235) : (i1) -> ()
  }, {
    %constv_248 = "arith.constant"() <{value = 0 : i1}> : () -> i1
    "scf.yield"(%constv_248) : (i1) -> ()
  }) : (i1) -> i1
  %if_249 = "scf.if"(%if_210) ({
    %fieldnode_251 = sdql.access_record %l "l_extendedprice" : record<"l_orderkey": i32, "l_partkey": i32, "l_suppkey": i32, "l_linenumber": i32, "l_quantity": f64, "l_extendedprice": f64, "l_discount": f64, "l_tax": f64, "l_returnflag": memref<1xi8>, "l_linestatus": memref<1xi8>, "l_shipdate": i32, "l_commitdate": i32, "l_receiptdate": i32, "l_shipinstruct": memref<25xi8>, "l_shipmode": memref<10xi8>, "l_comment": memref<44xi8>> -> f64
    %constd_253 = "arith.constant"() <{value = 1.0  : f64}> : () -> f64
    %fieldnode_255 = sdql.access_record %l "l_discount" : record<"l_orderkey": i32, "l_partkey": i32, "l_suppkey": i32, "l_linenumber": i32, "l_quantity": f64, "l_extendedprice": f64, "l_discount": f64, "l_tax": f64, "l_returnflag": memref<1xi8>, "l_linestatus": memref<1xi8>, "l_shipdate": i32, "l_commitdate": i32, "l_receiptdate": i32, "l_shipinstruct": memref<25xi8>, "l_shipmode": memref<10xi8>, "l_comment": memref<44xi8>> -> f64
    %zero_256 = "arith.constant"() <{value = 0.0 : f64}> : () -> f64
    %negf_254 = "arith.subf"(%zero_256, %fieldnode_255) <{fastmath = #arith.fastmath<none>}> : (f64, f64) -> f64
    %addf_252 = "arith.addf"(%constd_253, %negf_254) <{fastmath = #arith.fastmath<none>}> : (f64, f64) -> f64
    %multf_250 = "arith.mulf"(%fieldnode_251, %addf_252) <{fastmath = #arith.fastmath<none>}> : (f64, f64) -> f64
    "scf.yield"(%multf_250) : (f64) -> ()
  }, {
    %constd_257 = "arith.constant"() <{value = 0.0  : f64}> : () -> f64
    "scf.yield"(%constd_257) : (f64) -> ()
  }) : (i1) -> f64
  sdql.yield %if_249 : f64
}
%recnode_259 = sdql.create_record {fields = ["revenue"]} %res : f64 -> record<"revenue": f64>
%consti_260 = "arith.constant"() <{value = 1 : i32}> : () -> i32
%dict_258 = sdql.create_dictionary %recnode_259, %consti_260 : record<"revenue" : f64>, i32 -> dictionary<record<"revenue" : f64>, i32>
