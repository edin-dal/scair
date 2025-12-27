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
%orders = sdql.load "datasets/tpch/orders.tbl" : record<"o_orderkey" : dictionary<i32, i32>, "o_custkey" : dictionary<i32, i32>, "o_orderstatus" : dictionary<i32, memref<1xi8>>, "o_totalprice" : dictionary<i32, f64>, "o_orderdate" : dictionary<i32, i32>, "o_orderpriority" : dictionary<i32, memref<15xi8>>, "o_clerk" : dictionary<i32, memref<15xi8>>, "o_shippriority" : dictionary<i32, i32>, "o_comment" : dictionary<i32, memref<79xi8>>, "size" : i32>
%customer = sdql.load "datasets/tpch/customer.tbl" : record<"c_custkey" : dictionary<i32, i32>, "c_name" : dictionary<i32, memref<25xi8>>, "c_address" : dictionary<i32, memref<40xi8>>, "c_nationkey" : dictionary<i32, i32>, "c_phone" : dictionary<i32, memref<15xi8>>, "c_acctbal" : dictionary<i32, f64>, "c_mktsegment" : dictionary<i32, memref<10xi8>>, "c_comment" : dictionary<i32, memref<117xi8>>, "size" : i32>
// special
%constds_0 = "arith.constant"() <{value = dense<[115, 112, 101, 99, 105, 97, 108]> : memref<?xi8>}> : () -> memref<?xi8>
%consti_1 = "arith.constant"() <{value = 8 : i32}> : () -> i32
%special = sdql.external "ConstantString", %constds_0, %consti_1 : memref<?xi8>, i32 -> memref<7xi8>
// requests
%constds_2 = "arith.constant"() <{value = dense<[114, 101, 113, 117, 101, 115, 116, 115]> : memref<?xi8>}> : () -> memref<?xi8>
%consti_3 = "arith.constant"() <{value = 9 : i32}> : () -> i32
%requests = sdql.external "ConstantString", %constds_2, %consti_3 : memref<?xi8>, i32 -> memref<8xi8>
%nchars_special = "arith.constant"() <{value = 7 : i32}> : () -> i32
%fieldnode_4 = sdql.access_record %orders "size" : record<"o_orderkey": dictionary<i32, i32>, "o_custkey": dictionary<i32, i32>, "o_orderstatus": dictionary<i32, memref<1xi8>>, "o_totalprice": dictionary<i32, f64>, "o_orderdate": dictionary<i32, i32>, "o_orderpriority": dictionary<i32, memref<15xi8>>, "o_clerk": dictionary<i32, memref<15xi8>>, "o_shippriority": dictionary<i32, i32>, "o_comment": dictionary<i32, memref<79xi8>>, "size": i32> -> i32
%rangenode_5 = "func.call"(%fieldnode_4) <{callee = @range_builtin}> : (i32) -> dictionary<i32, i32>
%o_h = sdql.sum %rangenode_5 : dictionary<i32, i32> -> dictionary<i32, i32> {
^bb_6(%i: i32, %_: i32):
  %fieldnode_8 = sdql.access_record %orders "o_orderkey" : record<"o_orderkey": dictionary<i32, i32>, "o_custkey": dictionary<i32, i32>, "o_orderstatus": dictionary<i32, memref<1xi8>>, "o_totalprice": dictionary<i32, f64>, "o_orderdate": dictionary<i32, i32>, "o_orderpriority": dictionary<i32, memref<15xi8>>, "o_clerk": dictionary<i32, memref<15xi8>>, "o_shippriority": dictionary<i32, i32>, "o_comment": dictionary<i32, memref<79xi8>>, "size": i32> -> dictionary<i32, i32>
  %get_7 = sdql.lookup_dictionary %fieldnode_8 [%i : i32] : dictionary<i32, i32> -> i32
  %fieldnode_10 = sdql.access_record %orders "o_custkey" : record<"o_orderkey": dictionary<i32, i32>, "o_custkey": dictionary<i32, i32>, "o_orderstatus": dictionary<i32, memref<1xi8>>, "o_totalprice": dictionary<i32, f64>, "o_orderdate": dictionary<i32, i32>, "o_orderpriority": dictionary<i32, memref<15xi8>>, "o_clerk": dictionary<i32, memref<15xi8>>, "o_shippriority": dictionary<i32, i32>, "o_comment": dictionary<i32, memref<79xi8>>, "size": i32> -> dictionary<i32, i32>
  %get_9 = sdql.lookup_dictionary %fieldnode_10 [%i : i32] : dictionary<i32, i32> -> i32
  %fieldnode_12 = sdql.access_record %orders "o_orderstatus" : record<"o_orderkey": dictionary<i32, i32>, "o_custkey": dictionary<i32, i32>, "o_orderstatus": dictionary<i32, memref<1xi8>>, "o_totalprice": dictionary<i32, f64>, "o_orderdate": dictionary<i32, i32>, "o_orderpriority": dictionary<i32, memref<15xi8>>, "o_clerk": dictionary<i32, memref<15xi8>>, "o_shippriority": dictionary<i32, i32>, "o_comment": dictionary<i32, memref<79xi8>>, "size": i32> -> dictionary<i32, memref<1xi8>>
  %get_11 = sdql.lookup_dictionary %fieldnode_12 [%i : i32] : dictionary<i32, memref<1xi8>> -> memref<1xi8>
  %fieldnode_14 = sdql.access_record %orders "o_totalprice" : record<"o_orderkey": dictionary<i32, i32>, "o_custkey": dictionary<i32, i32>, "o_orderstatus": dictionary<i32, memref<1xi8>>, "o_totalprice": dictionary<i32, f64>, "o_orderdate": dictionary<i32, i32>, "o_orderpriority": dictionary<i32, memref<15xi8>>, "o_clerk": dictionary<i32, memref<15xi8>>, "o_shippriority": dictionary<i32, i32>, "o_comment": dictionary<i32, memref<79xi8>>, "size": i32> -> dictionary<i32, f64>
  %get_13 = sdql.lookup_dictionary %fieldnode_14 [%i : i32] : dictionary<i32, f64> -> f64
  %fieldnode_16 = sdql.access_record %orders "o_orderdate" : record<"o_orderkey": dictionary<i32, i32>, "o_custkey": dictionary<i32, i32>, "o_orderstatus": dictionary<i32, memref<1xi8>>, "o_totalprice": dictionary<i32, f64>, "o_orderdate": dictionary<i32, i32>, "o_orderpriority": dictionary<i32, memref<15xi8>>, "o_clerk": dictionary<i32, memref<15xi8>>, "o_shippriority": dictionary<i32, i32>, "o_comment": dictionary<i32, memref<79xi8>>, "size": i32> -> dictionary<i32, i32>
  %get_15 = sdql.lookup_dictionary %fieldnode_16 [%i : i32] : dictionary<i32, i32> -> i32
  %fieldnode_18 = sdql.access_record %orders "o_orderpriority" : record<"o_orderkey": dictionary<i32, i32>, "o_custkey": dictionary<i32, i32>, "o_orderstatus": dictionary<i32, memref<1xi8>>, "o_totalprice": dictionary<i32, f64>, "o_orderdate": dictionary<i32, i32>, "o_orderpriority": dictionary<i32, memref<15xi8>>, "o_clerk": dictionary<i32, memref<15xi8>>, "o_shippriority": dictionary<i32, i32>, "o_comment": dictionary<i32, memref<79xi8>>, "size": i32> -> dictionary<i32, memref<15xi8>>
  %get_17 = sdql.lookup_dictionary %fieldnode_18 [%i : i32] : dictionary<i32, memref<15xi8>> -> memref<15xi8>
  %fieldnode_20 = sdql.access_record %orders "o_clerk" : record<"o_orderkey": dictionary<i32, i32>, "o_custkey": dictionary<i32, i32>, "o_orderstatus": dictionary<i32, memref<1xi8>>, "o_totalprice": dictionary<i32, f64>, "o_orderdate": dictionary<i32, i32>, "o_orderpriority": dictionary<i32, memref<15xi8>>, "o_clerk": dictionary<i32, memref<15xi8>>, "o_shippriority": dictionary<i32, i32>, "o_comment": dictionary<i32, memref<79xi8>>, "size": i32> -> dictionary<i32, memref<15xi8>>
  %get_19 = sdql.lookup_dictionary %fieldnode_20 [%i : i32] : dictionary<i32, memref<15xi8>> -> memref<15xi8>
  %fieldnode_22 = sdql.access_record %orders "o_shippriority" : record<"o_orderkey": dictionary<i32, i32>, "o_custkey": dictionary<i32, i32>, "o_orderstatus": dictionary<i32, memref<1xi8>>, "o_totalprice": dictionary<i32, f64>, "o_orderdate": dictionary<i32, i32>, "o_orderpriority": dictionary<i32, memref<15xi8>>, "o_clerk": dictionary<i32, memref<15xi8>>, "o_shippriority": dictionary<i32, i32>, "o_comment": dictionary<i32, memref<79xi8>>, "size": i32> -> dictionary<i32, i32>
  %get_21 = sdql.lookup_dictionary %fieldnode_22 [%i : i32] : dictionary<i32, i32> -> i32
  %fieldnode_24 = sdql.access_record %orders "o_comment" : record<"o_orderkey": dictionary<i32, i32>, "o_custkey": dictionary<i32, i32>, "o_orderstatus": dictionary<i32, memref<1xi8>>, "o_totalprice": dictionary<i32, f64>, "o_orderdate": dictionary<i32, i32>, "o_orderpriority": dictionary<i32, memref<15xi8>>, "o_clerk": dictionary<i32, memref<15xi8>>, "o_shippriority": dictionary<i32, i32>, "o_comment": dictionary<i32, memref<79xi8>>, "size": i32> -> dictionary<i32, memref<79xi8>>
  %get_23 = sdql.lookup_dictionary %fieldnode_24 [%i : i32] : dictionary<i32, memref<79xi8>> -> memref<79xi8>
  %o = sdql.create_record {fields = ["o_orderkey", "o_custkey", "o_orderstatus", "o_totalprice", "o_orderdate", "o_orderpriority", "o_clerk", "o_shippriority", "o_comment"]} %get_7, %get_9, %get_11, %get_13, %get_15, %get_17, %get_19, %get_21, %get_23 : i32, i32, memref<1xi8>, f64, i32, memref<15xi8>, memref<15xi8>, i32, memref<79xi8> -> record<"o_orderkey": i32, "o_custkey": i32, "o_orderstatus": memref<1xi8>, "o_totalprice": f64, "o_orderdate": i32, "o_orderpriority": memref<15xi8>, "o_clerk": memref<15xi8>, "o_shippriority": i32, "o_comment": memref<79xi8>>
  %fieldnode_25 = sdql.access_record %o "o_comment" : record<"o_orderkey": i32, "o_custkey": i32, "o_orderstatus": memref<1xi8>, "o_totalprice": f64, "o_orderdate": i32, "o_orderpriority": memref<15xi8>, "o_clerk": memref<15xi8>, "o_shippriority": i32, "o_comment": memref<79xi8>> -> memref<79xi8>
  %idx_special = sdql.external "FirstIndex", %fieldnode_25, %special : memref<79xi8>, memref<7xi8> -> i32
  %consti_26 = "arith.constant"() <{value = -1 : i32}> : () -> i32
  %cmpi_27 = "arith.cmpi"(%idx_special, %consti_26) <{predicate = 0}> : (i32, i32) -> i1
  %if_28 = "scf.if"(%cmpi_27) ({
    %constv_29 = "arith.constant"() <{value = 1 : i1}> : () -> i1
    "scf.yield"(%constv_29) : (i1) -> ()
  }, {
    %fieldnode_30 = sdql.access_record %o "o_comment" : record<"o_orderkey": i32, "o_custkey": i32, "o_orderstatus": memref<1xi8>, "o_totalprice": f64, "o_orderdate": i32, "o_orderpriority": memref<15xi8>, "o_clerk": memref<15xi8>, "o_shippriority": i32, "o_comment": memref<79xi8>> -> memref<79xi8>
    %external_31 = sdql.external "LastIndex", %fieldnode_30, %requests : memref<79xi8>, memref<8xi8> -> i32
    %addi_32 = "arith.addi"(%idx_special, %nchars_special) : (i32, i32) -> i32
    %cmpi_33 = "arith.cmpi"(%external_31, %addi_32) <{predicate = 2}> : (i32, i32) -> i1
    "scf.yield"(%cmpi_33) : (i1) -> ()
  }) : (i1) -> i1
  %if_34 = "scf.if"(%if_28) ({
    %fieldnode_36 = sdql.access_record %o "o_custkey" : record<"o_orderkey": i32, "o_custkey": i32, "o_orderstatus": memref<1xi8>, "o_totalprice": f64, "o_orderdate": i32, "o_orderpriority": memref<15xi8>, "o_clerk": memref<15xi8>, "o_shippriority": i32, "o_comment": memref<79xi8>> -> i32
    %consti_37 = "arith.constant"() <{value = 1 : i32}> : () -> i32
    %dict_35 = sdql.create_dictionary %fieldnode_36, %consti_37 : i32, i32 -> dictionary<i32, i32>
    "scf.yield"(%dict_35) : (dictionary<i32, i32>) -> ()
  }, {
    %dict_38 = sdql.empty_dictionary : dictionary<i32, i32>
    "scf.yield"(%dict_38) : (dictionary<i32, i32>) -> ()
  }) : (i1) -> dictionary<i32, i32>
  sdql.yield %if_34 : dictionary<i32, i32>
}
%fieldnode_39 = sdql.access_record %customer "size" : record<"c_custkey": dictionary<i32, i32>, "c_name": dictionary<i32, memref<25xi8>>, "c_address": dictionary<i32, memref<40xi8>>, "c_nationkey": dictionary<i32, i32>, "c_phone": dictionary<i32, memref<15xi8>>, "c_acctbal": dictionary<i32, f64>, "c_mktsegment": dictionary<i32, memref<10xi8>>, "c_comment": dictionary<i32, memref<117xi8>>, "size": i32> -> i32
%rangenode_40 = "func.call"(%fieldnode_39) <{callee = @range_builtin}> : (i32) -> dictionary<i32, i32>
%c_h = sdql.sum %rangenode_40 : dictionary<i32, i32> -> dictionary<record<"custkey" : i32>, record<"count" : i32>> {
^bb_41(%i: i32, %_: i32):
  %fieldnode_43 = sdql.access_record %customer "c_custkey" : record<"c_custkey": dictionary<i32, i32>, "c_name": dictionary<i32, memref<25xi8>>, "c_address": dictionary<i32, memref<40xi8>>, "c_nationkey": dictionary<i32, i32>, "c_phone": dictionary<i32, memref<15xi8>>, "c_acctbal": dictionary<i32, f64>, "c_mktsegment": dictionary<i32, memref<10xi8>>, "c_comment": dictionary<i32, memref<117xi8>>, "size": i32> -> dictionary<i32, i32>
  %get_42 = sdql.lookup_dictionary %fieldnode_43 [%i : i32] : dictionary<i32, i32> -> i32
  %fieldnode_45 = sdql.access_record %customer "c_name" : record<"c_custkey": dictionary<i32, i32>, "c_name": dictionary<i32, memref<25xi8>>, "c_address": dictionary<i32, memref<40xi8>>, "c_nationkey": dictionary<i32, i32>, "c_phone": dictionary<i32, memref<15xi8>>, "c_acctbal": dictionary<i32, f64>, "c_mktsegment": dictionary<i32, memref<10xi8>>, "c_comment": dictionary<i32, memref<117xi8>>, "size": i32> -> dictionary<i32, memref<25xi8>>
  %get_44 = sdql.lookup_dictionary %fieldnode_45 [%i : i32] : dictionary<i32, memref<25xi8>> -> memref<25xi8>
  %fieldnode_47 = sdql.access_record %customer "c_address" : record<"c_custkey": dictionary<i32, i32>, "c_name": dictionary<i32, memref<25xi8>>, "c_address": dictionary<i32, memref<40xi8>>, "c_nationkey": dictionary<i32, i32>, "c_phone": dictionary<i32, memref<15xi8>>, "c_acctbal": dictionary<i32, f64>, "c_mktsegment": dictionary<i32, memref<10xi8>>, "c_comment": dictionary<i32, memref<117xi8>>, "size": i32> -> dictionary<i32, memref<40xi8>>
  %get_46 = sdql.lookup_dictionary %fieldnode_47 [%i : i32] : dictionary<i32, memref<40xi8>> -> memref<40xi8>
  %fieldnode_49 = sdql.access_record %customer "c_nationkey" : record<"c_custkey": dictionary<i32, i32>, "c_name": dictionary<i32, memref<25xi8>>, "c_address": dictionary<i32, memref<40xi8>>, "c_nationkey": dictionary<i32, i32>, "c_phone": dictionary<i32, memref<15xi8>>, "c_acctbal": dictionary<i32, f64>, "c_mktsegment": dictionary<i32, memref<10xi8>>, "c_comment": dictionary<i32, memref<117xi8>>, "size": i32> -> dictionary<i32, i32>
  %get_48 = sdql.lookup_dictionary %fieldnode_49 [%i : i32] : dictionary<i32, i32> -> i32
  %fieldnode_51 = sdql.access_record %customer "c_phone" : record<"c_custkey": dictionary<i32, i32>, "c_name": dictionary<i32, memref<25xi8>>, "c_address": dictionary<i32, memref<40xi8>>, "c_nationkey": dictionary<i32, i32>, "c_phone": dictionary<i32, memref<15xi8>>, "c_acctbal": dictionary<i32, f64>, "c_mktsegment": dictionary<i32, memref<10xi8>>, "c_comment": dictionary<i32, memref<117xi8>>, "size": i32> -> dictionary<i32, memref<15xi8>>
  %get_50 = sdql.lookup_dictionary %fieldnode_51 [%i : i32] : dictionary<i32, memref<15xi8>> -> memref<15xi8>
  %fieldnode_53 = sdql.access_record %customer "c_acctbal" : record<"c_custkey": dictionary<i32, i32>, "c_name": dictionary<i32, memref<25xi8>>, "c_address": dictionary<i32, memref<40xi8>>, "c_nationkey": dictionary<i32, i32>, "c_phone": dictionary<i32, memref<15xi8>>, "c_acctbal": dictionary<i32, f64>, "c_mktsegment": dictionary<i32, memref<10xi8>>, "c_comment": dictionary<i32, memref<117xi8>>, "size": i32> -> dictionary<i32, f64>
  %get_52 = sdql.lookup_dictionary %fieldnode_53 [%i : i32] : dictionary<i32, f64> -> f64
  %fieldnode_55 = sdql.access_record %customer "c_mktsegment" : record<"c_custkey": dictionary<i32, i32>, "c_name": dictionary<i32, memref<25xi8>>, "c_address": dictionary<i32, memref<40xi8>>, "c_nationkey": dictionary<i32, i32>, "c_phone": dictionary<i32, memref<15xi8>>, "c_acctbal": dictionary<i32, f64>, "c_mktsegment": dictionary<i32, memref<10xi8>>, "c_comment": dictionary<i32, memref<117xi8>>, "size": i32> -> dictionary<i32, memref<10xi8>>
  %get_54 = sdql.lookup_dictionary %fieldnode_55 [%i : i32] : dictionary<i32, memref<10xi8>> -> memref<10xi8>
  %fieldnode_57 = sdql.access_record %customer "c_comment" : record<"c_custkey": dictionary<i32, i32>, "c_name": dictionary<i32, memref<25xi8>>, "c_address": dictionary<i32, memref<40xi8>>, "c_nationkey": dictionary<i32, i32>, "c_phone": dictionary<i32, memref<15xi8>>, "c_acctbal": dictionary<i32, f64>, "c_mktsegment": dictionary<i32, memref<10xi8>>, "c_comment": dictionary<i32, memref<117xi8>>, "size": i32> -> dictionary<i32, memref<117xi8>>
  %get_56 = sdql.lookup_dictionary %fieldnode_57 [%i : i32] : dictionary<i32, memref<117xi8>> -> memref<117xi8>
  %c = sdql.create_record {fields = ["c_custkey", "c_name", "c_address", "c_nationkey", "c_phone", "c_acctbal", "c_mktsegment", "c_comment"]} %get_42, %get_44, %get_46, %get_48, %get_50, %get_52, %get_54, %get_56 : i32, memref<25xi8>, memref<40xi8>, i32, memref<15xi8>, f64, memref<10xi8>, memref<117xi8> -> record<"c_custkey": i32, "c_name": memref<25xi8>, "c_address": memref<40xi8>, "c_nationkey": i32, "c_phone": memref<15xi8>, "c_acctbal": f64, "c_mktsegment": memref<10xi8>, "c_comment": memref<117xi8>>
  %fieldnode_60 = sdql.access_record %c "c_custkey" : record<"c_custkey": i32, "c_name": memref<25xi8>, "c_address": memref<40xi8>, "c_nationkey": i32, "c_phone": memref<15xi8>, "c_acctbal": f64, "c_mktsegment": memref<10xi8>, "c_comment": memref<117xi8>> -> i32
  %get_59 = sdql.lookup_dictionary %o_h [%fieldnode_60 : i32] : dictionary<i32, i32> -> i32
  %dict_61 = sdql.empty_dictionary : i32
  %cmpi_62 = "arith.cmpi"(%get_59, %dict_61) <{predicate = 1}> : (i32, i32) -> i1
  %if_63 = "scf.if"(%cmpi_62) ({
    %fieldnode_65 = sdql.access_record %c "c_custkey" : record<"c_custkey": i32, "c_name": memref<25xi8>, "c_address": memref<40xi8>, "c_nationkey": i32, "c_phone": memref<15xi8>, "c_acctbal": f64, "c_mktsegment": memref<10xi8>, "c_comment": memref<117xi8>> -> i32
    %get_64 = sdql.lookup_dictionary %o_h [%fieldnode_65 : i32] : dictionary<i32, i32> -> i32
    "scf.yield"(%get_64) : (i32) -> ()
  }, {
    %consti_66 = "arith.constant"() <{value = 0 : i32}> : () -> i32
    "scf.yield"(%consti_66) : (i32) -> ()
  }) : (i1) -> i32
  %recnode_67 = sdql.create_record {fields = ["custkey"]} %if_63 : i32 -> record<"custkey": i32>
  %consti_68 = "arith.constant"() <{value = 1 : i32}> : () -> i32
  %recnode_69 = sdql.create_record {fields = ["count"]} %consti_68 : i32 -> record<"count": i32>
  %dict_58 = sdql.create_dictionary %recnode_67, %recnode_69 : record<"custkey" : i32>, record<"count" : i32> -> dictionary<record<"custkey" : i32>, record<"count" : i32>>
  sdql.yield %dict_58 : dictionary<record<"custkey" : i32>, record<"count" : i32>>
}
%sum_70 = sdql.sum %c_h : dictionary<record<"custkey" : i32>, record<"count" : i32>> -> dictionary<record<"custkey" : i32, "count" : i32>, i32> {
^bb_71(%k: record<"custkey" : i32>, %v: record<"count" : i32>):
  %concat_73 = sdql.concat %k, %v : record<"custkey" : i32>, record<"count" : i32> -> record<"custkey" : i32, "count" : i32>
  %consti_74 = "arith.constant"() <{value = 1 : i32}> : () -> i32
  %dict_72 = sdql.create_dictionary %concat_73, %consti_74 : record<"custkey" : i32, "count" : i32>, i32 -> dictionary<record<"custkey" : i32, "count" : i32>, i32>
  sdql.yield %dict_72 : dictionary<record<"custkey" : i32, "count" : i32>, i32>
}
