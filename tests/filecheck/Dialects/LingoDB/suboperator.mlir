// RUN: scair-opt %s | filecheck %s

"op1"() ({
^bb0():
  %1 = "test.op"() : () -> (!subop.result_table<[l_returnflag$0 : !db.char<1>, l_linestatus$0 : !db.char<1>]>)
  subop.set_result 0 %1 : !subop.result_table<[l_returnflag$0 : !db.char<1>, l_linestatus$0 : !db.char<1>]>

// CHECK: "op1"() ({
// CHECK: ^bb0():
// CHECK:   %0 = "test.op"() : () -> (!subop.result_table<[l_returnflag$0 : !db.char<1>, l_linestatus$0 : !db.char<1>]>)
// CHECK:   "subop.set_result"(%0) {result_id = 0} : (!subop.result_table<[l_returnflag$0 : !db.char<1>, l_linestatus$0 : !db.char<1>]>) -> ()
// CHECK: }) : () -> ()

}) : () -> ()
