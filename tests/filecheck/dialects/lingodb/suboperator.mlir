// RUN: scair-opt %s | filecheck %s

"test.op"() ({
  %1 = "test.op"() : () -> !subop.result_table<[l_returnflag$0 : !db.char<1>, l_linestatus$0 : !db.char<1>]>
  subop.set_result 0 %1 : !subop.result_table<[l_returnflag$0 : !db.char<1>, l_linestatus$0 : !db.char<1>]>
}) : () -> ()

// CHECK: builtin.module {
// CHECK:   "test.op"() ({
// CHECK:     %0 = "test.op"() : () -> !subop.result_table<[l_returnflag$0 : !db.char<1>, l_linestatus$0 : !db.char<1>]>
// CHECK:     "subop.set_result"(%0) {result_id = 0} : (!subop.result_table<[l_returnflag$0 : !db.char<1>, l_linestatus$0 : !db.char<1>]>) -> ()
// CHECK:   }) : () -> ()
