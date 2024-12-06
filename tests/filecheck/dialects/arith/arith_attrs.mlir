// RUN: scair-opt %s | filecheck %s

"test.op"() {"none" = #arith.fastmath<none>} : () -> ()
// CHECK:    "test.op"() {none = arith.fastmath<none>} : () -> ()
"test.op"() {"fast" = #arith.fastmath<fast>} : () -> ()
// CHECK:    "test.op"() {fast = arith.fastmath<fast>} : () -> ()
"test.op"() {"stuff" = #arith.fastmath<nnan,arcp>} : () -> ()
// CHECK:    "test.op"() {stuff = arith.fastmath<arcp,nnan>} : () -> ()
"test.op"() {"faststuff" = #arith.fastmath<fast,nnan,arcp>} : () -> ()
// CHECK:    "test.op"() {faststuff = arith.fastmath<fast>} : () -> ()
"test.op"() {"nonestuff" = #arith.fastmath<none,nnan,arcp>} : () -> ()
// CHECK:    "test.op"() {nonestuff = arith.fastmath<arcp,nnan>} : () -> ()