// RUN: scair-opt %s | mlir-opt --mlir-print-op-generic | scair-opt | filecheck %s

// The `no_inline` unit flag round-trips through real mlir-opt. In custom syntax
// it rides in the attribute dictionary, as MLIR prints it.

"func.func"() <{"function_type" = () -> (), "sym_name" = "callee", "no_inline"}> ({
  "func.return"() : () -> ()
}) : () -> ()

"func.func"() <{"function_type" = () -> (), "sym_name" = "caller"}> ({
  "func.call"() <{"callee" = @callee, "no_inline"}> : () -> ()
  "func.return"() : () -> ()
}) : () -> ()

// CHECK:       builtin.module {
// CHECK-NEXT:    func.func @callee() attributes {no_inline} {
// CHECK-NEXT:      func.return
// CHECK-NEXT:    }
// CHECK-NEXT:    func.func @caller() {
// CHECK-NEXT:      "func.call"() <{callee = @callee, no_inline}> : () -> ()
// CHECK-NEXT:      func.return
// CHECK-NEXT:    }
// CHECK-NEXT:  }
