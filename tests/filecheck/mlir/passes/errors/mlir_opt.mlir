// RUN: scair-opt %s -p 'mlir-opt{arguments=--hello}' 2>&1 | filecheck %s --check-prefix=ARGS
// RUN: scair-opt %s -p 'mlir-opt{executable=false}' 2>&1 | filecheck %s --check-prefix=EXEC
// RUN: scair-opt %s -p 'mlir-opt{executable=/nonexistent}' 2>&1 | filecheck %s --check-prefix=MISSING
// RUN: scair-opt %s -p 'mlir-opt[not-a-pass]' 2>&1 | filecheck %s --check-prefix=NOPASS

%0 = "arith.constant"() <{value = 1 : i32}> : () -> i32
%1 = "arith.constant"() <{value = 1 : i32}> : () -> i32
%2 = "arith.addi"(%0, %1) : (i32, i32) -> i32
"test.op"(%2) : (i32) -> ()

// ARGS: error: cannot parse the output of 'mlir-opt --hello':

// EXEC: error: 'false' failed

// MISSING: error: cannot run '/nonexistent'

// NOPASS: 'not-a-pass' does not refer to a registered pass
// NOPASS: error: 'mlir-opt --mlir-print-op-generic --allow-unregistered-dialect -p builtin.module(not-a-pass)' failed
