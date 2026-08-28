// RUN: scair-run %s | filecheck %s

builtin.module {
  func.func @double(%x: i32) -> i32 {
    %c2 = "arith.constant"() <{value = 2 : i32}> : () -> i32
    "llvm.br"(%x) [^body] : (i32) -> ()
  ^body(%v : i32):
    %res = "arith.muli"(%v, %c2) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
    "llvm.br"(%res) [^exit] : (i32) -> ()
  ^exit(%out : i32):
    func.return %out : i32
  }
  func.func @main() -> (i32) {
    %c21 = "arith.constant"() <{value = 21 : i32}> : () -> i32
    %c0 = "arith.constant"() <{value = 0 : i32}> : () -> i32
    %cond = "arith.cmpi"(%c21, %c0) <{"predicate" = 4 : i64}> : (i32, i32) -> i1
    "llvm.cond_br"(%cond) [^call, ^skip] <{operandSegmentSizes = array<i32: 1, 0, 0>}> : (i1) -> ()
  ^call:
    %d = "func.call"(%c21) <{"callee" = @double}> : (i32) -> i32
    "llvm.br"(%d) [^join] : (i32) -> ()
  ^skip:
    "llvm.br"(%c21) [^join] : (i32) -> ()
  ^join(%res : i32):
    func.return %res : i32
  }
}

// CHECK: Result: 42
