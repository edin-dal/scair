// RUN: scair-run %s | filecheck %s

builtin.module {
  func.func @main() -> i32 {
    %m0 = "memref.alloc"() <{"alignment" = 0 : i32, "operandSegmentSizes" = array<i32: 0, 0>}> : () -> memref<i32>
    %0 = "arith.constant"() <{value = 42 : i32}> : () -> i32
    "memref.store"(%0, %m0) : (i32, memref<i32>) -> ()
    %1 = "arith.constant"() <{value = 43 : i32}> : () -> i32
    "memref.store"(%1, %m0) : (i32, memref<i32>) -> ()
    %2 = "memref.load"(%m0) : (memref<i32>) -> i32
    func.return %2 : i32
  }
}

// CHECK: 43