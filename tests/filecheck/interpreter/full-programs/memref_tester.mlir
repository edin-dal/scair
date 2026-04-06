// RUN: scair-run %s | filecheck %s

builtin.module {
  func.func @main() -> (i32) {

    %lb   = "arith.constant"() <{value = 0 : i32}> : () -> i32
    %ub   = "arith.constant"() <{value = 100000 : i32}> : () -> i32
    %step = "arith.constant"() <{value = 1 : i32}> : () -> i32

    %one = "arith.constant"() <{value = 1 : i32}> : () -> i32

    // allocate 1-element memory buffer
    %buf = "memref.alloc"() <{"alignment" = 1 : i32, "operandSegmentSizes" = array<i32: 0, 0>}> : () -> memref<i32>

    // initialise buffer[0] = 1
    "memref.store"(%one, %buf) : (i32, memref<i32>) -> ()

    "scf.for"(%lb, %ub, %step) ({
    ^bb0(%iv: i32):

      // load current value
      %val = "memref.load"(%buf) : (memref<i32>) -> i32

      %new = "arith.addi"(%val, %one)
        <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32

      // store back to memory
      "memref.store"(%new, %buf) : (i32, memref<i32>) -> ()
      "scf.yield"() : () -> ()

    }) : (i32, i32, i32) -> ()

    // final result = last stored value
    %final = "memref.load"(%buf) : (memref<i32>) -> i32

    func.return %final : i32
  }
}

// CHECK: Result: