// RUN: scair-run %s | filecheck %s

builtin.module {
  func.func @main() -> (i32) {

    %c0 = "arith.constant"() <{value = 0 : i32}> : () -> i32

    %v1  = "arith.addi"(%c0, %c0) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
    %v2  = "arith.addi"(%c0, %c0) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
    %v3  = "arith.addi"(%c0, %c0) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
    %v4  = "arith.addi"(%c0, %c0) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
    %v5  = "arith.addi"(%c0, %c0) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
    %v6  = "arith.addi"(%c0, %c0) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
    %v7  = "arith.addi"(%c0, %c0) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
    %v8  = "arith.addi"(%c0, %c0) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
    %v9  = "arith.addi"(%c0, %c0) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
    %v10 = "arith.addi"(%c0, %c0) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32

    %v11 = "arith.addi"(%c0, %c0) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
    %v12 = "arith.addi"(%c0, %c0) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
    %v13 = "arith.addi"(%c0, %c0) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
    %v14 = "arith.addi"(%c0, %c0) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
    %v15 = "arith.addi"(%c0, %c0) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
    %v16 = "arith.addi"(%c0, %c0) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
    %v17 = "arith.addi"(%c0, %c0) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
    %v18 = "arith.addi"(%c0, %c0) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
    %v19 = "arith.addi"(%c0, %c0) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
    %v20 = "arith.addi"(%c0, %c0) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32

    %v21 = "arith.addi"(%c0, %c0) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
    %v22 = "arith.addi"(%c0, %c0) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
    %v23 = "arith.addi"(%c0, %c0) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
    %v24 = "arith.addi"(%c0, %c0) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
    %v25 = "arith.addi"(%c0, %c0) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
    %v26 = "arith.addi"(%c0, %c0) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
    %v27 = "arith.addi"(%c0, %c0) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
    %v28 = "arith.addi"(%c0, %c0) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
    %v29 = "arith.addi"(%c0, %c0) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
    %v30 = "arith.addi"(%c0, %c0) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32

    %v31 = "arith.addi"(%c0, %c0) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
    %v32 = "arith.addi"(%c0, %c0) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
    %v33 = "arith.addi"(%c0, %c0) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
    %v34 = "arith.addi"(%c0, %c0) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
    %v35 = "arith.addi"(%c0, %c0) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
    %v36 = "arith.addi"(%c0, %c0) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
    %v37 = "arith.addi"(%c0, %c0) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
    %v38 = "arith.addi"(%c0, %c0) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
    %v39 = "arith.addi"(%c0, %c0) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
    %v40 = "arith.addi"(%c0, %c0) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32

    %v41 = "arith.addi"(%c0, %c0) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
    %v42 = "arith.addi"(%c0, %c0) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
    %v43 = "arith.addi"(%c0, %c0) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
    %v44 = "arith.addi"(%c0, %c0) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
    %v45 = "arith.addi"(%c0, %c0) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
    %v46 = "arith.addi"(%c0, %c0) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
    %v47 = "arith.addi"(%c0, %c0) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
    %v48 = "arith.addi"(%c0, %c0) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
    %v49 = "arith.addi"(%c0, %c0) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
    %v50 = "arith.addi"(%c0, %c0) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32

    func.return %v1 : i32
  }
}

// CHECK: Result: