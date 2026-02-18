// RUN: scair-run %s | filecheck %s

builtin.module {
  func.func @main() -> (i64) {
    
    %x = "arith.constant"() <{value = 4 : i64}> : () -> i64
    %c3 = "arith.constant"() <{value = 3 : i64}> : () -> i64
    %c2 = "arith.constant"() <{value = 2 : i64}> : () -> i64
    %c5 = "arith.constant"() <{value = 5 : i64}> : () -> i64
    %c7 = "arith.constant"() <{value = 7 : i64}> : () -> i64
    
    %x_sq = "arith.muli"(%x, %x) : (i64, i64) -> i64
    
    %x_cube = "arith.muli"(%x_sq, %x) : (i64, i64) -> i64
    
    %term1 = "arith.muli"(%c3, %x_cube) : (i64, i64) -> i64
    
    %term2 = "arith.muli"(%c2, %x_sq) : (i64, i64) -> i64
    
    %term3 = "arith.muli"(%c5, %x) : (i64, i64) -> i64
    
    %sum1 = "arith.addi"(%term1, %term2) : (i64, i64) -> i64
    
    %sum2 = "arith.subi"(%sum1, %term3) : (i64, i64) -> i64
    
    %result = "arith.addi"(%sum2, %c7) : (i64, i64) -> i64
    
    func.return %result : i64
  }
}

// CHECK: Result: 211