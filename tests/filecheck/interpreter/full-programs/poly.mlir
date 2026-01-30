// RUN: scair-run %s | filecheck %s

builtin.module {
  func.func @main() -> (i64) {
    // Polynomial: 3x³ + 2x² - 5x + 7 at x=4
    // = 3(64) + 2(16) - 5(4) + 7
    // = 192 + 32 - 20 + 7
    // = 211
    
    %x = "arith.constant"() <{value = 4 : i64}> : () -> i64
    %c3 = "arith.constant"() <{value = 3 : i64}> : () -> i64
    %c2 = "arith.constant"() <{value = 2 : i64}> : () -> i64
    %c5 = "arith.constant"() <{value = 5 : i64}> : () -> i64
    %c7 = "arith.constant"() <{value = 7 : i64}> : () -> i64
    
    // x² = 16
    %x_sq = "arith.muli"(%x, %x) : (i64, i64) -> i64
    
    // x³ = 64
    %x_cube = "arith.muli"(%x_sq, %x) : (i64, i64) -> i64
    
    // 3x³ = 192
    %term1 = "arith.muli"(%c3, %x_cube) : (i64, i64) -> i64
    
    // 2x² = 32
    %term2 = "arith.muli"(%c2, %x_sq) : (i64, i64) -> i64
    
    // 5x = 20
    %term3 = "arith.muli"(%c5, %x) : (i64, i64) -> i64
    
    // 3x³ + 2x² = 224
    %sum1 = "arith.addi"(%term1, %term2) : (i64, i64) -> i64
    
    // - 5x = 224 - 20 = 204
    %sum2 = "arith.subi"(%sum1, %term3) : (i64, i64) -> i64
    
    // + 7 = 211
    %result = "arith.addi"(%sum2, %c7) : (i64, i64) -> i64
    
    func.return %result : i64
  }
}

// CHECK: 211