package scair.dialects

import scair.dialects.affine.AffineDialect
import scair.dialects.arith.ArithDialect
import scair.dialects.builtin.BuiltinDialect
import scair.dialects.cmath.cmath
import scair.dialects.complex.Complex
import scair.dialects.func.FuncDialect
import scair.dialects.irdl.IRDL
import scair.dialects.llvm.LLVMDialect
import scair.dialects.math.MathDialect
import scair.dialects.memref.MemrefDialect
import scair.dialects.scf.SCFDialect
import scair.dialects.test.Test
import scair.ir.Dialect

val allDialects: Seq[Dialect] =
  Seq(
    BuiltinDialect,
    Complex,
    MathDialect,
    Test,
    IRDL,
    ArithDialect,
    MemrefDialect,
    cmath,
    AffineDialect,
    FuncDialect,
    LLVMDialect,
    SCFDialect
  )
