package scair.utils

import scair.MLContext
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
import scair.transformations.ModulePass
import scair.transformations.benchmark_constant_folding.BenchmarkConstantFolding
import scair.transformations.canonicalization.Canonicalize
import scair.transformations.cdt.DummyPass
import scair.transformations.cdt.TestInsertionPass
import scair.transformations.cdt.TestReplacementPass
import scair.transformations.cse.CommonSubexpressionElimination
import scair.transformations.reconcile.ReconcileUnrealizedCasts

val allDialects: Seq[Dialect] =
  Seq(
    BuiltinDialect,
    Complex,
    MathDialect,
    Test,
    // Clair
    IRDL,
    ArithDialect,
    MemrefDialect,
    cmath,
    AffineDialect,
    FuncDialect,
    LLVMDialect,
    SCFDialect
  )

val allPasses: Seq[MLContext => ModulePass] =
  Seq(
    BenchmarkConstantFolding(_),
    CommonSubexpressionElimination(_),
    DummyPass(_),
    ReconcileUnrealizedCasts(_),
    TestInsertionPass(_),
    TestReplacementPass(_),
    Canonicalize(_)
  )
