package scair.utils

import scair.dialects.LingoDB.DBOps.DBOps
import scair.dialects.LingoDB.RelAlgOps.RelAlgOps
import scair.dialects.LingoDB.SubOperatorOps.SubOperatorOps
import scair.dialects.LingoDB.TupleStream.TupleStreamDialect
import scair.dialects.affine.AffineDialect
import scair.dialects.arith.ArithDialect
import scair.dialects.builtin.BuiltinDialect
import scair.dialects.cmath.cmath
import scair.dialects.samplemath.samplemath
import scair.dialects.func.FuncDialect
import scair.dialects.irdl.IRDL
import scair.dialects.llvm.LLVMDialect
import scair.dialects.math.MathDialect
import scair.dialects.memref.MemrefDialect
import scair.dialects.scf.SCFDialect
import scair.dialects.test.Test
import scair.dialects.samplecnstr.samplecnstr
import scair.ir.Dialect
import scair.transformations.ModulePass
import scair.transformations.benchmark_constant_folding.BenchmarkConstantFolding
import scair.transformations.canonicalization.Canonicalize
import scair.transformations.cdt.DummyPass
import scair.transformations.cdt.TestInsertionPass
import scair.transformations.cdt.TestReplacementPass
import scair.transformations.cse.CommonSubexpressionElimination
import scair.transformations.reconcile.ReconcileUnrealizedCasts
import scair.transformations.samplemathcanon.SampleMathCanon

val allDialects: Seq[Dialect] =
  Seq(
    BuiltinDialect,
    MathDialect,
    TupleStreamDialect,
    DBOps,
    SubOperatorOps,
    RelAlgOps,
    Test,
    // Clair
    IRDL,
    ArithDialect,
    MemrefDialect,
    cmath,
    samplemath,
    samplecnstr,
    AffineDialect,
    FuncDialect,
    LLVMDialect,
    SCFDialect
  )

val allPasses: Seq[ModulePass] =
  Seq(
    BenchmarkConstantFolding,
    CommonSubexpressionElimination,
    DummyPass,
    ReconcileUnrealizedCasts,
    TestInsertionPass,
    TestReplacementPass,
    Canonicalize,
    SampleMathCanon
  )
