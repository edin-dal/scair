package scair.utils

import scair.dialects.LingoDB.DBOps.DBOps
import scair.dialects.LingoDB.RelAlgOps.RelAlgOps
import scair.dialects.LingoDB.SubOperatorOps.SubOperatorOps
import scair.dialects.LingoDB.TupleStream.TupleStreamDialect
import scair.dialects.affine.AffineDialect
import scair.dialects.arith.ArithDialect
import scair.dialects.builtin.BuiltinDialect
import scair.dialects.cmath.CMathDialect
import scair.dialects.func.FuncDialect
import scair.dialects.llvm.LLVMDialect
import scair.dialects.math.MathDialect
import scair.dialects.memref.MemrefDialect
import scair.dialects.scf.SCFDialect
import scair.dialects.test.Test
import scair.ir.Dialect
import scair.transformations.ModulePass
import scair.transformations.cdt.DummyPass
import scair.transformations.cdt.TestInsertionPass
import scair.transformations.cdt.TestReplacementPass

val allDialects: Seq[Dialect] = Seq(
  BuiltinDialect,
  MathDialect,
  TupleStreamDialect,
  DBOps,
  SubOperatorOps,
  RelAlgOps,
  Test,
  // Clair
  ArithDialect,
  MemrefDialect,
  CMathDialect,
  AffineDialect,
  FuncDialect,
  LLVMDialect,
  SCFDialect
)

val allPasses: Seq[ModulePass] =
  Seq(DummyPass, TestInsertionPass, TestReplacementPass)
