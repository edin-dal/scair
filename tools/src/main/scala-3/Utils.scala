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
import scair.dialects.memref.MemrefDialect
import scair.dialects.test.Test
import scair.ir.Dialect

val allDialects: Seq[Dialect] =
  Seq(
    ArithDialect,
    BuiltinDialect,
    CMathDialect,
    FuncDialect,
    LLVMDialect,
    MemrefDialect,
    TupleStreamDialect,
    DBOps,
    SubOperatorOps,
    RelAlgOps,
    AffineDialect,
    Test
  )

import scair.transformations.ModulePass
import scair.transformations.cdt.{
  DummyPass,
  TestInsertionPass,
  TestReplacementPass
}

val allPasses: Seq[ModulePass] =
  Seq(DummyPass, TestInsertionPass, TestReplacementPass)
