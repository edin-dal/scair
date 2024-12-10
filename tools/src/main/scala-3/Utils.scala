package scair.utils

import scair.dialects.LingoDB.DBOps.DBOps
import scair.dialects.LingoDB.RelAlgOps.RelAlgOps
import scair.dialects.LingoDB.SubOperatorOps.SubOperatorOps
import scair.dialects.LingoDB.TupleStream.TupleStreamDialect
import scair.dialects._func._Func
import scair.dialects.affine.Affine
import scair.dialects.arith.Arith
import scair.dialects.cmath.CMath
import scair.dialects.memref.Memref
import scair.ir.Dialect

val allDialects: Seq[Dialect] =
  Seq(
    Arith,
    CMath,
    _Func,
    Memref,
    TupleStreamDialect,
    DBOps,
    SubOperatorOps,
    RelAlgOps,
    Affine
  )

import scair.transformations.ModulePass
import scair.transformations.cdt.{
  DummyPass,
  TestInsertionPass,
  TestReplacementPass
}

val allPasses: Seq[ModulePass] =
  Seq(DummyPass, TestInsertionPass, TestReplacementPass)
