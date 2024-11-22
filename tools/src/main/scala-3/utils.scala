package scair.utils

import scair.ir.Dialect
import scair.dialects.cmath.CMath
import scair.dialects.LingoDB.TupleStream.TupleStreamDialect
import scair.dialects.LingoDB.DBOps.DBOps
import scair.dialects.LingoDB.SubOperatorOps.SubOperatorOps
import scair.dialects.LingoDB.RelAlgOps.RelAlgOps
import scair.dialects.affine.Affine

val allDialects: Seq[Dialect] =
  Seq(CMath, TupleStreamDialect, DBOps, SubOperatorOps, RelAlgOps, Affine)

import scair.transformations.ModulePass
import scair.transformations.cdt.{
  DummyPass,
  TestInsertionPass,
  TestReplacementPass
}

val allPasses: Seq[ModulePass] =
  Seq(DummyPass, TestInsertionPass, TestReplacementPass)
