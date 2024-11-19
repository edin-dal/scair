package scair.utils

import scair.ir._
import scair.dialects.CMath.cmath.CMath
import scair.dialects.LingoDB.TupleStream.TupleStreamDialect
import scair.dialects.LingoDB.DBOps.DBOps
import scair.dialects.LingoDB.SubOperatorOps.SubOperatorOps
import scair.dialects.LingoDB.RelAlgOps.RelAlgOps
import scair.dialects.affine.Affine

val allDialects: Seq[Dialect] =
  Seq(CMath, TupleStreamDialect, DBOps, SubOperatorOps, RelAlgOps, Affine)
