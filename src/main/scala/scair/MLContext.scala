package scair

import scair.dialects.cmath.CMath
import scair.dialects.LingoDB.TupleStream.TupleStreamDialect
import scair.dialects.LingoDB.DBOps.DBOps
import scair.dialects.LingoDB.SubOperatorOps.SubOperatorOps
import scala.collection.mutable
import scair._

val allDialects: Seq[Dialect] =
  Seq(CMath, TupleStreamDialect, DBOps, SubOperatorOps)

class MLContext() {

  val dialectOpContext: mutable.Map[String, DialectOperation] = mutable.Map()
  val dialectAttrContext: mutable.Map[String, DialectAttribute] = mutable.Map()

  for (dialect <- allDialects) {

    dialectOpContext ++= {
      for { dialectOp <- dialect.operations } yield {
        dialectOp.name -> dialectOp
      }
    }

    dialectAttrContext ++= {
      for { dialectAttr <- dialect.attributes } yield {
        dialectAttr.name -> dialectAttr
      }
    }
  }

  def getOperation(name: String) = dialectOpContext.get(name)

  def getAttribute(name: String) = dialectAttrContext.get(name)
}
