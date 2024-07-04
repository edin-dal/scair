package scair

import scair.dialects.cmath.CMath
import scala.collection.mutable
import scair._

val allDialects: Seq[Dialect] = Seq(CMath)

class MLContext() {

  var dialectOpContext: Map[String, DialectOperation] = Map()
  var dialectAttrContext: Map[String, DialectAttribute] = Map()

  for (dialect <- allDialects) {

    dialectOpContext = {
      for { dialectOp <- dialect.operations } yield {
        dialectOp.name -> dialectOp
      }
    }.toMap

    dialectAttrContext = {
      for { dialectAttr <- dialect.attributes } yield {
        dialectAttr.name -> dialectAttr
      }
    }.toMap
  }

  def getOperation(name: String) = dialectOpContext.get(name)

  def getAttribute(name: String) = dialectAttrContext.get(name)
}
