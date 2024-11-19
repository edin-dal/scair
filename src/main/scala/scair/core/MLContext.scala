package scair

import scala.collection.mutable
import scair.ir._
import scair.utils.allDialects

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
